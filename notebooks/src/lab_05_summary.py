# %% [markdown]
# # ЛР2 · lab_05 — сводка: сравнение, выбор, holdout, применение, ошибки, гипотезы (CPU)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA`, выводы **всех** ноутбуков: `lab-00-data`, `lab-01-embed`, `lab-02-gbm`,
# `lab-03-neural`, `lab-04-vlm-qwen3b`, `lab-04-vlm-qwen7b`, `lab-04-vlm-internvl`.
# **Accelerator:** нет. Время: ~10 мин.
#
# Задания 3–7: единая таблица, E-1200 для всех моделей, правило выбора (протокол, разд. 10),
# подтверждение на holdout, применение, анализ ошибок, проверка гипотез H1–H5.

# %%
import os, sys, glob, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
def _find_labkit():
    """labkit.py лежит в выводе ноутбука lab-00-data; ищем во всех подключённых Inputs."""
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if "labkit.py" in fn:
                return dp
    tree = []
    for dp, dn, fn in os.walk("/kaggle/input", followlinks=True):
        if dp.count(os.sep) <= 5:
            tree.append(dp)
    raise FileNotFoundError("labkit.py не найден: подключите вывод ноутбука lab-00-data "
                            "(Add Input → Your Work → Notebooks → lab-00-data). Сейчас в /kaggle/input:\n"
                            + "\n".join(tree[:60]))


sys.path.insert(0, _find_labkit())
print("labkit:", sys.path[0])
import labkit as lk
OUT = lk.results_dir(); FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
T0 = time.time()


def find_all(pattern):
    import fnmatch
    hits = []
    for root in ["/kaggle/input", "results"]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            hits += [os.path.join(dp, x) for x in fn if fnmatch.fnmatch(x, pattern)]
        if hits:
            break
    return sorted(set(hits), key=lambda p: Path(p).name)


O = {p: lk.load_objects(p) for p in ["val", "test", "holdout"]}
A = lk.load_alerts_clean()
UNI = ["U1_num", "U2_text", "U3_image"]
TRAINED = ["U0_persistence", "U1_num", "U2_text", "U3_image", "M1_num_txtstruct", "M2_early_all", "M3_late_stack",
           "M4_gated", "M5_crossattn"]
VLM = ["M6_qwen25vl_3b", "M7_qwen25vl_7b", "M8_internvl3_2b"]


def pred(model, part):
    f = find_all(f"{model}__{part}.parquet")
    if not f:
        return None
    x = pd.read_parquet(f[0])
    assert np.array_equal(x["station"].to_numpy(), O[part]["station"].to_numpy()), model
    return x["p"].to_numpy()


P = {(m, p): pred(m, p) for m in TRAINED for p in ["val", "test", "holdout"]}
missing = sorted({m for (m, p), v in P.items() if v is None})
print("нет прогнозов:", missing or "—")
TRAINED = [m for m in TRAINED if m not in missing]
best_uni = json.load(open(find_all("lab02_summary.json")[0]))["best_unimodal"]
print("лучший одномодальный baseline (по val):", best_uni)

# %% [markdown]
# ## 1. Задание 3 — таблица сравнения на test (обучаемые модели, естественное распределение)

# %%
o = O["test"]
rows = []
for m in TRAINED:
    thr = lk.best_threshold(O["val"]["y_drop"].to_numpy(), P[(m, "val")])
    r = {"модель": m, **lk.metrics(o["y_drop"], P[(m, "test")], thr)}
    r["pr_auc_lo"], r["pr_auc_hi"] = lk.block_bootstrap(o["y_drop"], P[(m, "test")], o["day"])
    if m != best_uni and m != "U0_persistence":
        g = lk.paired_diff(o["y_drop"], P[(m, "test")], P[(best_uni, "test")], o["day"], n_perm=2000)
        r.update({"Δpr_auc": g["diff"], "Δ_lo": g["ci_low"], "Δ_hi": g["ci_high"], "p_perm": g["p_perm"]})
    rows.append(r)
T1 = pd.DataFrame(rows)
T1.to_csv(OUT / "table_test.csv", index=False)
print(T1[["модель", "pr_auc", "pr_auc_lo", "pr_auc_hi", "auroc", "macro_f1", "mcc", "brier", "ece", "Δpr_auc",
          "Δ_lo", "Δ_hi", "p_perm"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
t = T1.sort_values("pr_auc")
ax.barh(t["модель"], t["pr_auc"], xerr=[t["pr_auc"] - t["pr_auc_lo"], t["pr_auc_hi"] - t["pr_auc"]],
        color=["#8aa" if m.startswith("U") else "#2a6" for m in t["модель"]])
ax.axvline(o["y_drop"].mean(), color="grey", ls="--", lw=1, label="доля провалов (случайный прогноз)")
ax.set_xlabel("PR-AUC на test, 95%-ный ДИ (блочный бутстрэп по дням)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "01_test_pr_auc.png", dpi=130); plt.close()

# %% [markdown]
# ## 2. E-1200 — все модели, включая VLM (три варианта входа)

# %%
E = O["test"][O["test"]["in_e1200"]].copy()
e_idx = E.index.to_numpy()
rows, PE = [], {}
for m in TRAINED:
    PE[m] = P[(m, "test")][e_idx]
for m in VLM:
    f = find_all(f"{m}__e1200.parquet")
    if not f:
        print("нет", m); continue
    x = pd.read_parquet(f[0]).merge(E[["T", "station"]].reset_index(), on=["T", "station"]).set_index("index").loc[e_idx]
    for v in ["text+chart", "text", "chart"]:
        PE[f"{m} [{v}]"] = x[f"p_{v}"].to_numpy()
for m, p in PE.items():
    r = {"модель": m, **lk.metrics(E["y_drop"], p)}
    r["pr_auc_lo"], r["pr_auc_hi"] = lk.block_bootstrap(E["y_drop"], p, E["day"], n_boot=500)
    for s in "abc":
        mm = (E["stratum"] == s).to_numpy()
        r[f"pr_auc_{s}"] = lk.metrics(E["y_drop"][mm], p[mm])["pr_auc"]
    if m != best_uni:
        g = lk.paired_diff(E["y_drop"], p, PE[best_uni], E["day"], n_boot=500, n_perm=1000)
        r.update({"Δpr_auc": g["diff"], "Δ_lo": g["ci_low"], "Δ_hi": g["ci_high"], "p_perm": g["p_perm"]})
    rows.append(r)
T2 = pd.DataFrame(rows)

# поздняя склейка VLM с числами: логистическая регрессия на [logit U1, logit VLM], 5 фолдов по дням
days = E["day"].to_numpy(); ud = np.unique(days); fold = np.array([np.where(ud == x)[0][0] % 5 for x in days])
y = E["y_drop"].to_numpy()


def cv_stack(cols):
    X = np.column_stack([lk.logit(c) for c in cols]); out = np.zeros(len(y))
    for k in range(5):
        lr = LogisticRegression(max_iter=1000).fit(X[fold != k], y[fold != k]); out[fold == k] = lr.predict_proba(X[fold == k])[:, 1]
    return out


base_cv = cv_stack([PE[best_uni]])
for m in [k for k in PE if k.startswith("M") and "[text+chart]" in k]:
    pf = cv_stack([PE[best_uni], PE[m]])
    g = lk.paired_diff(y, pf, base_cv, days, n_boot=500, n_perm=1000)
    T2 = pd.concat([T2, pd.DataFrame([{"модель": f"{best_uni} + {m} (поздняя склейка, CV)", **lk.metrics(y, pf),
                                       "Δpr_auc": g["diff"], "Δ_lo": g["ci_low"], "Δ_hi": g["ci_high"],
                                       "p_perm": g["p_perm"]}])])
T2.to_csv(OUT / "table_e1200.csv", index=False)
print(T2[["модель", "pr_auc", "pr_auc_lo", "pr_auc_hi", "pr_auc_a", "pr_auc_b", "pr_auc_c", "auroc", "brier",
          "Δpr_auc", "Δ_lo", "Δ_hi", "p_perm"]].round(4).to_string(index=False))

# %% [markdown]
# ## 3. Ресурсы и устойчивость к отсутствующей модальности

# %%
res = pd.concat([pd.read_csv(f) for f in find_all("resources_*.csv")], ignore_index=True)
for m in VLM:
    f = find_all(f"metrics_{m}.csv")
    if f:
        x = pd.read_csv(f[0]); x = x[x["variant"] == "text+chart"].iloc[0]
        res = pd.concat([res, pd.DataFrame([{"model": m, "p50_ms": x["p50_ms"], "p95_ms": x["p95_ms"],
                                             "gpu_gb": x["gpu_gb"], "cost_usd_per_1000": x["cost_usd_per_1000"]}])])
emb = pd.read_csv(find_all("embed_timing.csv")[0])
res.to_csv(OUT / "table_resources.csv", index=False)
print(res.round(3).to_string(index=False)); print(emb.round(3).to_string(index=False))

rob = pd.concat([pd.read_csv(f) for f in find_all("robustness_*.csv")], ignore_index=True)
robust_score = (rob[rob["условие"].isin(["без текста", "без изображения"])]
                .groupby("model")["доля от полного"].min())
print(rob.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Правило выбора (протокол, разд. 10) и подтверждение на holdout

# %%
# отрицательные контроли (lab_02): шумовые признаки и другой seed — та же модель U1 без новой информации.
# Их ДИ показывает, какой «прирост» даёт одна случайность обучения; допуск требует Δ_lo выше него.
gctl = pd.read_csv(find_all("gain_gbm.csv")[0])
gctl = gctl[gctl["model"].str.startswith("C") & (gctl["part"] == "test") & (gctl["metric"] == "pr_auc")]
noise_hi = float(max(0.0, gctl["ci_high"].max())) if len(gctl) else 0.0
print("отрицательные контроли:\n", gctl[["model", "diff", "ci_low", "ci_high", "p_perm"]].round(4).to_string(index=False))
print("порог допуска по Δ_lo:", round(noise_hi, 4))
cand = T1[~T1["модель"].isin(UNI + ["U0_persistence"])].copy()
cand["допуск"] = (cand["Δpr_auc"] > 0) & (cand["Δ_lo"] > noise_hi) & (cand["p_perm"] < 0.05)
lat = res.set_index("model")["p95_ms"]
emb_ms = emb.set_index("what")["per_item_ms"]
extra = {"M2_early_all": emb_ms.filter(like="img").mean() + emb_ms.get("text_bge_small", 0),
         "M3_late_stack": emb_ms.filter(like="img").mean() + emb_ms.get("text_bge_small", 0),
         "M4_gated": emb_ms.filter(like="img").mean(), "M5_crossattn": emb_ms.filter(like="img").mean()}
cand["p95_ms"] = [lat.get(m, np.nan) + extra.get(m, 0) for m in cand["модель"]]
cand["p95_ms"] = cand["p95_ms"].fillna(lat.get("M1_num_txtstruct", 1.0))
cand["устойчивость"] = [robust_score.get(m, 1.0 if m == "M1_num_txtstruct" else np.nan) for m in cand["модель"]]
adm = cand[cand["допуск"]].copy()


def mm(s, higher=True):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(1.0, index=s.index)
    z = (s - s.min()) / (s.max() - s.min())
    return z if higher else 1 - z


if len(adm):
    adm["балл"] = (0.30 * mm(adm["pr_auc"]) + 0.15 * mm(adm["macro_f1"]) + 0.20 * mm(adm["auroc"])
                  + 0.15 * mm(adm["brier"], False) + 0.10 * mm(adm["устойчивость"].fillna(adm["устойчивость"].min()))
                  + 0.10 * mm(np.log(adm["p95_ms"]), False))
    adm = adm.sort_values("балл", ascending=False)
    best = adm.iloc[0]["модель"]
    if len(adm) > 1 and adm.iloc[0]["балл"] - adm.iloc[1]["балл"] < 0.02:
        best = adm.iloc[:2].sort_values("p95_ms").iloc[0]["модель"]
    selection_note = "допущена по правилу"
else:
    best = cand.sort_values("pr_auc", ascending=False).iloc[0]["модель"]
    selection_note = "ни одна схема не прошла допуск; применяется лучшая по PR-AUC с пометкой «прирост не доказан»"
cand.to_csv(OUT / "selection_table.csv", index=False)
print(cand[["модель", "pr_auc", "Δpr_auc", "Δ_lo", "p_perm", "допуск", "устойчивость", "p95_ms"]].round(4).to_string(index=False))
if len(adm):
    print(adm[["модель", "балл"]].round(3).to_string(index=False))
print("ВЫБРАНА:", best, "—", selection_note)

oh = O["holdout"]
hold = {"модель": best, **lk.metrics(oh["y_drop"], P[(best, "holdout")]),
        **{f"{k}_vs_{best_uni}": v for k, v in lk.paired_diff(oh["y_drop"], P[(best, "holdout")], P[(best_uni, "holdout")],
                                                              oh["day"], n_perm=2000).items()}}
hold[f"pr_auc_{best_uni}"] = lk.metrics(oh["y_drop"], P[(best_uni, "holdout")])["pr_auc"]
hold["подтверждено"] = bool(hold[f"diff_vs_{best_uni}"] > 0 and hold[f"ci_low_vs_{best_uni}"] > 0)
print(json.dumps(hold, ensure_ascii=False, indent=1, default=float))

# %% [markdown]
# ## 5. Задание 4 — применение на holdout: районы, слои, часы пик; список для диспетчера

# %%
d = lk.load()
boro = d.static["borough"].to_numpy()
oh = oh.assign(p_best=P[(best, "holdout")], p_base=P[(best_uni, "holdout")], borough=boro[oh["s"]],
               peak=np.where(oh["hour"].isin([7, 8, 9, 16, 17, 18, 19]) & (oh["is_weekend"] == 0), "пик", "вне пика"))
rows = []
for col in ["borough", "stratum", "peak"]:
    for k, g in oh.groupby(col):
        if g["y_drop"].sum() < 5:
            continue
        rows.append({"срез": col, "значение": k, "n": len(g), "провалов": int(g["y_drop"].sum()),
                     f"pr_auc {best}": lk.metrics(g["y_drop"], g["p_best"])["pr_auc"],
                     f"pr_auc {best_uni}": lk.metrics(g["y_drop"], g["p_base"])["pr_auc"]})
app = pd.DataFrame(rows); app["Δ"] = app[f"pr_auc {best}"] - app[f"pr_auc {best_uni}"]
app.to_csv(OUT / "application_holdout.csv", index=False)
print(app.round(4).to_string(index=False))

# диспетчерский список: час holdout с наибольшим числом станций слоя «a»
hh = oh[oh["stratum"] == "a"].groupby("T").size().idxmax()
lst = oh[oh["T"] == hh].sort_values("p_best", ascending=False).head(12)
lst = lst.assign(station_name=d.static["station_complex"].to_numpy()[lst["s"]],
                 alerts=[lk.format_alerts(r, A, with_level=False).split("\n")[0][:140] for _, r in lst.iterrows()])
lst[["T", "station_name", "borough", "p_best", "p_base", "y_drop", "y", "ref", "alerts"]].to_csv(
    OUT / "dispatcher_list.csv", index=False)
print("Час:", hh); print(lst[["station_name", "p_best", "p_base", "y_drop", "alerts"]].round(3).to_string(index=False))

# %% [markdown]
# ## 6. Задания 5–6 — анализ ошибок, режимы, конфликты модальностей

# %%
o = O["test"].assign(p_best=P[(best, "test")], p_base=P[(best_uni, "test")], borough=boro[O["test"]["s"]])
top_type = np.where(o["alert_0"] >= 0, A["status_label"].to_numpy()[o["alert_0"].clip(0)], "нет оповещений")
o["тип оповещения"] = pd.Series(top_type).str.split(" | ", regex=False).str[0].to_numpy()
o["с оповещения"] = pd.cut(o["txt_min_since"], [0, 15, 60, 180], labels=["0–15", "15–60", "60–180"]).astype(str)
o["час"] = np.where(o["hour"].isin([7, 8, 9, 16, 17, 18, 19]) & (o["is_weekend"] == 0), "пик", "вне пика")
o["выходной"] = np.where(o["is_weekend"] == 1, "выходной", "будний")
o["масштаб"] = pd.qcut(o["st_scale_weekday"], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
o["осадки"] = np.where(o["wx_precipitation"] > 0.5, "дождь/снег", "сухо")
o["начало провала"] = np.where(o["r_lag1"].fillna(1) >= lk.DROP_RATIO, "в T−1 провала нет", "провал уже идёт")
rows = []
for col in ["stratum", "тип оповещения", "с оповещения", "час", "выходной", "borough", "масштаб", "осадки", "начало провала"]:
    for k, g in o.groupby(col):
        if g["y_drop"].sum() < 10:
            continue
        a_ = lk.metrics(g["y_drop"], g["p_best"])["pr_auc"]; b_ = lk.metrics(g["y_drop"], g["p_base"])["pr_auc"]
        rows.append({"срез": col, "значение": k, "n": len(g), "провалов": int(g["y_drop"].sum()),
                     f"pr_auc {best}": a_, f"pr_auc {best_uni}": b_, "Δ": a_ - b_})
regimes = pd.DataFrame(rows)
regimes.to_csv(OUT / "regimes_test.csv", index=False)
print(regimes.round(4).to_string(index=False))

# конфликт модальностей
conf = []
txt_says = o["stratum"].eq("a"); num_says = o["r_lag1"].fillna(1) < lk.DROP_RATIO
for name, mask in [("текст: сбой у станции; числа: норма", txt_says & ~num_says),
                   ("текст: нет оповещений; числа: провал", ~o["txt_n_line"].gt(0) & num_says),
                   ("согласны: сбой", txt_says & num_says), ("согласны: норма", ~txt_says & ~num_says)]:
    g = o[mask]
    conf.append({"ситуация": name, "n": len(g), "доля провалов": g["y_drop"].mean(),
                 f"средняя p {best}": g["p_best"].mean(), f"средняя p {best_uni}": g["p_base"].mean(),
                 f"pr_auc {best}": lk.metrics(g["y_drop"], g["p_best"])["pr_auc"] if g["y_drop"].sum() else np.nan,
                 f"pr_auc {best_uni}": lk.metrics(g["y_drop"], g["p_base"])["pr_auc"] if g["y_drop"].sum() else np.nan})
conf = pd.DataFrame(conf); conf.to_csv(OUT / "modality_conflicts.csv", index=False)
print(conf.round(4).to_string(index=False))

# %%
o["err"] = (o["y_drop"] - o["p_best"]).abs()
err = pd.concat([o[o["y_drop"] == 1].nlargest(5, "err"), o[o["y_drop"] == 0].nlargest(5, "err")])
Y = d.y.to_numpy("float32")
recs = []
for k, (_, r) in enumerate(err.iterrows()):
    lk.render_chart(*lk.chart_series(Y, int(r["t"]), int(r["s"]))).save(FIG / f"error_{k}.png")
    recs.append({"№": k, "тип": "пропуск провала (FN)" if r["y_drop"] else "ложная тревога (FP)", "T": r["T"],
                 "станция": d.static["station_complex"].iloc[int(r["s"])], "p": round(r["p_best"], 3),
                 "y / ref": f"{r['y']:.0f} / {r['ref']:.0f}", "r_lag1": round(r["r_lag1"], 2),
                 "оповещения": lk.format_alerts(r, A)[:300] or "—"})
errors = pd.DataFrame(recs); errors.to_csv(OUT / "top_errors.csv", index=False)
print(errors.to_string(index=False))

# %% [markdown]
# ## 7. Гипотезы H1–H5

# %%
g_best = T1.set_index("модель").loc[best]
strat_gain = {s: lk.metrics(o["y_drop"][o["stratum"] == s], o["p_best"][o["stratum"] == s])["pr_auc"]
              - lk.metrics(o["y_drop"][o["stratum"] == s], o["p_base"][o["stratum"] == s])["pr_auc"] for s in "abc"}
gain_all = pd.concat([pd.read_csv(f) for f in find_all("gain_*.csv")], ignore_index=True)
img_gain = gain_all[(gain_all["model"] == "A1_M2_no_text") & (gain_all["part"] == "test") & (gain_all["metric"] == "pr_auc")]
hz = pd.read_csv(find_all("ablations_horizon_window.csv")[0])
vlm_best = T2[T2["модель"].str.contains(r"\[text\+chart\]")]["pr_auc"].max() if len(T2) else np.nan
trained_best_e = T2[T2["модель"].isin([m for m in TRAINED if m.startswith("M")])]["pr_auc"].max()
H = [
    {"гипотеза": "H1: прирост PR-AUC ≥ 0,02, ДИ > 0, p < 0,05",
     "факт": f"{best}: Δ = {g_best.get('Δpr_auc', np.nan):+.4f} [{g_best.get('Δ_lo', np.nan):+.4f}; {g_best.get('Δ_hi', np.nan):+.4f}], p = {g_best.get('p_perm', np.nan):.3f}",
     "вывод": "подтверждена" if g_best.get("Δpr_auc", 0) >= 0.02 and g_best.get("Δ_lo", 0) > noise_hi and g_best.get("p_perm", 1) < 0.05
     else ("частично: прирост значим (выше контроля), но < 0,02" if g_best.get("Δ_lo", 0) > noise_hi else "отклонена")},
    {"гипотеза": "H2: прирост сосредоточен в слое a",
     "факт": ", ".join(f"слой {s}: Δ {v:+.4f}" for s, v in strat_gain.items()),
     "вывод": "подтверждена" if strat_gain["a"] > max(strat_gain["b"], strat_gain["c"]) + 0.005 else "не подтверждена"},
    {"гипотеза": "H3: изображение не даёт прироста сверх чисел",
     "факт": (f"числа + изображение против чисел: Δ = {img_gain['diff'].iloc[0]:+.4f} "
              f"[{img_gain['ci_low'].iloc[0]:+.4f}; {img_gain['ci_high'].iloc[0]:+.4f}]") if len(img_gain) else "н/д",
     "вывод": ("подтверждена" if img_gain["ci_low"].iloc[0] <= noise_hi else "отклонена") if len(img_gain) else "н/д"},
    {"гипотеза": "H4: прирост текста убывает с горизонтом",
     "факт": "; ".join(f"{r['вариант']}: Δ {r['diff']:+.4f}" for _, r in hz[hz["абляция"] == "A8 горизонт"].iterrows()),
     "вывод": "подтверждена" if hz[hz["абляция"] == "A8 горизонт"]["diff"].is_monotonic_decreasing else "не подтверждена"},
    {"гипотеза": "H5: VLM zero-shot уступают обучаемым схемам",
     "факт": f"лучшая VLM PR-AUC {vlm_best:.4f} против {trained_best_e:.4f} у лучшей обучаемой (E-1200)",
     "вывод": "подтверждена" if vlm_best < trained_best_e else "отклонена"},
]
H = pd.DataFrame(H); H.to_csv(OUT / "hypotheses.csv", index=False)
print(H.to_string(index=False))
json.dump({"best": best, "selection_note": selection_note, "best_unimodal": best_uni, "holdout": hold},
          open(OUT / "selection.json", "w"), ensure_ascii=False, indent=1, default=float)
print("готово, мин:", round((time.time() - T0) / 60, 1))
