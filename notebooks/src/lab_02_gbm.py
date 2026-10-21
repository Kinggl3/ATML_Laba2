# %% [markdown]
# # ЛР2 · lab_02 — одномодальные baseline'ы, раннее и позднее слияние, абляции (CPU)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA` (с `alerts/`), вывод `lab-00-data`, вывод `lab-01-embed`.
# **Accelerator:** нет (CPU). Время: ~40–60 мин.
#
# Модели (протокол, разд. 6): U0 инерция, U1 числа, U2 текст, U3 изображение,
# M1 числа + структурный текст, M2 числа + эмбеддинги текста и изображения, M3 стекинг U1–U3.
# Абляции (разд. 9): A1–A3, A5, A6, A7, A8, A9, A10; перестановка модальностей.

# %%
import os, sys, time, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
OUT = lk.results_dir()
PARTS = ["train", "val", "test", "holdout"]
T0 = time.time()

O = {p: lk.load_objects(p) for p in PARTS}
A = lk.load_alerts_clean()
E_alert = np.load(lk.find_file("alert_emb.npy"))
IMG = {p: np.load(lk.find_file(f"img_emb_{p}.npy")) for p in PARTS}
for p in PARTS:
    assert len(IMG[p]) == len(O[p]), f"img_emb_{p} не совпадает с objects_{p}"
    print(p, len(O[p]), "провалов:", round(O[p]["y_drop"].mean(), 4))

# %% [markdown]
# ## Признаки модальностей
# - числа: `lk.NUM_FEATURES`;
# - текст, структурный: `lk.TEXT_FEATURES`;
# - текст, эмбеддинг: взвешенное среднее эмбеддингов активных оповещений → PCA 32 (обучена на train);
# - изображение: эмбеддинг SigLIP → PCA 16 (обучена на train).

# %%
N_TXT_PC, N_IMG_PC = 32, 16
TE = {p: lk.object_text_emb(O[p], E_alert) for p in PARTS}
has_txt = TE["train"].any(1)
pca_t = PCA(N_TXT_PC, random_state=lk.SEED).fit(TE["train"][has_txt])
pca_i = PCA(N_IMG_PC, random_state=lk.SEED).fit(IMG["train"].astype("float32")[::4])
TXT_PC = [f"temb_{i}" for i in range(N_TXT_PC)]
IMG_PC = [f"iemb_{i}" for i in range(N_IMG_PC)]


def add_emb(o, te, ie):
    t = np.where(te.any(1, keepdims=True), pca_t.transform(te), np.nan)   # нет оповещений — пропуск
    o = o.copy()
    o[TXT_PC] = t.astype("float32")
    o[IMG_PC] = pca_i.transform(ie.astype("float32")).astype("float32")
    return o


O = {p: add_emb(O[p], TE[p], IMG[p]) for p in PARTS}
# отрицательный контроль: 16 столбцов шума — «модальность», в которой заведомо нет информации
NOISE = [f"noise_{i}" for i in range(16)]
_rng = np.random.default_rng(lk.SEED)
for p in PARTS:
    O[p][NOISE] = _rng.normal(size=(len(O[p]), len(NOISE))).astype("float32")
print("доля дисперсии PCA: текст", pca_t.explained_variance_ratio_.sum().round(3),
      "изображение", pca_i.explained_variance_ratio_.sum().round(3))

FEATS = {
    "U1_num": lk.NUM_FEATURES,
    "U2_text": lk.TEXT_FEATURES + TXT_PC,
    "M1_num_txtstruct": lk.NUM_FEATURES + lk.TEXT_FEATURES,
    "M2_early_all": lk.NUM_FEATURES + lk.TEXT_FEATURES + TXT_PC + IMG_PC,
    # абляции M2 (A1–A3), представление текста (A5), только линия (A6), без соседей (A10)
    "A1_M2_no_text": lk.NUM_FEATURES + IMG_PC,
    "A2_M2_no_image": lk.NUM_FEATURES + lk.TEXT_FEATURES + TXT_PC,
    "A3_M2_no_numbers": lk.TEXT_FEATURES + TXT_PC + IMG_PC,
    "A5_num_txtemb": lk.NUM_FEATURES + TXT_PC,
    "A6_num_txtline": lk.NUM_FEATURES + lk.TXT_LINE_ONLY,
    "A10_U1_no_nbr": [c for c in lk.NUM_FEATURES if c != "nbr_r_lag1"],
    "A10_M1_no_nbr": [c for c in lk.NUM_FEATURES if c != "nbr_r_lag1"] + lk.TEXT_FEATURES,
    # отрицательные контроли: насколько «значим» прирост от признаков без информации и от другого seed
    "C0_U1_plus_noise": lk.NUM_FEATURES + NOISE,
    "C1_U1_seed2": lk.NUM_FEATURES,
}
# A5: «только структурный» = M1, «оба» = A2_M2_no_image

# %% [markdown]
# ## Обучение: LightGBM, ранняя остановка по PR-AUC на val; holdout — переобучение на train + val

# %%
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric="average_precision", verbose=-1, seed=lk.SEED, num_threads=4)
MAX_ROUNDS = int(os.environ.get("LAB_MAX_ROUNDS", 3000))   # переменная — только для быстрой проверки кода
PRED = {}      # (модель, часть) → вероятности
INFO = {}


def fit_gbm(name, feats, target="y_drop", parts=O, store=True):
    params = {**PARAMS, "seed": lk.SEED + 1} if name.startswith("C1_") else PARAMS
    tr, va = parts["train"], parts["val"]
    dtr = lgb.Dataset(tr[feats], tr[target], weight=tr["weight"], free_raw_data=False)
    dva = lgb.Dataset(va[feats], va[target], reference=dtr)
    t0 = time.time()
    m = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    fit_s = time.time() - t0
    out = {p: m.predict(parts[p][feats], num_iteration=m.best_iteration) for p in ["val", "test"]}
    # holdout: переобучение на train + val с тем же числом деревьев × 1,1
    trva = pd.concat([tr, va])
    m2 = lgb.train(params, lgb.Dataset(trva[feats], trva[target], weight=trva["weight"]),
                   int(m.best_iteration * 1.1) + 1)
    out["holdout"] = m2.predict(parts["holdout"][feats])
    if store:
        for p, v in out.items():
            PRED[(name, p)] = v
        INFO[name] = {"feats": len(feats), "trees": m.best_iteration, "fit_sec": round(fit_s, 1), "model": m,
                      "model_full": m2}
    print(f"{name:<20} признаков {len(feats):>3}  деревьев {m.best_iteration:>4}  "
          f"PR-AUC val {lk.metrics(va[target], out['val'])['pr_auc']:.4f}  {fit_s:.0f} c")
    return m, out


for name, feats in FEATS.items():
    fit_gbm(name, feats)

# %% [markdown]
# ## U3 — только изображение (логистическая регрессия на эмбеддинге SigLIP), U0 — инерция

# %%
t0 = time.time()
u3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
u3.fit(IMG["train"].astype("float32"), O["train"]["y_drop"], logisticregression__sample_weight=O["train"]["weight"])
for p in ["val", "test"]:
    PRED[("U3_image", p)] = u3.predict_proba(IMG[p].astype("float32"))[:, 1]
u3f = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
X_tv = np.vstack([IMG["train"], IMG["val"]]).astype("float32")
w_tv = np.concatenate([O["train"]["weight"], O["val"]["weight"]])
u3f.fit(X_tv, np.concatenate([O["train"]["y_drop"], O["val"]["y_drop"]]), logisticregression__sample_weight=w_tv)
PRED[("U3_image", "holdout")] = u3f.predict_proba(IMG["holdout"].astype("float32"))[:, 1]
INFO["U3_image"] = {"feats": IMG["train"].shape[1], "fit_sec": round(time.time() - t0, 1), "model": u3f}

# U0: провал, если в T−1 поток < 75% обычного; «вероятность» — доля провалов в train при этом условии
tr = O["train"]
cond = (tr["r_lag1"] < lk.DROP_RATIO).to_numpy()
p1 = np.average(tr["y_drop"][cond], weights=tr["weight"][cond]); p0 = np.average(tr["y_drop"][~cond], weights=tr["weight"][~cond])
for p in PARTS:
    PRED[("U0_persistence", p)] = np.where(O[p]["r_lag1"] < lk.DROP_RATIO, p1, p0)
INFO["U0_persistence"] = {"feats": 1}

# %% [markdown]
# ## M3 — позднее слияние (стекинг U1, U2, U3)
# Логистическая регрессия на логитах одномодальных моделей. Обучается на val; прогноз для самого val —
# блочной кросс-валидацией по дням (5 фолдов), чтобы порог подбирался не на обучающих данных.

# %%
def stack_X(part):
    return np.column_stack([lk.logit(PRED[(m, part)]) for m in ["U1_num", "U2_text", "U3_image"]])


va = O["val"]
Xv, yv = stack_X("val"), va["y_drop"].to_numpy()
days = va["day"].to_numpy(); ud = np.unique(days)
fold = {dd: i % 5 for i, dd in enumerate(ud)}; f = np.array([fold[x] for x in days])
cv = np.zeros(len(va))
for k in range(5):
    lr = LogisticRegression(max_iter=1000).fit(Xv[f != k], yv[f != k])
    cv[f == k] = lr.predict_proba(Xv[f == k])[:, 1]
m3 = LogisticRegression(max_iter=1000).fit(Xv, yv)
PRED[("M3_late_stack", "val")] = cv
for p in ["test", "holdout"]:
    PRED[("M3_late_stack", p)] = m3.predict_proba(stack_X(p))[:, 1]
INFO["M3_late_stack"] = {"feats": 3, "coef": dict(zip(["U1", "U2", "U3"], m3.coef_[0].round(3)))}
print("M3 веса логитов:", INFO["M3_late_stack"]["coef"])

# %% [markdown]
# ## Метрики, интервалы, журнал

# %%
MAIN = ["U0_persistence", "U1_num", "U2_text", "U3_image", "M1_num_txtstruct", "M2_early_all", "M3_late_stack"]
ABL = [k for k in FEATS if k.startswith(("A", "C"))]
THR = {m: lk.best_threshold(O["val"]["y_drop"].to_numpy(), PRED[(m, "val")]) for m in MAIN + ABL}
rows = []
for m in MAIN + ABL:
    for p in ["val", "test", "holdout"]:
        o = O[p]
        r = {"model": m, "part": p, **lk.metrics(o["y_drop"], PRED[(m, p)], THR[m])}
        if p != "val":
            r["pr_auc_ci_low"], r["pr_auc_ci_high"] = lk.block_bootstrap(o["y_drop"], PRED[(m, p)], o["day"])
        rows.append(r)
        pd.DataFrame({"T": o["T"], "station": o["station"], "y_drop": o["y_drop"], "p": PRED[(m, p)].astype("float32")}
                     ).to_parquet(OUT / f"{m}__{p}.parquet", index=False)
res = pd.DataFrame(rows)
res.to_csv(OUT / "metrics_gbm.csv", index=False)
print(res[res["part"] == "test"][["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "auroc", "macro_f1",
                                  "mcc", "brier", "ece"]].round(4).to_string(index=False))

# %% [markdown]
# ### Лучший одномодальный baseline (по val) и прирост мультимодальных схем

# %%
val_pr = res[(res["part"] == "val")].set_index("model")["pr_auc"]
best_uni = val_pr[["U1_num", "U2_text", "U3_image"]].idxmax()
print("лучший одномодальный baseline по val:", best_uni)
gain = []
for p in ["test", "holdout"]:
    o = O[p]
    for m in ["M1_num_txtstruct", "M2_early_all", "M3_late_stack"] + ABL:
        for metric in ["pr_auc", "auroc", "brier"]:
            g = lk.paired_diff(o["y_drop"], PRED[(m, p)], PRED[(best_uni, p)], o["day"], metric=metric,
                               n_boot=1000, n_perm=1000 if metric == "pr_auc" else 200)
            gain.append({"model": m, "vs": best_uni, "part": p, "metric": metric, **g})
gain = pd.DataFrame(gain)
gain.to_csv(OUT / "gain_gbm.csv", index=False)
print(gain[(gain["metric"] == "pr_auc")].round(4).to_string(index=False))

# %% [markdown]
# ## Прирост по слоям (H2) и по времени с оповещения (H4)

# %%
rows = []
o = O["test"]
for strat, g in o.groupby("stratum"):
    for m in MAIN:
        rows.append({"срез": f"слой {strat}", "model": m, "n": len(g), "pos": int(g["y_drop"].sum()),
                     **{k: v for k, v in lk.metrics(g["y_drop"], PRED[(m, "test")][g.index]).items()
                        if k in ("pr_auc", "auroc", "brier")}})
# начало провала: в прошлый час провала не было — здесь числа «не видят» сбой, и текст может помочь
onset = o["r_lag1"].fillna(1) >= lk.DROP_RATIO
for name, g in [("начало провала (в T−1 провала нет)", o[onset]), ("начало провала, слой a", o[onset & (o["stratum"] == "a")])]:
    for m in MAIN:
        rows.append({"срез": name, "model": m, "n": len(g), "pos": int(g["y_drop"].sum()),
                     **{k: v for k, v in lk.metrics(g["y_drop"], PRED[(m, "test")][g.index]).items()
                        if k in ("pr_auc", "auroc", "brier")}})
bins = pd.cut(o["txt_min_since_core"], [0, 15, 60, 180], labels=["0–15 мин", "15–60 мин", "60–180 мин"])
for b, g in o.groupby(bins, observed=True):
    for m in ["U1_num", "M1_num_txtstruct", "M2_early_all"]:
        rows.append({"срез": f"с оповещения {b}", "model": m, "n": len(g), "pos": int(g["y_drop"].sum()),
                     **{k: v for k, v in lk.metrics(g["y_drop"], PRED[(m, "test")][g.index]).items()
                        if k in ("pr_auc", "auroc", "brier")}})
slices = pd.DataFrame(rows)
slices.to_csv(OUT / "slices_gbm.csv", index=False)
print(slices.pivot_table(index="срез", columns="model", values="pr_auc").round(3).to_string())

# %% [markdown]
# ## A9 — отсутствие модальности на входе без переобучения; перестановка модальностей

# %%
rows = []
o = O["test"]
m2 = INFO["M2_early_all"]["model"]; f2 = FEATS["M2_early_all"]
full = lk.metrics(o["y_drop"], PRED[("M2_early_all", "test")])["pr_auc"]
o_nt = lk.no_text(o); o_nt[TXT_PC] = np.nan
o_ni = o.copy(); o_ni[IMG_PC] = np.nan
for name, oo in [("без текста", o_nt), ("без изображения", o_ni),
                 ("перестановка текста внутри часа", lk.shuffle_within(o, lk.TEXT_FEATURES + TXT_PC)),
                 ("перестановка изображения внутри часа", lk.shuffle_within(o, IMG_PC)),
                 ("перестановка чисел внутри часа", lk.shuffle_within(o, lk.NUM_FEATURES))]:
    pr = lk.metrics(o["y_drop"], m2.predict(oo[f2], num_iteration=m2.best_iteration))["pr_auc"]
    rows.append({"model": "M2_early_all", "условие": name, "pr_auc": pr, "доля от полного": pr / full})
m1 = INFO["M1_num_txtstruct"]["model"]; f1 = FEATS["M1_num_txtstruct"]
full1 = lk.metrics(o["y_drop"], PRED[("M1_num_txtstruct", "test")])["pr_auc"]
for name, oo in [("без текста", lk.no_text(o)), ("перестановка текста внутри часа", lk.shuffle_within(o, lk.TEXT_FEATURES))]:
    pr = lk.metrics(o["y_drop"], m1.predict(oo[f1], num_iteration=m1.best_iteration))["pr_auc"]
    rows.append({"model": "M1_num_txtstruct", "условие": name, "pr_auc": pr, "доля от полного": pr / full1})
robust = pd.DataFrame(rows)
robust.to_csv(OUT / "robustness_gbm.csv", index=False)
print(robust.round(4).to_string(index=False))

# %% [markdown]
# ## A8 — горизонт h = 1, 2, 3 и A7 — окно активности оповещения 30 / 90 / 180 мин
# Сравнение «числа» (U1) и «числа + структурный текст» (M1) — прирост от текста.

# %%
d = lk.load()
ref = lk.reference_level(d)
rows = []
for h in (2, 3):
    tgt = f"y_drop_h{h}"
    P = {}
    for p in PARTS:
        o = O[p]
        P[p] = pd.concat([o[["weight", tgt, "day", "T"] + lk.TEXT_FEATURES].reset_index(drop=True),
                          lk.numeric_features(d, ref, o["t"].to_numpy(), o["s"].to_numpy(), h=h)], axis=1)
    for name, feats in [("U1_num", lk.NUM_FEATURES), ("M1_num_txtstruct", lk.NUM_FEATURES + lk.TEXT_FEATURES)]:
        _, out = fit_gbm(f"{name}_h{h}", feats, target=tgt, parts=P, store=False)
        P["test"][f"p_{name}"] = out["test"]
    g = lk.paired_diff(P["test"][tgt], P["test"]["p_M1_num_txtstruct"], P["test"]["p_U1_num"], P["test"]["day"],
                       n_perm=500)
    rows.append({"абляция": "A8 горизонт", "вариант": f"h={h}",
                 "pr_auc_U1": lk.metrics(P["test"][tgt], P["test"]["p_U1_num"])["pr_auc"],
                 "pr_auc_M1": lk.metrics(P["test"][tgt], P["test"]["p_M1_num_txtstruct"])["pr_auc"], **g})
o = O["test"]
g = lk.paired_diff(o["y_drop"], PRED[("M1_num_txtstruct", "test")], PRED[("U1_num", "test")], o["day"], n_perm=500)
rows.insert(0, {"абляция": "A8 горизонт", "вариант": "h=1 (основной)",
                "pr_auc_U1": lk.metrics(o["y_drop"], PRED[("U1_num", "test")])["pr_auc"],
                "pr_auc_M1": lk.metrics(o["y_drop"], PRED[("M1_num_txtstruct", "test")])["pr_auc"], **g})

# %%
a_full = lk.load_alerts(d.root)
a_full, _ = lk.clean_alerts(a_full)
a_full, _ = lk.relate_alerts(a_full, d)
idx = d.y.index
for amin, along in [(30, 60), (180, 360)]:
    act = lk.active_rows(a_full, idx, amin, along)
    P = {}
    for p in PARTS:                    # по частям — чтобы не держать в памяти всю развёртку сразу
        x = lk.station_alert_table(a_full, act, idx, np.unique(O[p]["t"]))
        txt = lk.aggregate_text_features(x, a_full); del x
        o = O[p][["t", "s", "weight", "y_drop", "day"] + lk.NUM_FEATURES].merge(
            txt[["t", "s"] + lk.TEXT_FEATURES], on=["t", "s"], how="left")
        del txt
        for c in lk.TEXT_FEATURES:
            if not c.startswith("txt_min"):
                o[c] = o[c].fillna(0)
        P[p] = o
    _, out = fit_gbm(f"M1_win{amin}", lk.NUM_FEATURES + lk.TEXT_FEATURES, parts=P, store=False)
    g = lk.paired_diff(P["test"]["y_drop"], out["test"], PRED[("U1_num", "test")], P["test"]["day"], n_perm=500)
    rows.append({"абляция": "A7 окно активности", "вариант": f"{amin} / {along} мин",
                 "pr_auc_U1": lk.metrics(O["test"]["y_drop"], PRED[("U1_num", "test")])["pr_auc"],
                 "pr_auc_M1": lk.metrics(P["test"]["y_drop"], out["test"])["pr_auc"], **g})
rows.append({"абляция": "A7 окно активности", "вариант": f"{lk.ACTIVE_MIN} / {lk.ACTIVE_MIN_LONG} мин (основной)",
             **{k: v for k, v in rows[0].items() if k not in ("абляция", "вариант")}})
hz = pd.DataFrame(rows)
hz.to_csv(OUT / "ablations_horizon_window.csv", index=False)
print(hz.round(4).to_string(index=False))

# %% [markdown]
# ## Ресурсы: задержка на объект (пачка 1) и пропускная способность

# %%
o = O["test"].sample(300, random_state=lk.SEED)
res_rows = []
for m in ["U1_num", "U2_text", "M1_num_txtstruct", "M2_early_all"]:
    mdl, feats = INFO[m]["model"], FEATS[m]
    tm = lk.Timer()
    for i in range(len(o)):
        with tm:
            mdl.predict(o.iloc[[i]][feats], num_iteration=mdl.best_iteration)
    t0 = time.time(); mdl.predict(O["test"][feats], num_iteration=mdl.best_iteration); thr = len(O["test"]) / (time.time() - t0)
    res_rows.append({"model": m, "p50_ms": 1000 * tm.p(50), "p95_ms": 1000 * tm.p(95), "objects_per_sec": thr,
                     "fit_sec": INFO[m]["fit_sec"], "gpu_gb": 0.0})
for m in ["U3_image", "M3_late_stack"]:
    res_rows.append({"model": m, "p50_ms": np.nan, "p95_ms": np.nan, "objects_per_sec": np.nan,
                     "fit_sec": INFO[m].get("fit_sec", np.nan), "gpu_gb": 0.0})
resources = pd.DataFrame(res_rows)
resources.to_csv(OUT / "resources_gbm.csv", index=False)
print(resources.round(2).to_string(index=False))
print("Без учёта эмбеддингов: для U2/M2 добавляется кодирование текста и изображения (см. lab_01, embed_timing.csv).")

# %%
imp = pd.DataFrame({"feature": FEATS["M2_early_all"],
                    "gain": INFO["M2_early_all"]["model"].feature_importance("gain")})
imp["modality"] = np.select([imp["feature"].isin(lk.TEXT_FEATURES + TXT_PC), imp["feature"].isin(IMG_PC)],
                            ["текст", "изображение"], "числа")
imp.sort_values("gain", ascending=False).to_csv(OUT / "importance_M2.csv", index=False)
print(imp.groupby("modality")["gain"].sum().pipe(lambda s: s / s.sum()).round(3))
json.dump({"best_unimodal": best_uni, "thresholds": THR,
           "info": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("model")} for k, v in INFO.items()}},
          open(OUT / "lab02_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
print("готово, мин:", round((time.time() - T0) / 60, 1))
