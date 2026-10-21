# %% [markdown]
# # ЛР2 · lab_00 — данные: синхронизация, объекты, паспорт оповещений
#
# Задания 2 и 7 (протокол: разделы 1–5, 12). Только CPU.
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA` (версия с папкой `alerts/`).
# **Output:** `labkit.py`, объекты `objects_*.parquet`, оповещения `alerts_clean.parquet`, паспорт, журналы.
#
# Что делает:
# 1. Оповещения: дубликаты (S3), часовой пояс (S1), провалы выгрузки (S8), линии и станции (S5–S7).
# 2. Проверка синхронизации S1: на каком сдвиге оповещение «совпадает» с провалом спроса.
# 3. Объекты (станция × час), цель, исключения, выборки train / val / test / holdout, E-1200, V-60.
# 4. Числовая и текстовая модальности; пример изображения; паспорт; схема данных.

# %%
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
sys.path.insert(0, str(HERE))        # labkit.py записан первой ячейкой (на Kaggle) или лежит рядом
import labkit as lk
OUT = lk.results_dir()
t0 = time.time()
d = lk.load()
idx = d.y.index
print(d.root, d.y.shape)

# %% [markdown]
# ## 1. Оповещения: очистка, часовой пояс, привязка к станциям

# %%
raw = lk.load_alerts(d.root)
a, excl_alerts = lk.clean_alerts(raw)
a, unmatched = lk.relate_alerts(a, d)
gaps = lk.gap_days(a)
print(f"строк в выгрузке: {len(raw):,}; после очистки: {len(a):,}; исключено: {len(excl_alerts):,}")
print(excl_alerts["reason"].value_counts().to_string())
print("дней-провалов выгрузки (UTC):", int(gaps["is_gap"].sum()))
print("доля оповещений с упоминанием станции:", round((a["st_mention"].map(len) > 0).mean(), 3),
      "| с участком from–to:", round((a["st_range"].map(len) > 0).mean(), 3),
      "| непривязанных упоминаний:", len(unmatched))

# %% [markdown]
# ## 2. Проверка синхронизации (S1)
# Для служебных оповещений с упомянутой станцией: доля провалов на этой станции в часы
# «час оповещения + k», k = −6…+6. Если время оповещений и входов в одной шкале, максимум должен быть
# при k = 0…+1. Считаем дважды: по исходному времени набора (`date_utc`) и после перевода в местное.

# %%
ref = lk.reference_level(d)
Y = d.y.to_numpy("float32"); C = d.closed.to_numpy()
valid = (~C) & np.isfinite(ref) & (ref >= lk.MIN_REF)
drop = np.where(valid, Y < lk.DROP_RATIO * ref, False)
hour_ok = np.isin(idx.hour, lk.HOURS)

def lag_profile(times, stations, ks=range(-6, 7)):
    h0 = idx.searchsorted(times.dt.floor("h"))
    res = {}
    for k in ks:
        v = [drop[h + k, s] for h, sts in zip(h0, stations) for s in sts
             if 0 <= h + k < len(idx) and valid[h + k, s] and hour_ok[h + k]]
        res[k] = np.mean(v)
    return pd.Series(res)

sv = a[a["is_service"] & (a["st_mention"].map(len) > 0)]
loc = sv["date_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
is_dst = loc.map(lambda x: bool(x.dst())).to_numpy()
sync = pd.DataFrame({
    "исходное время, зима (EST)": lag_profile(sv["date_utc"][~is_dst], sv["st_mention"][~is_dst]),
    "исходное время, лето (EDT)": lag_profile(sv["date_utc"][is_dst], sv["st_mention"][is_dst]),
    "местное время (после перевода)": lag_profile(sv["date"], sv["st_mention"]),
})
sync.index.name = "сдвиг k, ч"
sync.to_csv(OUT / "sync_check.csv")
print(sync.round(3).to_string())
ax = sync.plot(marker="o", figsize=(8, 4)); ax.axvline(0, color="grey", lw=0.8)
ax.set_ylabel("доля провалов на упомянутой станции"); ax.set_title("S1: оповещение и провал спроса")
plt.tight_layout(); plt.savefig(OUT / "sync_check.png", dpi=120); plt.close()
assert sync["местное время (после перевода)"].idxmax() in (0, 1), "после перевода максимум не на 0/+1 — проверить S1"

# %% [markdown]
# ## 3. Текстовая модальность на каждый (час, станция) — только оповещения с date < T (S4)

# %%
act = lk.active_rows(a, idx)
keep_t = np.where(hour_ok)[0]
x = lk.station_alert_table(a, act, idx, keep_t)
txt = lk.aggregate_text_features(x, a)
del x
print("активных пар (оповещение, час):", len(act), "| (час, станция) с оповещениями:", len(txt))

# %% [markdown]
# ## 4. Объекты, исключения, выборки

# %%
part_t = lk.part_of(idx)
gapm = lk.gap_mask(idx, gaps)
dst_bad = d.cov["is_dst_double_hour"].to_numpy().astype(bool)
dst_bad = np.convolve(dst_bad, np.ones(7), "same") > 0          # сдвоенный час и ±3 ч
in_scope = hour_ok & (part_t != "")

reasons = [
    ("станционный день закрыт (closed_wide)", C),
    ("ref не определён: < 2 открытых недель", ~np.isfinite(ref)),
    ("ref < 50 входов", np.isfinite(ref) & (ref < lk.MIN_REF)),
    ("переход на летнее/зимнее время ±3 ч", np.broadcast_to(dst_bad[:, None], C.shape)),
    ("провал выгрузки оповещений (S8)", np.broadcast_to(gapm[:, None], C.shape)),
]
ok = np.broadcast_to(in_scope[:, None], C.shape).copy()
excl_rows = []
for name, m in reasons:
    hit = ok & m
    for p in lk.SPLIT:
        excl_rows.append({"причина": name, "часть": p, "объектов": int(hit[part_t == p].sum())})
    ok &= ~m
# цель h = 2, 3 должна быть определена
for h in (2, 3):
    vv = np.zeros_like(valid); vv[:-(h - 1)] = valid[h - 1:]
    hit = ok & ~vv
    for p in lk.SPLIT:
        excl_rows.append({"причина": f"цель h={h} не определена", "часть": p, "объектов": int(hit[part_t == p].sum())})
    ok &= vv
excl_obj = pd.DataFrame(excl_rows)
excl_obj.to_csv(OUT / "excluded_objects.csv", index=False)
print(excl_obj.pivot(index="причина", columns="часть", values="объектов").to_string())

tt, ss = np.nonzero(ok)
obj = pd.DataFrame({"t": tt.astype(np.int32), "s": ss.astype(np.int16)})
obj = obj.merge(txt, on=["t", "s"], how="left")
for c in lk.TEXT_FEATURES:
    if c.startswith("txt_n") or c.endswith("_core"):
        obj[c] = obj[c].fillna(0).astype(np.float32)
for i in range(lk.TOP_ALERTS):
    obj[f"alert_{i}"] = obj[f"alert_{i}"].fillna(-1).astype(np.int32)
    obj[f"alert_lvl_{i}"] = obj[f"alert_lvl_{i}"].fillna(0).astype(np.int8)
obj["part"] = part_t[obj["t"]]
obj["stratum"] = lk.stratum(obj)
print(pd.crosstab(obj["part"], obj["stratum"]))
del txt

# %%
rng = np.random.default_rng(lk.SEED)
u = rng.random(len(obj))
keep = np.zeros(len(obj), bool); w = np.ones(len(obj), np.float32)
is_tr, is_va = obj["part"].eq("train").to_numpy(), obj["part"].eq("val").to_numpy()
a_str = obj["stratum"].eq("a").to_numpy()
keep |= is_tr & a_str
rest = is_tr & ~a_str & (u < lk.TRAIN_REST_FRAC)
keep |= rest; w[rest] = 1 / lk.TRAIN_REST_FRAC
keep |= is_va & (u < lk.VAL_FRAC)
keep |= obj["part"].isin(["test", "holdout"]).to_numpy()
obj = obj[keep].reset_index(drop=True); obj["weight"] = w[keep]

obj["T"] = idx[obj["t"]]
obj["station"] = np.array(d.stations)[obj["s"]]
for h in lk.HORIZONS:
    obj[f"y_drop_h{h}" if h > 1 else "y_drop"] = drop[obj["t"] + h - 1, obj["s"]].astype(np.int8)
obj["ref"] = ref[obj["t"], obj["s"]]
obj["y"] = Y[obj["t"], obj["s"]]
print(obj.groupby("part").agg(n=("t", "size"), drop=("y_drop", "mean"), strat_a=("stratum", lambda s: (s == "a").mean())))

# %%
num = []
for i in range(0, len(obj), 100_000):
    o = obj.iloc[i:i + 100_000]
    num.append(lk.numeric_features(d, ref, o["t"].to_numpy(), o["s"].to_numpy(), h=1))
num = pd.concat(num, ignore_index=True)
obj = pd.concat([obj, num], axis=1)
del num

# %% [markdown]
# ### E-1200 и V-60 — общие выборки для всех моделей, включая VLM

# %%
def strat_sample(df, n, seed):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("stratum")])

e1200 = strat_sample(obj[obj["part"] == "test"], lk.N_PER_STRATUM_E, lk.SEED)
v60 = strat_sample(obj[obj["part"] == "val"], lk.N_PER_STRATUM_V, lk.SEED)
obj["in_e1200"] = obj.index.isin(e1200.index); obj["in_v60"] = obj.index.isin(v60.index)
print("E-1200:", e1200.groupby("stratum")["y_drop"].agg(["size", "mean"]).to_dict())

# %% [markdown]
# ## 5. Сохранение

# %%
for p in lk.SPLIT:
    obj[obj["part"] == p].to_parquet(OUT / f"objects_{p}.parquet", index=False)
cols = ["alert_id", "event_id", "update_number", "date", "date_utc", "status_label", "affected", "header",
        "description", "text", "is_service", "is_long", "is_severe", "is_delay_only", "is_recovery"]
alerts_out = a[cols].copy()
for c in ["st_mention", "st_range", "st_nbr"]:
    alerts_out[c] = a[c].map(lambda v: ",".join(map(str, v)))
alerts_out["n_st_line"] = a["st_line"].map(len)
alerts_out.to_parquet(OUT / "alerts_clean.parquet")     # индекс = номер оповещения (alert_0…alert_4)
excl_alerts.to_csv(OUT / "excluded_alerts.csv", index=False)
unmatched.to_csv(OUT / "unmatched_mentions.csv", index=False)
gaps[gaps["is_gap"]].to_csv(OUT / "alert_gap_days.csv", index=False)
json.dump({"e1200": e1200[["T", "station"]].astype(str).values.tolist(),
           "v60": v60[["T", "station"]].astype(str).values.tolist()},
          open(OUT / "subsets.json", "w"), ensure_ascii=False)
print("сохранено, мин:", round((time.time() - t0) / 60, 1))

# %% [markdown]
# ## 6. Пример объекта, схема данных, паспорт оповещений

# %%
ex = obj[(obj["part"] == "test") & (obj["stratum"] == "a") & (obj["y_drop"] == 1)].iloc[0]
cur, prev = lk.chart_series(Y, int(ex["t"]), int(ex["s"]))
img = lk.render_chart(cur, prev); img.save(OUT / "object_example.png")
ex_text = lk.format_alerts(ex, a)
example = {"station": d.static["station_complex"].iloc[int(ex["s"])], "T": str(ex["T"]),
           "y_drop": int(ex["y_drop"]), "y": float(ex["y"]), "ref": float(ex["ref"]),
           "text": ex_text, "numbers": {c: float(ex[c]) for c in lk.NUM_FEATURES[:8] + ["nbr_r_lag1", "hour"]},
           "text_features": {c: float(ex[c]) for c in lk.TEXT_FEATURES}}
json.dump(example, open(OUT / "object_example.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(example, ensure_ascii=False, indent=1)[:1500])

# %%
schema = ["# Схема мультимодального объекта ЛР2", "",
          "Объект — (станция `s`, момент `T`), T — начало часа, 06:00–23:00, местное время Нью-Йорка.", "",
          "| Колонка | Модальность | Смысл | Доступно на момент T |", "| --- | --- | --- | --- |"]
desc = {
    "T": ("—", "момент прогноза", "да"), "station": ("—", "station_complex_id", "да"),
    "part": ("—", "train / val / test / holdout", "—"), "stratum": ("—", "слой a / b / c (п. 5 протокола)", "да"),
    "weight": ("—", "вес объекта (1/p для подвыборки train)", "—"),
    "y_drop": ("цель", "1, если y(T) < 0,75 · ref(T)", "нет — цель"), "y_drop_h2": ("цель", "то же для часа T+1", "нет"),
    "y_drop_h3": ("цель", "то же для часа T+2", "нет"), "y": ("цель", "входы в час T", "нет"),
    "ref": ("числа", "обычный уровень: медиана того же часа за 4 недели", "да"),
}
for c in obj.columns:
    if c in desc: m, s_, av = desc[c]
    elif c.startswith("r_lag") or c in ("r_3h", "nbr_r_lag1"): m, s_, av = "числа", "входы / обычный уровень в прошлые часы", "да (< T)"
    elif c.startswith("log_ref"): m, s_, av = "числа", "лог обычного уровня", "да"
    elif c in lk.CAL_COLS: m, s_, av = "числа", "календарь часа цели", "да (заранее)"
    elif c.startswith("wx_"): m, s_, av = "числа", "погода, наблюдённая в T−1", "да (< T)"
    elif c.startswith("st_"): m, s_, av = "числа", "признак станции (по train)", "да"
    elif c.startswith("txt_"): m, s_, av = "текст", "структурный признак активных оповещений", "да (date < T)"
    elif c.startswith("alert_lvl"): m, s_, av = "текст", "релевантность оповещения: 4 упомянута, 3 участок, 2 сосед, 1 линия", "да"
    elif c.startswith("alert_"): m, s_, av = "текст", "номер оповещения в alerts_clean.parquet (−1 — нет)", "да (date < T)"
    else: m, s_, av = "служебное", "", ""
    schema.append(f"| `{c}` | {m} | {s_} | {av} |")
schema += ["", "Изображение строится на лету: `lk.render_chart(*lk.chart_series(Y, t, s))` — 48 ч до T "
           "и те же 48 ч неделю назад (S9)."]
(OUT / "data_schema.md").write_text("\n".join(schema), encoding="utf-8")

# %%
def md_table(df):
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)

by_year = raw.groupby(raw["date"].dt.year).size().rename("строк").to_frame()
types = a["types"].explode().value_counts().head(15).rename("оповещений").to_frame()
types.index.name = "тип"
p = ["# Паспорт данных: служебные оповещения MTA (метро)", "",
     "| Параметр | Значение |", "| --- | --- |",
     "| Источник | MTA Service Alerts: Beginning April 2020, data.ny.gov `7kct-peq7` |",
     "| Фильтр | agency = NYCT Subway |",
     f"| Период | {a['date'].min():%d.%m.%Y %H:%M} – {a['date'].max():%d.%m.%Y %H:%M} (местное время) |",
     f"| Строк в выгрузке | {len(raw):,} |", f"| После очистки | {len(a):,} |",
     f"| Событий (event_id) | {a['event_id'].nunique():,} |",
     f"| Обновлений на событие (медиана / макс.) | {a.groupby('event_id').size().median():.0f} / {a.groupby('event_id').size().max()} |",
     f"| Длина header, символов (медиана) | {a['header'].str.len().median():.0f} |",
     f"| description заполнено | {a['description'].notna().mean():.1%} |",
     f"| Упомянута хотя бы одна станция | {(a['st_mention'].map(len) > 0).mean():.1%} |",
     f"| Есть участок «from X to Y» | {(a['st_range'].map(len) > 0).mean():.1%} |",
     f"| Станций на линиях оповещения (среднее) | {a['st_line'].map(len).mean():.1f} |",
     f"| Сообщений о восстановлении | {a['is_recovery'].mean():.1%} |",
     f"| Дней-провалов выгрузки | {int(gaps['is_gap'].sum())} |", "",
     "## Проблемы качества и решения", "",
     "| Проблема | Объём | Решение |", "| --- | --- | --- |",
     "| Время `date` в UTC, а не местное (обнаружено проверкой S1) | все строки | перевод в America/New_York; см. sync_check.png |",
     f"| Нет оповещений в последний день месяца (по UTC) с 02.2025 и 18–30.11.2024 | {int(gaps['is_gap'].sum())} сут | объекты в эти интервалы исключаются (S8) |",
     f"| Повторы (event_id, update_number) | {int((excl_alerts['reason'].str.contains('event_id')).sum())} | удалены (S3) |",
     f"| Только Staten Island Railway | {int((excl_alerts['reason'].str.startswith('S5')).sum())} | удалены (S5) |",
     f"| Названия станций без однозначной привязки | {len(unmatched)} упоминаний | журнал unmatched_mentions.csv |",
     "| Набор кончается 11.08.2026 | — | оценка до 11.08.2026 |",
     "| Почти нет анонсов плановых работ | ~3% текстов | постановка — прогноз на ближайший час |", "",
     "## Объём по годам (исходная выгрузка)", "", md_table(by_year), "",
     "## Типы оповещений (status_label)", "", md_table(types), "",
     "## Объекты", "", md_table(pd.crosstab(obj["part"], obj["stratum"], margins=True)), "",
     "## Доля провалов", "", md_table(obj.groupby(["part", "stratum"])["y_drop"].mean().round(4).unstack())]
(OUT / "alerts_passport.md").write_text("\n".join(p), encoding="utf-8")
print("\n".join(p)[:3000])
print("готово, мин:", round((time.time() - t0) / 60, 1))
