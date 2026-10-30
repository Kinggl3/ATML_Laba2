# %% [markdown]
# # ЛР2 · эксперимент 2 · lab_07 — радарные осадки MRMS (CPU, интернет)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA`, вывод `lab-00-data`.
# **Accelerator:** нет. **Internet:** включён. Время: ~1–3 ч (зависит от скорости IEM).
#
# Источник — архив Iowa Environmental Mesonet: растр `mrms_p1h` (осадки за последний час по данным
# радаров NEXRAD, система NCEP MRMS), PNG 8 бит, сетка широта/долгота 0,01° (~1 км).
# Индекс пикселя → мм по таблице IEM: 0–100 — шаг 0,25 мм; 100–180 — 1,25 мм; 180–254 — 5 мм; 255 — нет данных.
#
# Для каждого часа прогноза T (06–23, местное время) берётся последний файл, вышедший не позже T−5 мин
# (правило S11), и файл на час раньше. Из всей карты США сохраняется только вырезка вокруг Нью-Йорка.

# %%
import os, sys, re, json, time, io, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from PIL import Image


def _find_labkit():
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if "labkit.py" in fn:
                return dp
    raise FileNotFoundError("labkit.py не найден: подключите вывод ноутбука lab-00-data")


sys.path.insert(0, _find_labkit())
import labkit as lk
OUT = lk.results_dir()
st = lk.load().static
LAT, LON = st["latitude"].to_numpy(), st["longitude"].to_numpy()
MARGIN = 0.25                                       # запас вокруг станций, градусы (вырезка 32 пикселя — 0,32°)
BBOX = (LON.min() - MARGIN, LON.max() + MARGIN, LAT.min() - MARGIN, LAT.max() + MARGIN)
DAY0 = pd.Timestamp(os.environ.get("RADAR_START", "2023-01-01"))
DAY1 = pd.Timestamp(os.environ.get("RADAR_END", "2026-08-11"))
LAG_MIN = 5
ARCH = "https://mesonet.agron.iastate.edu/archive/data/{d:%Y/%m/%d}/GIS/mrms/"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ATML-Laba2 student project (Kaggle notebook)"
print("bbox", np.round(BBOX, 3), DAY0.date(), "…", DAY1.date())


def get(url, kind="bytes", tries=5):
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if kind == "bytes" else r.text
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                return None
            time.sleep(2 * (i + 1))


def p1h_mm(idx):
    """Индекс пикселя mrms_p1h → мм (255 — нет данных → NaN)."""
    idx = np.asarray(idx, dtype="float32")
    mm = np.where(idx <= 100, 0.25 * idx, np.where(idx <= 180, 25 + 1.25 * (idx - 100), 125 + 5 * (idx - 180)))
    return np.where(idx >= 255, np.nan, mm).astype("float32")

# %% [markdown]
# ## Список файлов по UTC-дням и выбор кадров для каждого часа T

# %%
def list_day(day_utc):
    html = get(ARCH.format(d=day_utc), kind="text")
    if not html:
        return []
    return sorted({pd.Timestamp(m) for m in re.findall(r"p1h_(\d{12})\.png", html)})


t0 = time.time()
utc_days = pd.date_range(DAY0 - pd.Timedelta(days=1), DAY1 + pd.Timedelta(days=1), freq="D")
with ThreadPoolExecutor(8) as ex:
    lists = list(ex.map(list_day, utc_days))
avail = pd.DatetimeIndex(sorted({t for L in lists for t in L}))
print(f"файлов в архиве: {len(avail)} за {time.time() - t0:.0f} c; дней без списка: {sum(len(L) == 0 for L in lists)}")
step = pd.Series(avail).diff().dropna().dt.total_seconds().div(60).round().value_counts().head(5)
print("шаг между файлами, мин:", step.to_dict())

hours = pd.date_range(DAY0, DAY1 + pd.Timedelta(hours=23), freq="h")
hours = hours[hours.hour.isin(lk.HOURS)]
T_utc = hours.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC").tz_localize(None)
ok = ~T_utc.isna()
hours, T_utc = hours[ok], T_utc[ok]
av = avail.values


def pick(limit):
    """Индекс последнего файла с временем ≤ limit (и не старше limit − 70 мин), иначе −1."""
    i = np.searchsorted(av, limit.values, side="right") - 1
    good = (i >= 0) & ((limit.values - av[np.clip(i, 0, None)]) <= np.timedelta64(70, "m"))
    return np.where(good, i, -1)


cur = pick(T_utc - pd.Timedelta(minutes=LAG_MIN))
prev = pick(T_utc - pd.Timedelta(minutes=LAG_MIN + 60))
need = np.unique(np.concatenate([cur[cur >= 0], prev[prev >= 0]]))
print(f"часов T: {len(hours)}; без текущего кадра: {(cur < 0).mean():.2%}; к загрузке файлов: {len(need)}")

# %% [markdown]
# ## Геопривязка (world-файл) и загрузка с обрезкой

# %%
def world(ts):
    txt = get(ARCH.format(d=ts) + f"p1h_{ts:%Y%m%d%H%M}.wld", kind="text")
    if txt:
        v = [float(x) for x in txt.split()[:6]]
        return {"dx": v[0], "dy": v[3], "x0": v[4], "y0": v[5], "source": "wld"}
    return {"dx": 0.01, "dy": -0.01, "x0": -129.995, "y0": 54.995, "source": "fallback MRMS CONUS"}


W = world(avail[need[0]])
W_last = world(avail[need[-1]])
print("world-файл:", W, "| последний:", W_last)
assert abs(W["dx"] - W_last["dx"]) < 1e-9 and abs(W["x0"] - W_last["x0"]) < 1e-6, "сетка менялась — проверить"
c0 = int(np.floor((BBOX[0] - W["x0"]) / W["dx"])); c1 = int(np.ceil((BBOX[1] - W["x0"]) / W["dx"])) + 1
r0 = int(np.floor((BBOX[3] - W["y0"]) / W["dy"])); r1 = int(np.ceil((BBOX[2] - W["y0"]) / W["dy"])) + 1
GRID = {"lon0": W["x0"] + c0 * W["dx"], "lat0": W["y0"] + r0 * W["dy"], "dx": W["dx"], "dy": W["dy"],
        "h": r1 - r0, "w": c1 - c0, "world_source": W["source"]}
print("вырезка:", GRID)


def fetch(k):
    ts = avail[k]
    raw = get(ARCH.format(d=ts) + f"p1h_{ts:%Y%m%d%H%M}.png")
    if raw is None:
        return k, None
    try:
        im = Image.open(io.BytesIO(raw))
        if im.mode != "P" and im.mode != "L":
            return k, None
        a = np.asarray(im.crop((c0, r0, c1, r1)), dtype="uint8")
        return k, a
    except Exception:  # noqa: BLE001
        return k, None


CROPS = np.full((len(need), GRID["h"], GRID["w"]), 255, dtype="uint8")
pos = {k: i for i, k in enumerate(need)}
got = np.zeros(len(need), bool)
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for j, (k, a) in enumerate(ex.map(fetch, need)):
        if a is not None and a.shape == CROPS.shape[1:]:
            CROPS[pos[k]] = a; got[pos[k]] = True
        if j % 2000 == 0:
            print(f"{j}/{len(need)}  {time.time() - t0:.0f} c  получено {got[:j + 1].mean():.1%}")
print(f"загружено {got.mean():.2%} файлов за {(time.time() - t0) / 60:.0f} мин")

# %% [markdown]
# ## Сохранение и проверка: осадки по радару против осадков ERA5 (город, тот же час)

# %%
fr = pd.DataFrame({"frame": np.arange(len(need)), "time_utc": avail[need], "ok": got})
fr["time_local"] = fr["time_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
remap = np.full(len(av), -1); remap[need] = np.arange(len(need)); remap[need[~got]] = -1
hm = pd.DataFrame({"T": hours, "cur": np.where(cur >= 0, remap[np.clip(cur, 0, None)], -1),
                   "prev": np.where(prev >= 0, remap[np.clip(prev, 0, None)], -1)})
np.save(OUT / "radar_nyc.npy", CROPS)
fr.to_parquet(OUT / "radar_frames.parquet", index=False)
hm.to_parquet(OUT / "radar_hours.parquet", index=False)
json.dump({**GRID, "lag_min": LAG_MIN, "files": int(len(need)), "loaded_share": float(got.mean()),
           "hours_without_frame": float((hm["cur"] < 0).mean()), "step_minutes": step.to_dict()},
          open(OUT / "radar_meta.json", "w"), indent=1, default=str)

cov = lk.load().cov
city = np.array([np.nanmean(p1h_mm(CROPS[i])) if i >= 0 else np.nan for i in hm["cur"]])
era = cov["precipitation"].reindex(hm["T"] - pd.Timedelta(hours=1)).to_numpy()     # ERA5: осадки за час до T
mask = np.isfinite(city) & np.isfinite(era)
rows = []
for lag in range(-3, 4):
    e = cov["precipitation"].reindex(hm["T"] - pd.Timedelta(hours=1) + pd.Timedelta(hours=lag)).to_numpy()
    m = np.isfinite(city) & np.isfinite(e)
    rows.append({"сдвиг ERA5, ч": lag, "корреляция": float(np.corrcoef(city[m], e[m])[0, 1])})
sync = pd.DataFrame(rows)
sync.to_csv(OUT / "radar_sync_check.csv", index=False)
print(sync.round(3).to_string(index=False))
print("пик корреляции должен быть на сдвиге 0 (или −1, если кадр взят на час раньше T)")
