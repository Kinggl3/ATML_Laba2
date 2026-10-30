# %% [markdown]
# # ЛР2 · эксперимент 2 · lab_06 — новые текстовые источники (CPU, интернет)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA`, вывод `lab-00-data`.
# **Accelerator:** нет (GPU не обязателен). **Internet:** включён. Время: ~40–60 мин.
#
# Три источника, каждый в своём блоке; сбой одного не останавливает остальные (итог — `sources_status.json`):
# 1. предупреждения NWS (Iowa Environmental Mesonet) → привязка к району станции;
# 2. мероприятия с разрешением города (NYC Open Data) → привязка к станциям по названиям улиц + 800 м;
# 3. расписания MTA с поправками (data.ny.gov) → число остановок по расписанию: станция × дата.
#
# Схемы наборов MTA и NYC заранее неизвестны: ноутбук сам читает список колонок, печатает его
# и сохраняет в `probe.json`. Всё время — местное время Нью-Йорка без часового пояса, как в ЛР2.

# %%
import os, sys, re, json, time, math, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests


def _find_labkit():
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if "labkit.py" in fn:
                return dp
    raise FileNotFoundError("labkit.py не найден: подключите вывод ноутбука lab-00-data")


sys.path.insert(0, _find_labkit())
import labkit as lk
OUT = lk.results_dir()
d = lk.load()
ST = d.static.reset_index()
S = len(ST)
START, END = pd.Timestamp("2022-12-01"), pd.Timestamp("2026-08-12")    # с запасом под ref и holdout
STATUS, PROBE = {}, {}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ATML-Laba2 student project (Kaggle notebook)"


def get(url, params=None, kind="json", tries=5, timeout=120):
    """GET с повторами; kind = json | text | bytes. Ошибка после всех попыток — исключение."""
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(f"404 {r.url}")
            r.raise_for_status()
            return r.json() if kind == "json" else (r.text if kind == "text" else r.content)
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 * (i + 1))


def to_local(ts):
    """UTC → местное время Нью-Йорка без часового пояса."""
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)


def save_status():
    json.dump(STATUS, open(OUT / "sources_status.json", "w"), ensure_ascii=False, indent=1, default=str)
    json.dump(PROBE, open(OUT / "probe.json", "w"), ensure_ascii=False, indent=1, default=str)


print(S, "станций;", d.y.index[0], "…", d.y.index[-1])

# %% [markdown]
# ## 1. Предупреждения NWS
# Список событий офиса OKX по годам (`json/vtec_events_bywfo.py`), фильтр по зонам и округам пяти районов,
# затем полный текст продукта. Привязка — район станции (у пригородов OKX свои коды, они отбрасываются).

# %%
UGC_BORO = {"NYC061": "Manhattan", "NYC005": "Bronx", "NYC047": "Brooklyn", "NYC081": "Queens",
            "NYC085": "Staten Island", "NYZ072": "Manhattan", "NYZ073": "Bronx", "NYZ074": "Staten Island",
            "NYZ075": "Brooklyn", "NYZ176": "Queens", "NYZ178": "Queens"}
PHEN_GROUP = {"FF": "flood", "FA": "flood", "FL": "flood", "SV": "storm", "TO": "storm", "HT": "heat",
              "EH": "heat", "XH": "heat", "WS": "winter", "WW": "winter", "BZ": "winter", "IS": "winter",
              "LE": "winter", "WC": "cold", "EC": "cold", "CW": "cold", "WI": "wind", "HW": "wind",
              "TR": "tropical", "TS": "tropical", "HU": "tropical", "CF": "coastal", "SS": "coastal",
              "DU": "other", "FG": "other", "AQ": "other", "SM": "other"}
IEM = "https://mesonet.agron.iastate.edu"


def nws_year(year):
    last = None
    for params in ({"wfo": "OKX", "year": year}, {"wfo": "KOKX", "year": year}):
        try:
            js = get(f"{IEM}/json/vtec_events_bywfo.py", params)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        rows = js.get("events", js) if isinstance(js, dict) else js
        df = pd.DataFrame(rows)
        if len(df) and df.get("wfo", pd.Series(["OKX"])).astype(str).str.contains("OKX").any():
            return df
    raise RuntimeError(f"vtec_events_bywfo {year}: нет данных OKX ({last})")


def nws_text(pid):
    for url in (f"{IEM}/api/1/nwstext/{pid}", f"{IEM}/cgi-bin/afos/retrieve.py?pil={pid.split('-')[-1]}&fmt=text"):
        try:
            t = get(url, kind="text", tries=3, timeout=60)
            if t and len(t) > 40:
                return t
        except Exception:  # noqa: BLE001
            pass
    return ""


def clean_text(t, limit=1200):
    t = re.sub(r"\r", "", t)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", t) if b.strip()]
    keep = [b for b in blocks if not re.fullmatch(r"[\d\sA-Z/\-.:$&]{0,80}", b)]      # служебные шапки
    s = re.sub(r"\s+", " ", " ".join(keep))
    return s[:limit]


try:
    ev = pd.concat([nws_year(y) for y in range(START.year, END.year + 1)], ignore_index=True)
    PROBE["nws_columns"] = list(ev.columns)
    print("NWS: столбцы", list(ev.columns), "строк", len(ev))
    ev = ev.explode("ugc") if ev["ugc"].map(lambda v: isinstance(v, list)).any() else ev
    ev["ugc"] = ev["ugc"].astype(str).str.strip()
    ev["borough"] = ev["ugc"].map(UGC_BORO)
    nyc = ev[ev["borough"].notna()].copy()
    for c, src in [("issued", "product_issued"), ("begin", "issue"), ("end", "expire")]:
        nyc[c] = to_local(nyc[src])
    nyc["phen"] = nyc["phenomena"].astype(str)
    nyc["sig"] = nyc["significance"].astype(str)
    nyc["group"] = nyc["phen"].map(PHEN_GROUP).fillna("other")
    nyc = nyc[(nyc["end"] >= START) & (nyc["issued"] <= END)]
    nyc = nyc.drop_duplicates(["product_id", "borough", "phen", "sig"])
    nyc = nyc[nyc["issued"].notna() & nyc["begin"].notna() & nyc["end"].notna()]
    pids = sorted(nyc["product_id"].dropna().unique())
    print("NWS: событий в районах НЙ", len(nyc), "продуктов", len(pids))
    with ThreadPoolExecutor(4) as ex:
        texts = dict(zip(pids, ex.map(nws_text, pids)))
    nyc["raw_len"] = nyc["product_id"].map(lambda p: len(texts.get(p, "")))
    names = {"W": "Warning", "Y": "Advisory", "A": "Watch", "S": "Statement"}
    head = nyc["phen"] + "." + nyc["sig"] + " " + nyc["sig"].map(names).fillna("")
    nyc["text"] = [f"{h} for {b}. " + clean_text(texts.get(p, "")) for h, b, p in
                   zip(head, nyc["borough"], nyc["product_id"])]
    nyc = nyc[["product_id", "borough", "phen", "sig", "group", "issued", "begin", "end", "text", "raw_len"]]
    nyc = nyc.sort_values("issued").reset_index(drop=True)
    nyc.to_parquet(OUT / "nws.parquet", index=False)
    STATUS["nws"] = {"ok": True, "rows": len(nyc), "products": len(pids),
                     "text_found_share": float((nyc["raw_len"] > 0).mean()),
                     "by_group": nyc.groupby("group").size().to_dict(),
                     "by_year": nyc.groupby(nyc["issued"].dt.year).size().to_dict()}
    print(STATUS["nws"])
    print(nyc.sample(min(3, len(nyc)), random_state=0)["text"].str[:300].to_string())
except Exception as e:  # noqa: BLE001
    STATUS["nws"] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
    print("NWS: ОШИБКА", e)
save_status()

# %% [markdown]
# ## 2. Мероприятия с разрешением города
# Исторический набор `bkfu-528j` + текущий `tvpp-9vvx` (NYC Open Data, Socrata), с 12.2022.
# Привязка к станциям: совпадение улиц из `event_location` с улицами из названия станции в том же районе
# (номерная улица — ещё и попадание в диапазон «between X and Y»), затем все станции в радиусе 800 м.

# %%
NYC_OD = "https://data.cityofnewyork.us"
RADIUS_KM = 0.8
MAX_HOURS = 48          # дольше двух суток — съёмки, сезонные площадки: в эксперимент не берём


def socrata_columns(base, ds):
    meta = get(f"{base}/api/views/{ds}.json")
    cols = [{"field": c.get("fieldName"), "name": c.get("name"), "type": c.get("dataTypeName")}
            for c in meta.get("columns", [])]
    return cols, meta


def pick(cols, *patterns, types=None):
    for pat in patterns:
        for c in cols:
            if re.search(pat, c["field"] or "", re.I) and (types is None or c["type"] in types):
                return c["field"]
    return None


def socrata_rows(base, ds, where=None, select=None, group=None, order=None, page=50000, max_rows=5_000_000):
    out, off = [], 0
    while off < max_rows:
        params = {"$limit": page, "$offset": off}
        if where: params["$where"] = where
        if select: params["$select"] = select
        if group: params["$group"] = group
        if order: params["$order"] = order
        rows = get(f"{base}/resource/{ds}.json", params)
        out.extend(rows)
        if len(rows) < page:
            break
        off += page
    return pd.DataFrame(out)


WORD = {"street": "st", "streets": "st", "sts": "st", "avs": "av", "avenue": "av", "ave": "av", "boulevard": "blvd", "road": "rd",
        "place": "pl", "parkway": "pkwy", "square": "sq", "drive": "dr", "lane": "ln", "court": "ct",
        "terrace": "ter", "plaza": "plz", "highway": "hwy", "expressway": "expwy", "saint": "st",
        "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
        "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12"}
DIRS = {"west", "east", "w", "e"}       # North/South оставляем: в Бруклине «N 7 St» и «7 St» — разные улицы
SIDE = {"north": "n", "south": "s"}


def norm_street(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)
    toks = [SIDE.get(t, WORD.get(t, t)) for t in s.split()]
    if len(toks) > 1 and toks[0] in DIRS and toks[1].isdigit():       # «W 4 St» = «4 St», но East Broadway ≠ Broadway
        toks = toks[1:]
    return " ".join(toks).strip()


def station_streets(name):
    base = re.sub(r"\s*\([^()]*\)\s*$", "", name)
    return {norm_street(p) for p in re.split(r"[-/]", base) if norm_street(p)}


def street_num(s):
    m = re.fullmatch(r"(\d+) st", s)
    return int(m.group(1)) if m else None


ST_STREETS = [station_streets(n) for n in ST["station_complex"]]
ST_BORO = ST["borough"].to_numpy()
DIST = np.load(d.root / "graph" / "dist_km.npy")


def parse_location(loc):
    """→ список сегментов (главная улица, {поперечные}, (мин, макс) номеров поперечных улиц)."""
    segs = []
    for part in re.split(r",|;", str(loc)):
        m = re.match(r"\s*(.+?)\s+between\s+(.+?)\s+and\s+(.+)$", part, re.I)
        if m:
            main, a, b = (norm_street(x) for x in m.groups())
            nums = [n for n in (street_num(a), street_num(b)) if n is not None]
            segs.append((main, {a, b}, (min(nums), max(nums)) if len(nums) == 2 else None))
        elif part.strip():
            main = norm_street(re.sub(r":.*$", "", part))
            segs.append((main, set(), None))
    return segs


def spread_km(idx):
    idx = list(idx)
    return float(DIST[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0.0


def link_event(loc, boro):
    """Станции, к которым относится мероприятие (точность важнее полноты: при сомнении — не привязываем).
    1) в названии станции главная улица и поперечная (или номер из диапазона between) — берём их;
    2) есть станции на главной улице и на поперечных — берём пары ближе 1,5 км друг к другу;
    3) главная улица — номерная (короткая): до четырёх станций с ней в радиусе 3 км;
    4) поперечные улицы, если главная длинная (авеню, Broadway) или не нашлась: станции кучно (≤ 2 км);
    5) только главная улица (площадь, парк): до трёх станций в радиусе 2 км.
    Станции, чей номер улицы противоречит диапазону between, не берутся."""
    direct = set()
    cand = np.where(ST_BORO == boro)[0]
    for main, cross, rng in parse_location(loc):
        strong, by_cross, by_main = set(), set(), set()
        for s in cand:
            toks = ST_STREETS[s]
            nums = [n for n in (street_num(t) for t in toks) if n is not None]
            in_rng = bool(rng) and any(rng[0] - 2 <= n <= rng[1] + 2 for n in nums)
            on_main = bool(main) and (main in toks or any(" " in t and len(t) >= 6 and re.search(rf"\b{re.escape(t)}\b", main)
                                                         for t in toks))
            if on_main and (toks & cross or in_rng):
                strong.add(s)
            elif toks & cross:
                by_cross.add(s)
            elif on_main and not (rng and nums):          # номер есть, но вне диапазона — противоречие
                by_main.add(s)
        if strong:
            direct |= strong
            continue
        pairs = {x for m in by_main for c in by_cross if DIST[m, c] <= 1.5 for x in (m, c)}
        main_short = street_num(main) is not None                  # номерная улица — короткая, ей можно верить
        main_long = main.endswith(("av", "broadway", "blvd", "pkwy")) or main == "broadway"
        if pairs:
            direct |= pairs
        elif main_short and by_main and len(by_main) <= 4 and spread_km(by_main) <= 3.0:
            direct |= by_main
        elif by_cross and (not by_main or main_long) and spread_km(by_cross) <= 2.0:
            direct |= by_cross
        elif by_main and not by_cross and len(by_main) <= 3 and spread_km(by_main) <= 2.0:
            direct |= by_main
    near = set()
    for s in direct:
        near |= set(np.where(DIST[s] <= RADIUS_KM)[0])
    return sorted(direct), sorted(near - direct)


try:
    frames, used = [], []
    for ds in ["bkfu-528j", "tvpp-9vvx"]:
        try:
            cols, meta = socrata_columns(NYC_OD, ds)
        except Exception as e:  # noqa: BLE001
            print(ds, "недоступен:", e)
            continue
        PROBE[f"permits_{ds}_columns"] = cols
        f_start = pick(cols, r"start", types={"floating_timestamp", "calendar_date", "text"})
        f_end = pick(cols, r"end", types={"floating_timestamp", "calendar_date", "text"})
        print(ds, "колонки:", [c["field"] for c in cols], "| start:", f_start, "end:", f_end)
        where = f"{f_start} >= '{START:%Y-%m-%dT%H:%M:%S}' AND {f_start} < '{END:%Y-%m-%dT%H:%M:%S}'"
        df = socrata_rows(NYC_OD, ds, where=where)
        df["_src"] = ds
        frames.append(df); used.append(ds)
        print(ds, "строк", len(df))
    raw = pd.concat(frames, ignore_index=True)
    cols = [{"field": c, "type": ""} for c in raw.columns]
    F = {k: pick(cols, *p) for k, p in {
        "id": [r"event_id", r"^id$"], "name": [r"event_name", r"name"], "type": [r"event_type", r"type"],
        "start": [r"start_date_time", r"start"], "end": [r"end_date_time", r"end"],
        "boro": [r"borough"], "loc": [r"event_location", r"location"],
        "closure": [r"closure"], "agency": [r"agency"]}.items()}
    PROBE["permits_fields"] = F
    print("поля:", F)
    ev = pd.DataFrame({k: raw[v] if v else None for k, v in F.items()})
    ev["start"] = pd.to_datetime(ev["start"], errors="coerce")
    ev["end"] = pd.to_datetime(ev["end"], errors="coerce")
    ev = ev.dropna(subset=["start", "end", "loc"])
    ev = ev.drop_duplicates(subset=[c for c in ["id", "name", "start", "loc"] if ev[c].notna().any()])
    ev["hours"] = (ev["end"] - ev["start"]) / pd.Timedelta(hours=1)
    ev["type"] = ev["type"].astype(str)
    PROBE["permits_type_counts"] = ev["type"].value_counts().head(60).to_dict()
    print(ev["type"].value_counts().head(30).to_string())
    drop_type = ev["type"].str.contains(r"sport|production|construction|filming|film|theater load", case=False)
    keep = (~drop_type) & (ev["hours"] > 0) & (ev["hours"] <= MAX_HOURS)
    ev = ev[keep].copy()
    ev["boro"] = ev["boro"].astype(str).str.strip().str.title().replace({"The Bronx": "Bronx"})
    t0 = time.time()
    links = [link_event(l, b) for l, b in zip(ev["loc"], ev["boro"])]
    ev["direct"] = [a for a, _ in links]
    ev["near"] = [b for _, b in links]
    ev["full_closure"] = ev["closure"].astype(str).str.contains("full", case=False) if F["closure"] else False
    ev["text"] = (ev["type"] + " — " + ev["name"].astype(str) + " — " + ev["loc"].astype(str).str[:200] + " — "
                  + ev["start"].dt.strftime("%a %H:%M") + "–" + ev["end"].dt.strftime("%H:%M")
                  + np.where(ev["full_closure"], " — full street closure", ""))
    ev = ev.sort_values("start").reset_index(drop=True)
    linked = ev["direct"].map(len) > 0
    print(f"привязка {time.time() - t0:.0f} c; привязано {linked.mean():.1%} мероприятий")
    pairs = [(i, s, 2) for i, a in enumerate(ev["direct"]) for s in a] + \
            [(i, s, 1) for i, b in enumerate(ev["near"]) for s in b]
    pd.DataFrame(pairs, columns=["event", "s", "level"]).to_parquet(OUT / "permit_station.parquet", index=False)
    ev.drop(columns=["direct", "near"]).to_parquet(OUT / "permits.parquet", index=False)
    ev[~linked].groupby("type").size().sort_values(ascending=False).head(30).to_csv(OUT / "permits_unlinked_types.csv")
    ev[~linked]["loc"].sample(min(300, (~linked).sum()), random_state=0).to_csv(OUT / "permits_unlinked_sample.csv")
    STATUS["permits"] = {"ok": True, "datasets": used, "events": len(ev), "linked_share": float(linked.mean()),
                         "pairs": len(pairs), "by_year": ev.groupby(ev["start"].dt.year).size().to_dict()}
    print(STATUS["permits"])
except Exception as e:  # noqa: BLE001
    STATUS["permits"] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
    print("Мероприятия: ОШИБКА", e)
save_status()

# %% [markdown]
# ## 3. Плановые работы: расписания MTA с поправками
# Наборы «MTA Subway Schedules» по годам. Нужна колонка с датой, на которую действует строка расписания:
# без неё поправки не привязать к дню, и источник честно отмечается как непригодный.
# Сервер Socrata сам считает число остановок: `count(*)` по (остановка, дата).

# %%
NY = "https://data.ny.gov"
SCHED = {2023: "7pnn-mafy", 2024: "ebrw-j62c", 2025: "q9nv-uegs", 2026: "g8es-h7gb"}
STATIONS_DS = "39hk-dx4f"         # MTA Subway Stations: GTFS Stop ID → Complex ID


def month_starts(a, b):
    return list(pd.date_range(a.normalize().replace(day=1), b, freq="MS"))


try:
    # GTFS stop_id → station_complex_id
    cols_st, _ = socrata_columns(NY, STATIONS_DS)
    PROBE["stations_columns"] = cols_st
    f_stop = pick(cols_st, r"gtfs_stop_id", r"stop_id")
    f_cx = pick(cols_st, r"complex_id", r"complex")
    stn = socrata_rows(NY, STATIONS_DS, select=f"{f_stop},{f_cx}")
    stop2cx = dict(zip(stn[f_stop].astype(str).str.strip().str.upper(), stn[f_cx].astype(str).str.strip()))
    cx_index = {c: i for i, c in enumerate(ST["station_complex_id"].astype(str))}
    print("станций MTA:", len(stop2cx), "| поля", f_stop, f_cx)

    parts, info = [], {}
    for year, ds in SCHED.items():
        try:
            cols, meta = socrata_columns(NY, ds)
            PROBE[f"sched_{year}_columns"] = cols
            PROBE[f"sched_{year}_description"] = meta.get("description", "")[:2000]
            sample = get(f"{NY}/resource/{ds}.json", {"$limit": 5})
            PROBE[f"sched_{year}_sample"] = sample
            f_stop_s = pick(cols, r"gtfs_stop_id", r"stop_id", r"stop")
            f_date = (pick(cols, r"service_date", r"^date$", r"date", types={"floating_timestamp", "calendar_date"})
                      or pick(cols, r"service_date", r"^date$"))
            f_supp = pick(cols, r"supplement", r"schedule_type", r"plan")
            date_type = next((c["type"] for c in cols if c["field"] == f_date), None)
            info[year] = {"stop": f_stop_s, "date": f_date, "date_type": date_type, "supplement": f_supp,
                          "fields": [c["field"] for c in cols]}
            print(year, ds, info[year])
            if not (f_stop_s and f_date):
                continue
            day_expr = f"date_trunc_ymd({f_date})" if date_type in ("floating_timestamp", "calendar_date") else f_date
            for ms in month_starts(max(START, pd.Timestamp(f"{year}-01-01")), min(END, pd.Timestamp(f"{year}-12-31"))):
                me = ms + pd.offsets.MonthBegin(1)
                where = f"{f_date} >= '{ms:%Y-%m-%dT00:00:00}' AND {f_date} < '{me:%Y-%m-%dT00:00:00}'"
                if day_expr == f_date:          # дата текстом: сравниваем строки вида YYYY-MM-DD
                    where = f"{f_date} >= '{ms:%Y-%m-%d}' AND {f_date} < '{me:%Y-%m-%d}'"
                sel = f"{f_stop_s} AS stop, {day_expr} AS day, count(*) AS n"
                df = socrata_rows(NY, ds, where=where, select=sel, group=f"{f_stop_s}, {day_expr}")
                if len(df):
                    parts.append(df)
                print(f"  {ms:%Y-%m}: {len(df)} строк")
        except Exception as e:  # noqa: BLE001
            info[year] = {**info.get(year, {}), "error": repr(e)}
            print(year, "ОШИБКА", e)
    STATUS["schedules_fields"] = info
    if not parts:
        raise RuntimeError("ни в одном наборе расписаний не найдена колонка даты действия — "
                           "поправки нельзя привязать к дням (см. probe.json)")
    sc = pd.concat(parts, ignore_index=True)
    sc["stop"] = sc["stop"].astype(str).str.strip().str.upper()
    sc["cx"] = sc["stop"].map(stop2cx)
    sc.loc[sc["cx"].isna(), "cx"] = sc.loc[sc["cx"].isna(), "stop"].str.rstrip("NS").map(stop2cx)
    sc["s"] = sc["cx"].map(cx_index)
    sc["day"] = pd.to_datetime(sc["day"]).dt.normalize()
    sc["n"] = pd.to_numeric(sc["n"])
    miss = sc["s"].isna().mean()
    daily = sc.dropna(subset=["s"]).groupby(["day", "s"], as_index=False)["n"].sum()
    daily["s"] = daily["s"].astype(int)
    daily.to_parquet(OUT / "sched_daily.parquet", index=False)
    STATUS["schedules"] = {"ok": True, "rows": len(daily), "days": int(daily["day"].nunique()),
                           "unmapped_stop_share": float(miss)}
    print(STATUS["schedules"])
except Exception as e:  # noqa: BLE001
    STATUS["schedules"] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
    print("Расписания: ОШИБКА", e)
save_status()

# %% [markdown]
# ## 4. Эмбеддинги текстов (bge-small-en-v1.5, как в ЛР2)

# %%
try:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
    for name in ["nws", "permits"]:
        f = OUT / f"{name}.parquet"
        if not f.exists():
            continue
        tx = pd.read_parquet(f, columns=["text"])["text"].fillna("").tolist()
        t0 = time.time()
        E = st_model.encode(tx, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        np.save(OUT / f"{name}_emb.npy", E.astype("float16"))
        STATUS.setdefault(name, {})["emb_sec"] = round(time.time() - t0, 1)
        print(name, E.shape, f"{time.time() - t0:.0f} c на {dev}")
except Exception as e:  # noqa: BLE001
    STATUS["embeddings"] = {"ok": False, "error": repr(e)}
    print("Эмбеддинги: ОШИБКА", e)
save_status()

# %% [markdown]
# ## Итог по источникам

# %%
lines = ["# Новые источники: паспорт (эксперимент 2)", ""]
for k in ["nws", "permits", "schedules"]:
    v = STATUS.get(k, {})
    lines.append(f"## {k}: {'готов' if v.get('ok') else 'НЕ ПОЛУЧЕН'}")
    lines += [f"- {kk}: {vv}" for kk, vv in v.items() if kk != "trace"]
    lines.append("")
(OUT / "sources_passport.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
