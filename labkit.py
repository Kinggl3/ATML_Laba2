"""
labkit — общий код ЛР2: данные, синхронизация оповещений, объекты, изображения, метрики, журнал.

Один файл для всех ноутбуков, чтобы протокол (PROTOCOL.md) нигде не разъехался:
одни объекты, одни правила синхронизации S1–S10, одни метрики.

На Kaggle: ноутбук lab_00 кладёт этот файл в /kaggle/working вместе с объектами; остальные ноутбуки
подключают вывод lab_00 как Input и делают `sys.path.append(<папка с labkit.py>)`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# --- постановка (PROTOCOL.md, разд. 1) ---
HOURS = tuple(range(6, 24))     # моменты прогноза T: 06:00 … 23:00
DROP_RATIO = 0.75               # провал: y < 0,75 · ref
REF_WEEKS = 4                   # ref — медиана того же часа недели за 4 прошлые недели
MIN_REF_WEEKS = 2               # нужно минимум 2 открытые недели
MIN_REF = 50.0                  # ref < 50 входов — объект исключается
HORIZONS = (1, 2, 3)

# --- разбиение (разд. 5) ---
SPLIT = {
    "train":   ("2023-01-01 00:00", "2026-03-31 23:00"),
    "val":     ("2026-04-01 00:00", "2026-05-31 23:00"),
    "test":    ("2026-06-01 00:00", "2026-07-15 23:00"),
    "holdout": ("2026-07-16 00:00", "2026-08-11 23:00"),
}
TRAIN_REST_FRAC = 0.05          # train: все объекты слоя «а» + 5% остальных (веса 1/p)
VAL_FRAC = 0.30                 # val: 30% случайных объектов (в протоколе было 50%, см. отступления)
N_PER_STRATUM_E = 400           # E-1200
N_PER_STRATUM_V = 20            # V-60

# --- оповещения (разд. 4) ---
ACTIVE_MIN = 90                 # S4: окно активности, мин
ACTIVE_MIN_LONG = 180           # S4: для приостановок, объездов, пропуска станций
GAP_DAY_RATIO = 0.25            # S8: день с провалом выгрузки
TOP_ALERTS = 5                  # до 5 оповещений в текстовой модальности

LONG_TYPES = {"suspended", "part-suspended", "reroute", "trains-rerouted", "some-reroutes",
              "stops-skipped", "stations-skipped"}
SERVICE_TYPES = LONG_TYPES | {"local-to-express", "multiple-changes", "cancellations"}
DELAY_TYPES = {"delays", "some-delays", "severe-delays", "slow-speeds"}
SEVERE_TYPES = {"suspended", "part-suspended", "severe-delays"}

ROUTE_MAP = {"H": "S", "FS": "S", "GS": "S", "6X": "6", "7X": "7", "FX": "F"}
ROUTE_DROP = {"SI", "SIM7"}     # Staten Island Railway — нет в данных о входах

RECOVERY_RE = re.compile(
    r"\b(have|has) resumed\b|\bresumed\b|\brestored\b|\bregular (service|stops)\b|\brunning normally\b|"
    r"\bback to normal\b|\bnow (making|running) (all|regular|local)\b", re.I)

TZ_NOTE = "местное время America/New_York без часового пояса (оба набора)"


# =============================================================================
# Данные
# =============================================================================
def find_data_root() -> Path:
    """Папка, где лежат lab/, graph/ и alerts/. Ищет на Kaggle, затем локально."""
    env = os.environ.get("VKR_DATA_ROOT")
    if env:
        return Path(env)
    hits = glob.glob("/kaggle/input/**/lab/split.json", recursive=True)
    if hits:
        return Path(hits[0]).parent.parent
    for cand in [Path("dataset_kaggle"), Path("../dataset_kaggle"), Path("data"), Path("../data")]:
        if (cand / "lab" / "split.json").exists():
            return cand
    raise FileNotFoundError("Не найден lab/split.json: подключите датасет vkr-transit-lab или задайте VKR_DATA_ROOT")


def find_alerts_dir(root: Path) -> Path:
    for cand in [root / "alerts", root / "raw" / "alerts"]:
        if list(cand.glob("alerts_*.parquet")):
            return cand
    hits = glob.glob("/kaggle/input/**/alerts_2023-01.parquet", recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError("Не найдена папка alerts/ с alerts_YYYY-MM.parquet")


def station_lines(name: str) -> set[str]:
    """Линии станции из последних скобок названия комплекса: 'Hewes St (M,J)' → {'M','J'}."""
    m = re.search(r"\(([^()]*)\)\s*$", name)
    if not m:
        return set()
    out = set()
    for r in m.group(1).split(","):
        r = r.strip()
        r = ROUTE_MAP.get(r, r)
        if re.fullmatch(r"[0-9A-Z]{1,2}", r) and r not in ROUTE_DROP:
            out.add(r)
    return out


def parse_routes(affected: str) -> set[str]:
    out = set()
    for r in str(affected).split("|"):
        r = r.strip().upper()
        r = ROUTE_MAP.get(r, r)
        if r and r not in ROUTE_DROP:
            out.add(r)
    return out


@dataclass
class LabData:
    y: pd.DataFrame          # час × станция, входы
    closed: pd.DataFrame     # час × станция, True = станция закрыта (маска, не признак)
    cov: pd.DataFrame        # час × ковариаты
    static: pd.DataFrame     # станции, индекс = station_complex_id, порядок = столбцы y
    nodes: pd.DataFrame
    adj_line: np.ndarray     # [S, S] соседи по линиям
    root: Path
    lines: list = field(default_factory=list)   # множество линий каждой станции

    @property
    def stations(self) -> list[str]:
        return list(self.y.columns)


def load(root: Path | None = None) -> LabData:
    root = Path(root) if root else find_data_root()
    lab = root / "lab"
    y = pd.read_parquet(lab / "ridership_wide.parquet")
    closed = pd.read_parquet(lab / "closed_wide.parquet")
    cov = pd.read_parquet(lab / "covariates_time.parquet")
    static = pd.read_parquet(lab / "station_static.parquet")
    static["station_complex_id"] = static["station_complex_id"].astype(str)
    static = static.set_index("station_complex_id")
    y.columns = y.columns.astype(str)
    closed.columns = closed.columns.astype(str)
    assert list(y.columns) == list(closed.columns) and y.index.equals(closed.index)
    static = static.loc[y.columns]
    nodes = pd.read_csv(root / "graph" / "nodes.csv", dtype={"station_complex_id": str})
    assert list(nodes["station_complex_id"]) == list(y.columns), "порядок nodes.csv ≠ столбцы ridership_wide"
    adj_line = np.load(root / "graph" / "adj_line.npy")
    lines = [station_lines(n) for n in static["station_complex"]]
    return LabData(y=y, closed=closed, cov=cov, static=static, nodes=nodes, adj_line=adj_line,
                   root=root, lines=lines)


def part_of(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Имя части разбиения для каждой метки времени ('' — вне разбиения)."""
    ts = pd.DatetimeIndex(ts)
    out = np.full(len(ts), "", dtype=object)
    for name, (a, b) in SPLIT.items():
        out[(ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))] = name
    return out


# =============================================================================
# Цель: обычный уровень и провал
# =============================================================================
def reference_level(d: LabData) -> np.ndarray:
    """ref[t, s] — медиана y[t − 168k], k = 1…4, по открытым дням; NaN, если открытых недель < 2.

    Сдвиг по позиции корректен: индекс — полная сетка часов (как в ЛР1), 168 строк = 168 часов.
    """
    Y = d.y.to_numpy(dtype="float32")
    Y = np.where(d.closed.to_numpy(), np.nan, Y)
    H, S = Y.shape
    stack = np.full((REF_WEEKS, H, S), np.nan, dtype="float32")
    for k in range(1, REF_WEEKS + 1):
        stack[k - 1, 168 * k:] = Y[:-168 * k]
    cnt = np.isfinite(stack).sum(0)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(stack, axis=0)
    ref[cnt < MIN_REF_WEEKS] = np.nan
    return ref.astype("float32")


# =============================================================================
# Оповещения: очистка (S3), линии (S5), упоминания и участки (S6–S7), активность (S4)
# =============================================================================
def load_alerts(root: Path) -> pd.DataFrame:
    adir = find_alerts_dir(root)
    a = pd.concat([pd.read_parquet(f) for f in sorted(adir.glob("alerts_*.parquet"))], ignore_index=True)
    # S1: проверка в lab_00 показала, что `date` в наборе оповещений — UTC (провалы спроса на упомянутых
    # станциях идут за 5 ч до метки зимой и за 4 ч летом). Переводим в местное время Нью-Йорка без пояса,
    # как в данных о входах. Исходное время сохраняется в date_utc.
    a["date_utc"] = pd.to_datetime(a["date"])
    a["date"] = (a["date_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                 .dt.tz_localize(None))
    for c in ["alert_id", "event_id", "update_number"]:
        a[c] = pd.to_numeric(a[c], errors="coerce").astype("Int64")
    return a


def clean_alerts(a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S3: дубликаты. Возвращает (оповещения, журнал исключённых)."""
    log = []
    n0 = len(a)
    dup = a.duplicated("alert_id")
    log.append(a[dup].assign(reason="S3: повтор alert_id"))
    a = a[~dup]
    a = a.sort_values(["event_id", "update_number", "date", "alert_id"])
    dup = a.duplicated(["event_id", "update_number"])
    log.append(a[dup].assign(reason="S3: повтор (event_id, update_number)"))
    a = a[~dup]
    # одинаковый header подряд внутри события НЕ удаляется: такие обновления подтверждают,
    # что сбой продолжается, и продлевают активность (S4); в текст всё равно идёт только последнее
    a = a.copy()
    a["routes"] = a["affected"].map(parse_routes)
    no_route = a["routes"].map(len) == 0
    log.append(a[no_route].assign(reason="S5: нет линий метро (только Staten Island)"))
    a = a[~no_route].copy()
    a["types"] = a["status_label"].fillna("").map(lambda s: {x.strip() for x in s.split("|") if x.strip()})
    a["is_service"] = a["types"].map(lambda t: bool(t & SERVICE_TYPES))
    a["is_long"] = a["types"].map(lambda t: bool(t & LONG_TYPES))
    a["is_severe"] = a["types"].map(lambda t: bool(t & SEVERE_TYPES))
    a["is_delay_only"] = a["types"].map(lambda t: bool(t) and t <= DELAY_TYPES)
    a["is_recovery"] = a["header"].fillna("").map(lambda h: bool(RECOVERY_RE.search(h)))
    a["text"] = (a["header"].fillna("").str.strip() + " "
                 + a["description"].fillna("").str.slice(0, 300).str.strip()).str.strip()
    a = a.sort_values(["date", "alert_id"]).reset_index(drop=True)
    excl = pd.concat(log, ignore_index=True)[["alert_id", "event_id", "update_number", "date", "reason"]]
    assert len(a) + len(excl) == n0
    return a, excl


def gap_days(a: pd.DataFrame, start="2023-01-01", end="2026-08-12") -> pd.DataFrame:
    """S8: сутки (по UTC — так режется выгрузка) с провалом: оповещений < 25% скользящей медианы за 28 дней.
    Колонки local_start / local_end — тот же интервал в местном времени."""
    days = pd.date_range(start, end)
    cnt = a.groupby(a["date_utc"].dt.normalize()).size().reindex(days, fill_value=0)
    med = cnt.rolling(28, center=True, min_periods=7).median()
    gap = cnt < GAP_DAY_RATIO * med
    loc = lambda x: x.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    out = pd.DataFrame({"day_utc": days, "n_alerts": cnt.values, "median28": med.values, "is_gap": gap.values})
    out["local_start"] = loc(pd.DatetimeIndex(days))
    out["local_end"] = loc(pd.DatetimeIndex(days) + pd.Timedelta(days=1))
    return out


def gap_mask(index: pd.DatetimeIndex, gaps: pd.DataFrame) -> np.ndarray:
    """True для моментов T, на которые влияет провал выгрузки: T ∈ (начало, конец + ACTIVE_MIN_LONG]."""
    m = np.zeros(len(index), bool)
    for s, e in gaps.loc[gaps["is_gap"], ["local_start", "local_end"]].itertuples(index=False):
        m |= (index > s) & (index <= e + pd.Timedelta(minutes=ACTIVE_MIN_LONG))
    return m


_SUFFIX_RE = re.compile(r"\s*\((?:brooklyn|bklyn|qns|queens|manhattan|bronx|bx)\)", re.I)


def _alias_variants(s: str) -> set[str]:
    s = s.strip()
    out = {s}
    rules = [(r"\bPl\b", "Place"), (r"\bE (?=\d)", "East "), (r"\bEast (?=\d)", "E "),
             (r"\bW (?=\d)", "West "), (r"\bAv\b", "Ave"), (r"\bSq\b", "Square")]
    for pat, rep in rules:
        for x in list(out):
            y = re.sub(pat, rep, x)
            if y != x:
                out.add(y)
    return {x for x in out if len(x) >= 3}


class StationMatcher:
    """S6: поиск названий станций в тексте с ограничением по линиям оповещения."""

    EXTRA = {"World Trade Center": "WTC", "Port Authority": "42 St-Port Authority Bus Terminal",
             "Central Park North": "110 St-Malcolm X Plaza", "Astor Place": "Astor Pl"}

    def __init__(self, d: LabData):
        self.idx = {sid: i for i, sid in enumerate(d.stations)}
        self.lines = d.lines
        alias = defaultdict(set)
        for i, name in enumerate(d.static["station_complex"]):
            base = re.sub(r"\s*\([^()]*\)\s*$", "", name)
            parts = {base} | {p.strip() for p in base.split("/")}
            for p in list(parts):
                m = re.match(r"(.*)\((.*)\)", p)
                if m:
                    parts |= {m.group(1), m.group(2), f"{m.group(1).strip()}-{m.group(2).strip()}"}
            for p in list(parts):                     # '103 St-Corona Plaza' → '103 St', 'Corona Plaza'
                if "-" in p:
                    parts |= {x for x in p.split("-") if len(x.strip()) >= 4}
            for p in parts:
                for v in _alias_variants(p):
                    alias[v].add(i)
        for k, target in self.EXTRA.items():
            for v, ids in list(alias.items()):
                if v == target or v.startswith(target):
                    alias[k] |= ids
        self.alias = dict(alias)
        keys = sorted(self.alias, key=len, reverse=True)
        self.pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w-])")

    def find(self, text: str, routes: set[str]) -> tuple[list[tuple[int, int, set]], list[str]]:
        """→ ([(start, end, {индексы станций})], [непривязанные упоминания])."""
        text = _SUFFIX_RE.sub("", text)
        hits, unmatched = [], []
        for m in self.pat.finditer(text):
            cand = self.alias[m.group(1)]
            on_route = {i for i in cand if self.lines[i] & routes}
            if len(on_route) >= 1:
                hits.append((m.start(), m.end(), on_route))
            elif len(cand) == 1:
                hits.append((m.start(), m.end(), set(cand)))
            else:
                unmatched.append(m.group(1))
        return hits, unmatched


_RANGE_RE = re.compile(r"^\s*(to|and|-|–)\s*$", re.I)


def _line_paths(adj: np.ndarray, allowed: np.ndarray, src: set, dst: set) -> set[int]:
    """S7: станции на кратчайшем пути src → dst по adj_line внутри разрешённых станций."""
    import networkx as nx
    sub = np.where(allowed)[0]
    g = nx.Graph()
    g.add_nodes_from(sub)
    sel = set(sub.tolist())
    rows, cols = np.nonzero(adj[np.ix_(sub, sub)])
    g.add_edges_from((sub[r], sub[c]) for r, c in zip(rows, cols))
    best = None
    for s in src & sel:
        for t in dst & sel:
            try:
                p = nx.shortest_path(g, s, t)
            except nx.NetworkXNoPath:
                continue
            if best is None or len(p) < len(best):
                best = p
    return set(best) if best else set()


def relate_alerts(a: pd.DataFrame, d: LabData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Для каждого оповещения: станции на его линиях, упомянутые, на участке, соседи (≤ 2 перегонов).

    Возвращает (a с колонками-списками индексов станций, журнал непривязанных упоминаний)."""
    m = StationMatcher(d)
    S = len(d.stations)
    line_sets = [frozenset(x) for x in d.lines]
    adj = (d.adj_line > 0)
    adj2 = adj | ((adj.astype(np.int32) @ adj.astype(np.int32)) > 0)
    on_line, mention, in_range, nbr, unmatched_log = [], [], [], [], []
    cache = {}
    for aid, text, routes in zip(a["alert_id"], a["header"].fillna(""), a["routes"]):
        key = (text, frozenset(routes))
        if key not in cache:
            allowed = np.array([bool(ls & routes) for ls in line_sets])
            hits, unm = m.find(text, routes)
            men = set().union(*[h[2] for h in hits]) if hits else set()
            rng = set()
            for (s1, e1, ids1), (s2, e2, ids2) in zip(hits, hits[1:]):
                before = text[max(0, s1 - 9):s1].lower()
                if _RANGE_RE.match(text[e1:s2]) and ("from" in before or "between" in before):
                    rng |= _line_paths(adj, allowed, ids1, ids2)
            core = men | rng
            nb = set(np.where(adj2[list(core)].any(0) & allowed)[0].tolist()) - core if core else set()
            cache[key] = (np.where(allowed)[0].astype(np.int16), np.array(sorted(men), np.int16),
                          np.array(sorted(rng - men), np.int16), np.array(sorted(nb), np.int16), unm)
        ol, mn, rg, nb, unm = cache[key]
        on_line.append(ol); mention.append(mn); in_range.append(rg); nbr.append(nb)
        for u in unm:
            unmatched_log.append((aid, u, text[:160]))
    a = a.copy()
    a["st_line"], a["st_mention"], a["st_range"], a["st_nbr"] = on_line, mention, in_range, nbr
    unm = pd.DataFrame(unmatched_log, columns=["alert_id", "mention", "header"])
    return a, unm


def active_rows(a: pd.DataFrame, hour_index: pd.DatetimeIndex, active_min: int = ACTIVE_MIN,
                active_min_long: int = ACTIVE_MIN_LONG) -> pd.DataFrame:
    """S4: строки (t, u) — оповещение u является последним обновлением активного события в момент T=index[t].

    Используются только обновления с date < T; события после сообщения о восстановлении не активны;
    устаревание — ACTIVE_MIN (ACTIVE_MIN_LONG для приостановок, объездов, пропусков)."""
    a = a.sort_values(["event_id", "date", "update_number"])
    nxt = a.groupby("event_id")["date"].shift(-1)
    win = np.where(a["is_long"], active_min_long, active_min).astype("timedelta64[m]")
    d0 = a["date"].to_numpy()
    end = np.minimum(nxt.fillna(pd.Timestamp.max).to_numpy(), d0 + win)   # T ≤ end
    start = pd.DatetimeIndex(d0).floor("h") + pd.Timedelta(hours=1)        # первый час T > date
    start = start.to_numpy()
    h0 = hour_index.searchsorted(start, side="left")
    h1 = hour_index.searchsorted(end, side="right")                         # T ≤ end
    ok = (~a["is_recovery"].to_numpy()) & (h1 > h0)
    rows_u = np.repeat(np.arange(len(a))[ok], (h1 - h0)[ok])
    rows_t = np.concatenate([np.arange(s, e) for s, e in zip(h0[ok], h1[ok])]) if ok.any() else np.array([], int)
    out = pd.DataFrame({"t": rows_t.astype(np.int32), "u": a.index.to_numpy()[rows_u].astype(np.int32)})
    T = hour_index[out["t"].to_numpy()]
    assert (a.loc[out["u"], "date"].to_numpy() < T.to_numpy()).all(), "утечка: оповещение позже T"
    return out


TEXT_FEATURES = ["txt_n_line", "txt_n_service_line", "txt_n_delay_line", "txt_n_mention", "txt_n_range",
                 "txt_n_nbr", "txt_service_core", "txt_severe_core", "txt_min_since", "txt_min_since_core",
                 "txt_min_since_service"]


def station_alert_table(a: pd.DataFrame, act: pd.DataFrame, hour_index: pd.DatetimeIndex,
                        keep_t: np.ndarray | None = None) -> pd.DataFrame:
    """Разворачивает активные оповещения на станции: строки (t, s, u, level, minutes).

    level: 4 — станция упомянута; 3 — на участке «from X to Y»; 2 — соседняя (≤ 2 перегонов); 1 — только линия."""
    if keep_t is not None:
        act = act[np.isin(act["t"], keep_t)]
    parts = []
    T = hour_index.to_numpy()
    for col, lvl in [("st_line", 1), ("st_nbr", 2), ("st_range", 3), ("st_mention", 4)]:
        sts = a[col].to_numpy()
        lens = np.array([len(x) for x in sts])
        n = lens[act["u"].to_numpy()]
        u = np.repeat(act["u"].to_numpy(), n)
        t = np.repeat(act["t"].to_numpy(), n)
        s = np.concatenate([sts[i] for i in act["u"].to_numpy()]) if len(act) else np.array([], np.int16)
        parts.append(pd.DataFrame({"t": t, "s": s.astype(np.int16), "u": u, "level": np.int8(lvl)}))
    x = pd.concat(parts, ignore_index=True)
    x = x.sort_values("level").drop_duplicates(["t", "s", "u"], keep="last")
    minutes = (T[x["t"].to_numpy()] - a["date"].to_numpy()[x["u"].to_numpy()]) / np.timedelta64(1, "m")
    x["minutes"] = minutes.astype("float32")
    return x.reset_index(drop=True)


def aggregate_text_features(x: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Структурные признаки текста на (t, s) — колонки TEXT_FEATURES и до TOP_ALERTS оповещений."""
    svc = a["is_service"].to_numpy()[x["u"].to_numpy()]
    dly = a["is_delay_only"].to_numpy()[x["u"].to_numpy()]
    sev = a["is_severe"].to_numpy()[x["u"].to_numpy()]
    lvl = x["level"].to_numpy()
    core = lvl >= 3
    f = pd.DataFrame({"t": x["t"], "s": x["s"],
                      "txt_n_line": 1, "txt_n_service_line": svc.astype(int), "txt_n_delay_line": dly.astype(int),
                      "txt_n_mention": (lvl == 4).astype(int), "txt_n_range": (lvl == 3).astype(int),
                      "txt_n_nbr": (lvl == 2).astype(int),
                      "txt_service_core": (core & svc).astype(int), "txt_severe_core": (core & sev).astype(int),
                      "m": x["minutes"].to_numpy(),
                      "m_core": np.where(core, x["minutes"], np.nan),
                      "m_svc": np.where(svc, x["minutes"], np.nan)})
    g = f.groupby(["t", "s"], sort=False)
    agg = g[["txt_n_line", "txt_n_service_line", "txt_n_delay_line", "txt_n_mention", "txt_n_range",
             "txt_n_nbr", "txt_service_core", "txt_severe_core"]].sum()
    agg["txt_min_since"] = g["m"].min()
    agg["txt_min_since_core"] = g["m_core"].min()
    agg["txt_min_since_service"] = g["m_svc"].min()
    # топ оповещений: уровень ↓, служебные ↑, свежесть ↓
    xs = x.assign(svc=svc.astype(np.int8)).sort_values(["t", "s", "level", "svc", "minutes"],
                                                       ascending=[True, True, False, False, True])
    xs["rank"] = xs.groupby(["t", "s"], sort=False).cumcount()
    top = xs[xs["rank"] < TOP_ALERTS].pivot_table(index=["t", "s"], columns="rank", values="u", aggfunc="first")
    top.columns = [f"alert_{i}" for i in top.columns]
    lv = xs[xs["rank"] < TOP_ALERTS].pivot_table(index=["t", "s"], columns="rank", values="level", aggfunc="first")
    lv.columns = [f"alert_lvl_{i}" for i in lv.columns]
    out = agg.join(top).join(lv).reset_index()
    for i in range(TOP_ALERTS):
        for c, fill in [(f"alert_{i}", -1), (f"alert_lvl_{i}", 0)]:
            if c not in out:
                out[c] = fill
            out[c] = out[c].fillna(fill).astype(np.int32 if c.startswith("alert_") and "lvl" not in c else np.int8)
    return out


def stratum(df: pd.DataFrame) -> np.ndarray:
    """Слои E-1200: a — станция в служебном оповещении (упомянута / участок), b — по линиям только
    задержки или служебные без привязки к станции, c — активных оповещений нет."""
    n = df["txt_n_line"].fillna(0).to_numpy()
    core = df["txt_service_core"].fillna(0).to_numpy()
    return np.where(core > 0, "a", np.where(n > 0, "b", "c"))


# =============================================================================
# Числовая модальность
# =============================================================================
CAL_COLS = ["hour", "dow", "is_weekend", "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
            "is_school_day", "xmas_week", "is_city_event", "regime_student_omny", "cp_active"]
WX_COLS = ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m"]
STATIC_COLS = ["scale_weekday", "commute_index", "cluster", "n_lines", "in_cbd"]
N_LAGS = 24


def numeric_features(d: LabData, ref: np.ndarray, t: np.ndarray, s: np.ndarray, h: int = 1) -> pd.DataFrame:
    """Числа для объектов (t, s): только часы < T = index[t]. Цель — час t + h − 1."""
    Y = d.y.to_numpy(dtype="float32")
    C = d.closed.to_numpy()
    Yc = np.where(C, np.nan, Y)
    R = np.maximum(ref, 10.0)
    cols = {}
    for k in range(1, N_LAGS + 1):
        cols[f"r_lag{k}"] = Yc[t - k, s] / R[t - k, s]
    cols["r_lag168"] = Yc[t - 168, s] / R[t + h - 1, s]
    last3 = np.stack([Yc[t - k, s] for k in (1, 2, 3)])
    ref3 = np.stack([R[t - k, s] for k in (1, 2, 3)])
    cols["r_3h"] = np.where(np.isfinite(last3).any(0),
                            np.nansum(last3, 0) / np.sum(np.where(np.isfinite(last3), ref3, 0), 0).clip(1), np.nan)
    cols["log_ref_target"] = np.log1p(np.nan_to_num(ref[t + h - 1, s], nan=0.0))
    cols["log_ref_prev"] = np.log1p(np.nan_to_num(ref[t - 1, s], nan=0.0))
    # соседи по линии: среднее отношение «факт / обычный» за последний час (S: только < T)
    A = (d.adj_line > 0).astype("float32")
    ratio1 = Yc[t - 1] / R[t - 1]                                  # [N, S]
    m = np.isfinite(ratio1)
    num = (np.nan_to_num(ratio1) * A[s]).sum(1)
    den = (m * A[s]).sum(1)
    cols["nbr_r_lag1"] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    cov = d.cov
    tgt = t + h - 1
    for c in CAL_COLS:
        cols[c] = cov[c].to_numpy()[tgt].astype("float32")         # календарь известен заранее
    for c in WX_COLS:
        cols[f"wx_{c}"] = cov[c].to_numpy()[t - 1].astype("float32")  # S10: погода, наблюдённая в T−1
    cols["wx_precip_3h"] = sum(cov["precipitation"].to_numpy()[t - k] for k in (1, 2, 3)).astype("float32")
    st = d.static
    for c in STATIC_COLS:
        cols[f"st_{c}"] = st[c].to_numpy()[s].astype("float32")
    cols["st_borough"] = pd.Categorical(st["borough"]).codes[s].astype("float32")
    return pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in cols.items()})


NUM_FEATURES = ([f"r_lag{k}" for k in range(1, N_LAGS + 1)] + ["r_lag168", "r_3h", "log_ref_target",
                "log_ref_prev", "nbr_r_lag1"] + CAL_COLS + [f"wx_{c}" for c in WX_COLS] + ["wx_precip_3h"]
                + [f"st_{c}" for c in STATIC_COLS] + ["st_borough"])


# =============================================================================
# Изображение (S9)
# =============================================================================
IMG_W, IMG_H = 336, 224


def chart_series(Y: np.ndarray, t: int, s: int, hours: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Y — d.y.to_numpy(float32). Входы за последние 48 ч до T и те же 48 ч неделю назад. Только часы < T."""
    return Y[t - hours:t, s], Y[t - 168 - hours:t - 168, s]


def render_chart(cur: np.ndarray, prev: np.ndarray, w: int = IMG_W, h: int = IMG_H):
    """PNG-изображение (PIL): сплошная — последние 48 ч, пунктир — неделю назад, черта — момент T."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 8, 18, 10, 14
    top = float(np.nanmax(np.concatenate([cur, prev, [1.0]])))
    n = len(cur)

    def xy(arr):
        pts = []
        for i, v in enumerate(arr):
            if not np.isfinite(v):
                continue
            x = pad_l + (w - pad_l - pad_r) * i / (n - 1)
            yv = pad_t + (h - pad_t - pad_b) * (1 - v / top)
            pts.append((x, yv))
        return pts

    for k in range(0, n + 1, 24):   # полночь не отмечаем — только сетка суток от T
        x = pad_l + (w - pad_l - pad_r) * min(k, n - 1) / (n - 1)
        dr.line([(x, pad_t), (x, h - pad_b)], fill=(225, 225, 225), width=1)
    p = xy(prev)
    for i in range(0, len(p) - 1, 2):
        dr.line([p[i], p[i + 1]], fill=(150, 150, 150), width=2)
    dr.line(xy(cur), fill=(20, 70, 160), width=3)
    xT = w - pad_r + 6
    dr.line([(xT, pad_t), (xT, h - pad_b)], fill=(200, 30, 30), width=2)
    dr.text((pad_l, h - pad_b + 1), "-48h", fill=(90, 90, 90))
    dr.text((xT - 6, h - pad_b + 1), "T", fill=(200, 30, 30))
    return img


# =============================================================================
# Текстовая модальность для моделей и VLM
# =============================================================================
def format_alerts(row, a: pd.DataFrame, with_level: bool = True) -> str:
    """До TOP_ALERTS активных оповещений объекта в виде текста ('' — модальность отсутствует)."""
    out = []
    T = row["T"]
    for i in range(TOP_ALERTS):
        u = int(row[f"alert_{i}"])
        if u < 0:
            break
        r = a.loc[u]
        mins = int((T - r["date"]) / pd.Timedelta(minutes=1))
        lvl = {4: "this station", 3: "this station is in the affected section", 2: "nearby station",
               1: "same line"}.get(int(row[f"alert_lvl_{i}"]), "")
        head = f"[{mins} min ago; {r['status_label']}" + (f"; {lvl}]" if with_level else "]")
        out.append(f"{head} {r['header'].strip()}")
    return "\n".join(out)


# =============================================================================
# Загрузка результатов lab_00 / lab_01 и признаки для моделей
# =============================================================================
def find_file(name: str) -> Path:
    """Файл из вывода предыдущих ноутбуков: /kaggle/input/**/name или results/name."""
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return Path(hits[0])
    for cand in [Path("results"), Path("../results"), Path("/kaggle/working/results")]:
        for p in cand.glob(f"**/{name}"):
            return p
    raise FileNotFoundError(f"{name}: подключите вывод нужного ноутбука как Input")


def load_objects(part: str) -> pd.DataFrame:
    o = pd.read_parquet(find_file(f"objects_{part}.parquet"))
    o["day"] = o["T"].dt.normalize()
    return o


def load_alerts_clean() -> pd.DataFrame:
    return pd.read_parquet(find_file("alerts_clean.parquet"))


LEVEL_W = {4: 1.0, 3: 1.0, 2: 0.5, 1: 0.25, 0: 0.0}
TXT_LINE_ONLY = ["txt_n_line", "txt_n_service_line", "txt_n_delay_line", "txt_min_since"]


def object_text_emb(o: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    """Эмбеддинг текстовой модальности объекта: среднее эмбеддингов активных оповещений с весом
    релевантности (упомянута 1, участок 1, сосед 0,5, линия 0,25). Нет оповещений — нули."""
    out = np.zeros((len(o), E.shape[1]), dtype="float32")
    wsum = np.zeros(len(o), dtype="float32")
    for i in range(TOP_ALERTS):
        u = o[f"alert_{i}"].to_numpy()
        w = np.vectorize(LEVEL_W.get)(o[f"alert_lvl_{i}"].to_numpy()).astype("float32")
        m = u >= 0
        out[m] += w[m, None] * E[u[m]].astype("float32")
        wsum[m] += w[m]
    out[wsum > 0] /= wsum[wsum > 0, None]
    return out


def no_text(o: pd.DataFrame) -> pd.DataFrame:
    """Объекты с удалённой текстовой модальностью — так, как если бы активных оповещений не было (A9)."""
    o = o.copy()
    for c in TEXT_FEATURES:
        o[c] = np.nan if c.startswith("txt_min") else 0.0
    for i in range(TOP_ALERTS):
        o[f"alert_{i}"] = -1
        o[f"alert_lvl_{i}"] = 0
    return o


def shuffle_within(o: pd.DataFrame, cols: list[str], by: str = "T", seed: int = SEED) -> pd.DataFrame:
    """Перестановка модальности: значения cols перемешиваются между объектами одного часа T."""
    o = o.copy()
    rng = np.random.default_rng(seed)
    pos = np.arange(len(o))
    perm = pos.copy()
    for _, g in pd.Series(pos).groupby(o[by].to_numpy()):
        idx = g.to_numpy()
        perm[idx] = rng.permutation(idx)
    o[cols] = o[cols].to_numpy()[perm]
    return o


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


# =============================================================================
# Метрики (разд. 7)
# =============================================================================
def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Порог, максимизирующий macro-F1 (подбирается только на val)."""
    from sklearn.metrics import f1_score
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
    best, thr = -1, 0.5
    for q in qs:
        f = f1_score(y, p >= q, average="macro")
        if f > best:
            best, thr = f, float(q)
    return thr


def metrics(y: np.ndarray, p: np.ndarray, thr: float | None = None) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
                                 roc_auc_score)
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), 0, 1)
    out = {"n": len(y), "pos_rate": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p)) if y.any() else np.nan,
           "auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
           "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    if thr is not None:
        yh = p >= thr
        out.update({"thr": thr, "macro_f1": float(f1_score(y, yh, average="macro")),
                    "mcc": float(matthews_corrcoef(y, yh))})
    return out


def _edges(*ps, n_bins: int = 4000) -> np.ndarray:
    p = np.concatenate([np.asarray(x, dtype="float64") for x in ps])
    return np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))


def _day_hist(y, p, inv, n_days, edges):
    """Гистограммы позитивов и негативов по бинам вероятности для каждого дня: [дни, бины]."""
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 1)
    pos = np.zeros((n_days, len(edges))); neg = np.zeros((n_days, len(edges)))
    np.add.at(pos, (inv, b), y)
    np.add.at(neg, (inv, b), 1 - y)
    return pos, neg


def _curve_metric(pos, neg, metric):
    """PR-AUC / AUROC по гистограммам [..., бины] (бины по возрастанию вероятности)."""
    pos_r, neg_r = pos[..., ::-1], neg[..., ::-1]                 # по убыванию вероятности
    tp, fp = np.cumsum(pos_r, -1), np.cumsum(neg_r, -1)
    P, N = tp[..., -1], fp[..., -1]
    if metric == "pr_auc":
        prec = tp / np.maximum(tp + fp, 1e-12)
        return (pos_r / np.maximum(P[..., None], 1e-12) * prec).sum(-1)
    lower_neg = N[..., None] - fp                                 # негативы с меньшей вероятностью
    return ((pos_r * lower_neg).sum(-1) + 0.5 * (pos_r * neg_r).sum(-1)) / np.maximum(P * N, 1e-12)


class DayStats:
    """Быстрый блочный бутстрэп и перестановки по дням через гистограммы по бинам вероятности.

    PR-AUC и AUROC считаются по ≤ 4 000 квантильным бинам (отличие от точного значения — в 4-м знаке);
    Brier — точно (раскладывается по дням)."""

    def __init__(self, y, day, *ps):
        self.y = np.asarray(y, dtype="float64")
        self.days, self.inv = np.unique(np.asarray(day), return_inverse=True)
        self.D = len(self.days)
        self.ps = [np.clip(np.asarray(p, dtype="float64"), 0, 1) for p in ps]
        self.edges = _edges(*self.ps)
        self.h = [_day_hist(self.y, p, self.inv, self.D, self.edges) for p in self.ps]
        self.sq = []
        for p in self.ps:
            s = np.zeros(self.D); np.add.at(s, self.inv, (p - self.y) ** 2); self.sq.append(s)
        self.cnt = np.bincount(self.inv, minlength=self.D).astype("float64")

    def value(self, k: int, metric: str, W=None):
        """W — веса дней [повторы, дни] (бутстрэп) или None (вся выборка)."""
        W = np.ones((1, self.D)) if W is None else W
        if metric == "brier":
            return (W @ self.sq[k]) / (W @ self.cnt)
        pos, neg = self.h[k]
        return _curve_metric(W @ pos, W @ neg, metric)

    def boot_weights(self, n, seed=SEED):
        rng = np.random.default_rng(seed)
        W = np.zeros((n, self.D))
        np.add.at(W, (np.repeat(np.arange(n), self.D), rng.integers(0, self.D, n * self.D)), 1)
        return W

    def ci(self, k: int = 0, metric: str = "pr_auc", n_boot: int = 1000, seed: int = SEED):
        v = self.value(k, metric, self.boot_weights(n_boot, seed))
        return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))

    def diff(self, metric: str = "pr_auc", n_boot: int = 1000, n_perm: int = 2000, seed: int = SEED, a=0, b=1):
        """→ (бутстрэп разностей A − B, перестановочное распределение разности)."""
        W = self.boot_weights(n_boot, seed)
        bs = self.value(a, metric, W) - self.value(b, metric, W)
        rng = np.random.default_rng(seed + 1)
        S = (rng.random((n_perm, self.D)) < 0.5).astype("float64")    # 1 — в этот день A и B меняются
        if metric == "brier":
            sa = (1 - S) @ self.sq[a] + S @ self.sq[b]
            sb = S @ self.sq[a] + (1 - S) @ self.sq[b]
            perm = (sa - sb) / self.cnt.sum()
        else:
            (pa, na), (pb, nb) = self.h[a], self.h[b]
            posA = pa.sum(0) + S @ (pb - pa); negA = na.sum(0) + S @ (nb - na)
            posB = pb.sum(0) + S @ (pa - pb); negB = nb.sum(0) + S @ (na - nb)
            perm = _curve_metric(posA, negA, metric) - _curve_metric(posB, negB, metric)
        return bs, perm


def _metric_fn(name: str):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"pr_auc": average_precision_score, "auroc": roc_auc_score,
            "brier": lambda y, p: float(np.mean((p - y) ** 2))}[name]


def block_bootstrap(y, p, day, metric: str = "pr_auc", n_boot: int = 1000, seed: int = SEED) -> tuple:
    """95%-ный ДИ блочным бутстрэпом по дням (все объекты дня — один блок)."""
    return DayStats(y, day, p).ci(0, metric, n_boot, seed)


def paired_diff(y, p_a, p_b, day, metric: str = "pr_auc", n_boot: int = 1000, n_perm: int = 2000,
                seed: int = SEED) -> dict:
    """Разность метрики A − B: точная оценка, 95%-ный ДИ (парный блочный бутстрэп по дням) и
    односторонний p permutation test (внутри дня прогнозы A и B меняются местами с вероятностью ½).
    Для Brier «лучше» — меньше, поэтому p считается для A − B < 0."""
    fn = _metric_fn(metric)
    y = np.asarray(y)
    est = fn(y, np.asarray(p_a, dtype="float64")) - fn(y, np.asarray(p_b, dtype="float64"))
    bs, perm = DayStats(y, day, p_a, p_b).diff(metric, n_boot, n_perm, seed)
    sign = -1 if metric == "brier" else 1
    p_val = float((np.sum(sign * perm >= sign * est) + 1) / (len(perm) + 1))
    return {"diff": float(est), "ci_low": float(np.nanpercentile(bs, 2.5)),
            "ci_high": float(np.nanpercentile(bs, 97.5)), "p_perm": p_val}


# =============================================================================
# Журнал и ресурсы
# =============================================================================
class Timer:
    def __init__(self):
        self.times = []

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self.t0)

    def p(self, q: float) -> float:
        return float(np.percentile(self.times, q)) if self.times else np.nan


def gpu_peak_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception:
        pass
    return 0.0


def results_dir() -> Path:
    p = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else Path("results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_run(model: str, mode: str, part: str, objs: pd.DataFrame, p: np.ndarray, thr: float | None,
             extra: dict | None = None) -> dict:
    """Пишет прогнозы и строку журнала runs.csv. objs — объекты с колонками T, station, y_drop, day."""
    rd = results_dir()
    pred = objs[["T", "station", "y_drop"]].assign(p=np.asarray(p, dtype="float32"))
    pred.to_parquet(rd / f"{model}__{mode}__{part}.parquet", index=False)
    row = {"date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "model": model, "mode": mode, "part": part}
    row.update(metrics(objs["y_drop"].to_numpy(), p, thr))
    lo, hi = block_bootstrap(objs["y_drop"].to_numpy(), p, objs["T"].dt.normalize().to_numpy())
    row.update({"pr_auc_ci_low": lo, "pr_auc_ci_high": hi})
    if extra:
        row.update(extra)
    f = rd / "runs.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    return row
