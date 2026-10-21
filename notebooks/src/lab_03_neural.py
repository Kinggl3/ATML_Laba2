# %% [markdown]
# # ЛР2 · lab_03 — нейросетевое слияние: M4 gated fusion, M5 cross-attention (GPU)
#
# **Inputs:** датасет `vkr-transit-lab`, вывод `lab-00-data`, `lab-01-embed`, `lab-02-gbm` (для сравнения).
# **Accelerator:** GPU T4 ×1. Время: ~30–45 мин.
#
# - **M4 gated fusion:** свой кодировщик на модальность, обучаемые ворота, выпадение модальностей p = 0,2.
# - **M5 cross-attention:** токены часов ряда смотрят на токены оповещений и токен изображения.
# - Абляции: A1–A3 для M4 (обучение без модальности), A4 (M5 без cross-attention), A9 (модальность
#   пропадает на входе без переобучения), разброс по 3 seed.

# %%
import os, sys, time, glob, json, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
_kit = glob.glob("/kaggle/input/**/labkit.py", recursive=True) or ["./labkit.py"]
sys.path.insert(0, str(Path(_kit[0]).parent))
import labkit as lk
OUT = lk.results_dir()
dev = "cuda" if torch.cuda.is_available() else "cpu"
PARTS = ["train", "val", "test", "holdout"]
MAX_EPOCHS = int(os.environ.get("LAB_MAX_EPOCHS", 12))
T0 = time.time()

SMOKE = int(os.environ.get("LAB_SMOKE", 0))          # только для быстрой проверки кода: первые N объектов
O = {p: lk.load_objects(p) for p in PARTS}
A = lk.load_alerts_clean()
E_alert = np.load(lk.find_file("alert_emb.npy")).astype("float32")
IMG = {p: np.load(lk.find_file(f"img_emb_{p}.npy")) for p in PARTS}
if SMOKE:
    O = {p: O[p].head(SMOKE) for p in PARTS}; IMG = {p: IMG[p][:SMOKE] for p in PARTS}

# %% [markdown]
# ## Тензоры модальностей

# %%
num_mu = O["train"][lk.NUM_FEATURES].mean(); num_sd = O["train"][lk.NUM_FEATURES].std().replace(0, 1)
txt_mu = O["train"][lk.TEXT_FEATURES].mean(); txt_sd = O["train"][lk.TEXT_FEATURES].std().replace(0, 1)
a_date = A["date"].to_numpy(); a_svc = A["is_service"].to_numpy(); a_sev = A["is_severe"].to_numpy()
LAGS = [f"r_lag{k}" for k in range(1, lk.N_LAGS + 1)][::-1]          # от T−24 к T−1
OTHER = [c for c in lk.NUM_FEATURES if c not in LAGS]


def tensors(o, img):
    num = ((o[lk.NUM_FEATURES] - num_mu) / num_sd).fillna(0).to_numpy("float32")
    lags = o[LAGS].fillna(1.0).clip(0, 3).to_numpy("float32")
    other = ((o[OTHER] - num_mu[OTHER]) / num_sd[OTHER]).fillna(0).to_numpy("float32")
    tstruct = ((o[lk.TEXT_FEATURES] - txt_mu) / txt_sd).fillna(0).to_numpy("float32")
    ids = o[[f"alert_{i}" for i in range(lk.TOP_ALERTS)]].to_numpy()
    lvl = o[[f"alert_lvl_{i}" for i in range(lk.TOP_ALERTS)]].to_numpy()
    mask = ids >= 0
    safe = np.where(mask, ids, 0)
    minutes = np.where(mask, (o["T"].to_numpy()[:, None] - a_date[safe]) / np.timedelta64(1, "m"), 0) / 180.0
    meta = np.concatenate([np.eye(5, dtype="float32")[lvl], minutes[..., None],
                           a_svc[safe][..., None], a_sev[safe][..., None]], -1).astype("float32") * mask[..., None]
    return {"num": torch.tensor(num), "lags": torch.tensor(lags), "other": torch.tensor(other),
            "tstruct": torch.tensor(tstruct), "tid": torch.tensor(safe), "tmeta": torch.tensor(meta),
            "tmask": torch.tensor(mask), "img": torch.tensor(img.astype("float32")),
            "y": torch.tensor(o["y_drop"].to_numpy("float32")), "w": torch.tensor(o["weight"].to_numpy("float32"))}


TS = {p: tensors(O[p], IMG[p]) for p in PARTS}
E_t = torch.tensor(E_alert, device=dev)
print({p: len(TS[p]["y"]) for p in PARTS}, "мин:", round((time.time() - T0) / 60, 1))

# %% [markdown]
# ## Модели

# %%
D = 64


class TextTokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(E_alert.shape[1] + 8, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, b):
        return self.proj(torch.cat([E_t[b["tid"]], b["tmeta"]], -1))           # [B, 5, D]


class GatedFusion(nn.Module):
    """M4. use — какие модальности есть в модели (A1–A3); p_drop — выпадение модальностей при обучении."""

    def __init__(self, n_num, n_txt, n_img, use=("num", "txt", "img"), p_drop=0.2):
        super().__init__()
        self.use, self.p_drop = set(use), p_drop
        self.enc = nn.ModuleDict({
            "num": nn.Sequential(nn.Linear(n_num, 128), nn.GELU(), nn.Linear(128, D)),
            "txt": nn.Sequential(nn.Linear(D + n_txt, 128), nn.GELU(), nn.Linear(128, D)),
            "img": nn.Sequential(nn.Linear(n_img, 128), nn.GELU(), nn.Linear(128, D))})
        self.tok = TextTokens()
        self.gate = nn.Linear(3 * D + 3, 3)
        self.head = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, b, drop=None):
        B = len(b["y"])
        tm = b["tmask"].float()
        tok = (self.tok(b) * tm[..., None]).sum(1) / tm.sum(1, keepdim=True).clamp(min=1)
        h = {"num": self.enc["num"](b["num"]), "txt": self.enc["txt"](torch.cat([tok, b["tstruct"]], -1)),
             "img": self.enc["img"](b["img"])}
        present = {"num": torch.ones(B, device=dev), "txt": (tm.sum(1) > 0).float(), "img": torch.ones(B, device=dev)}
        for m in ("num", "txt", "img"):
            if m not in self.use or (drop and m in drop):
                present[m] = torch.zeros(B, device=dev)
            elif self.training and self.p_drop > 0:
                present[m] = present[m] * (torch.rand(B, device=dev) > self.p_drop).float()
        pres = torch.stack([present[m] for m in ("num", "txt", "img")], -1)
        hs = torch.stack([h[m] * present[m][:, None] for m in ("num", "txt", "img")], 1)
        g = torch.sigmoid(self.gate(torch.cat([hs.flatten(1), pres], -1))) * pres
        self.last_gates = g.detach()
        return self.head((g[..., None] * hs).sum(1)).squeeze(-1)


class CrossAttn(nn.Module):
    """M5. Токены ряда (24 часа + CLS из прочих чисел) → self-attention → cross-attention к оповещениям и
    изображению (+ «пустой» токен). attention=False — A4: вместо внимания склейка среднего контекста с CLS."""

    def __init__(self, n_other, n_txt, n_img, attention=True, layers=2):
        super().__init__()
        self.attention = attention
        self.lag = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(lk.N_LAGS, D) * 0.02)
        self.cls = nn.Linear(n_other, D)
        self.tok = TextTokens(); self.tstruct = nn.Linear(n_txt, D)
        self.img = nn.Linear(n_img, D); self.null = nn.Parameter(torch.zeros(1, 1, D))
        self.self_l = nn.ModuleList([nn.TransformerEncoderLayer(D, 4, 128, 0.1, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(D, 4, 0.1, batch_first=True) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(D if attention else 2 * D, 64), nn.GELU(), nn.Linear(64, 1))

    def context(self, b, drop):
        B = len(b["y"])
        t = self.tok(b) + self.tstruct(b["tstruct"])[:, None]
        tm = b["tmask"].clone()
        if drop and "txt" in drop:
            tm[:] = False
        im = self.img(b["img"])[:, None]
        imask = torch.ones(B, 1, dtype=torch.bool, device=dev) & ~torch.tensor(bool(drop and "img" in drop), device=dev)
        kv = torch.cat([self.null.expand(B, 1, D), t, im], 1)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), tm, imask], 1)
        return kv, keep

    def forward(self, b, drop=None):
        x = torch.cat([self.cls(b["other"])[:, None], self.lag(b["lags"][..., None]) + self.pos], 1)
        kv, keep = self.context(b, drop)
        for sl, ca, nm in zip(self.self_l, self.cross, self.norm):
            x = sl(x)
            if self.attention:
                x = x + ca(nm(x), kv, kv, key_padding_mask=~keep, need_weights=False)[0]
        if self.attention:
            return self.head(x[:, 0]).squeeze(-1)
        ctx = (kv * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)
        return self.head(torch.cat([x[:, 0], ctx], -1)).squeeze(-1)


def make(kind, **kw):
    n_num, n_txt, n_img = TS["train"]["num"].shape[1], TS["train"]["tstruct"].shape[1], TS["train"]["img"].shape[1]
    if kind == "gated":
        return GatedFusion(n_num, n_txt, n_img, **kw).to(dev)
    return CrossAttn(TS["train"]["other"].shape[1], n_txt, n_img, **kw).to(dev)

# %% [markdown]
# ## Обучение: AdamW, ранняя остановка по PR-AUC на val (терпение 3 эпохи)

# %%
def batches(ts, bs, shuffle, seed=0):
    n = len(ts["y"])
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = order[i:i + bs]
        yield {k: v[j].to(dev, non_blocking=True) for k, v in ts.items()}


@torch.no_grad()
def predict(model, ts, drop=None, bs=4096):
    model.eval()
    return torch.cat([torch.sigmoid(model(b, drop)).float().cpu() for b in batches(ts, bs, False)]).numpy()


def train(kind, seed=lk.SEED, epochs=None, parts=("train",), **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(kind, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = {k: torch.cat([TS[p][k] for p in parts]) for k in TS["train"]}
    best, best_state, best_ep, bad = -1, None, 0, 0
    n_ep = epochs or MAX_EPOCHS
    t0 = time.time()
    for ep in range(n_ep):
        model.train()
        for b in batches(tr, 1024, True, seed + ep):
            loss = (F.binary_cross_entropy_with_logits(model(b), b["y"], reduction="none") * b["w"]).sum() / b["w"].sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epochs:                       # фиксированное число эпох (переобучение на train + val)
            continue
        pr = lk.metrics(O["val"]["y_drop"], predict(model, TS["val"]))["pr_auc"]
        if pr > best:
            best, best_state, best_ep, bad = pr, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep or n_ep, "val_pr_auc": best, "fit_sec": round(time.time() - t0, 1)}


PRED, INFO, MODELS = {}, {}, {}
RUNS = {"M4_gated": ("gated", {}), "M5_crossattn": ("cross", {}),
        "A1_M4_no_text": ("gated", {"use": ("num", "img")}), "A2_M4_no_image": ("gated", {"use": ("num", "txt")}),
        "A3_M4_no_numbers": ("gated", {"use": ("txt", "img")}), "A4_M5_no_attention": ("cross", {"attention": False})}
for name, (kind, kw) in RUNS.items():
    model, info = train(kind, **kw)
    for p in ["val", "test"]:
        PRED[(name, p)] = predict(model, TS[p])
    full, _ = train(kind, epochs=info["best_epoch"], parts=("train", "val"), **kw)   # holdout
    PRED[(name, "holdout")] = predict(full, TS["holdout"])
    INFO[name] = info; MODELS[name] = model
    print(f"{name:<20} эпох {info['best_epoch']:>2}  PR-AUC val {info['val_pr_auc']:.4f}  {info['fit_sec']:.0f} c")

# %% [markdown]
# ## Метрики, прирост к U1 (лучший одномодальный из lab_02), устойчивость

# %%
U1 = {}
for p in ["val", "test", "holdout"]:
    u = pd.read_parquet(lk.find_file(f"U1_num__{p}.parquet")).head(len(O[p]))
    assert np.array_equal(u["station"].to_numpy(), O[p]["station"].to_numpy())
    U1[p] = u["p"].to_numpy()
THR = {m: lk.best_threshold(O["val"]["y_drop"].to_numpy(), PRED[(m, "val")]) for m in RUNS}
rows, gains = [], []
for m in RUNS:
    for p in ["val", "test", "holdout"]:
        o = O[p]
        r = {"model": m, "part": p, **lk.metrics(o["y_drop"], PRED[(m, p)], THR[m])}
        if p != "val":
            r["pr_auc_ci_low"], r["pr_auc_ci_high"] = lk.block_bootstrap(o["y_drop"], PRED[(m, p)], o["day"])
            for metric in ["pr_auc", "auroc", "brier"]:
                gains.append({"model": m, "vs": "U1_num", "part": p, "metric": metric,
                              **lk.paired_diff(o["y_drop"], PRED[(m, p)], U1[p], o["day"], metric=metric,
                                               n_perm=1000 if metric == "pr_auc" else 200)})
        rows.append(r)
        pd.DataFrame({"T": o["T"], "station": o["station"], "y_drop": o["y_drop"], "p": PRED[(m, p)].astype("float32")}
                     ).to_parquet(OUT / f"{m}__{p}.parquet", index=False)
res = pd.DataFrame(rows); res.to_csv(OUT / "metrics_neural.csv", index=False)
gain = pd.DataFrame(gains); gain.to_csv(OUT / "gain_neural.csv", index=False)
print(res[res["part"] == "test"][["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "auroc", "macro_f1", "mcc",
                                  "brier", "ece"]].round(4).to_string(index=False))
print(gain[gain["metric"] == "pr_auc"].round(4).to_string(index=False))

# %%
rows = []
o = O["test"]
for m in ["M4_gated", "M5_crossattn"]:
    full = lk.metrics(o["y_drop"], PRED[(m, "test")])["pr_auc"]
    for name, drop in [("без текста", {"txt"}), ("без изображения", {"img"}), ("без текста и изображения", {"txt", "img"})]:
        pr = lk.metrics(o["y_drop"], predict(MODELS[m], TS["test"], drop))["pr_auc"]
        rows.append({"model": m, "условие": name, "pr_auc": pr, "доля от полного": pr / full})
    for strat, g in o.groupby("stratum"):
        rows.append({"model": m, "условие": f"слой {strat}", "pr_auc": lk.metrics(g["y_drop"], PRED[(m, "test")][g.index])["pr_auc"],
                     "доля от полного": np.nan})
robust = pd.DataFrame(rows); robust.to_csv(OUT / "robustness_neural.csv", index=False)
print(robust.round(4).to_string(index=False))

# средние ворота M4 по слоям: насколько модель «слушает» каждую модальность
gates = []
MODELS["M4_gated"].eval()
with torch.no_grad():
    for b in batches(TS["test"], 4096, False):
        MODELS["M4_gated"](b); gates.append(MODELS["M4_gated"].last_gates.cpu().numpy())
gates = pd.DataFrame(np.concatenate(gates), columns=["ворота: числа", "ворота: текст", "ворота: изображение"])
gates["stratum"] = o["stratum"].to_numpy()
gates.groupby("stratum").mean().round(3).to_csv(OUT / "gates_M4.csv")
print(gates.groupby("stratum").mean().round(3))

# %% [markdown]
# ## Разброс по seed и ресурсы

# %%
rows = []
for m in ["M4_gated", "M5_crossattn"]:
    kind, kw = RUNS[m]
    vals = [lk.metrics(o["y_drop"], PRED[(m, "test")])["pr_auc"]]
    for seed in (1, 2):
        mdl, _ = train(kind, seed=seed, **kw)
        vals.append(lk.metrics(o["y_drop"], predict(mdl, TS["test"]))["pr_auc"])
    rows.append({"model": m, "pr_auc_seeds": [round(v, 4) for v in vals], "mean": np.mean(vals), "std": np.std(vals)})
seeds = pd.DataFrame(rows); seeds.to_csv(OUT / "seeds_neural.csv", index=False)
print(seeds.to_string(index=False))

res_rows = []
sub = {k: v[:300] for k, v in TS["test"].items()}
for m in ["M4_gated", "M5_crossattn"]:
    mdl = MODELS[m].eval(); tm = lk.Timer()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(300):
            b = {k: v[i:i + 1].to(dev) for k, v in sub.items()}
            with tm:
                torch.sigmoid(mdl(b)).item()
    t0 = time.time(); predict(mdl, TS["test"]); thr = len(O["test"]) / (time.time() - t0)
    res_rows.append({"model": m, "p50_ms": 1000 * tm.p(50), "p95_ms": 1000 * tm.p(95), "objects_per_sec": thr,
                     "fit_sec": INFO[m]["fit_sec"], "gpu_gb": lk.gpu_peak_gb(),
                     "params": sum(p.numel() for p in mdl.parameters())})
pd.DataFrame(res_rows).to_csv(OUT / "resources_neural.csv", index=False)
print(pd.DataFrame(res_rows).round(3).to_string(index=False))
json.dump({"info": INFO, "thresholds": THR}, open(OUT / "lab03_summary.json", "w"), indent=1, default=str)
print("готово, мин:", round((time.time() - T0) / 60, 1))
