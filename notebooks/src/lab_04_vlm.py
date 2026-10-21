# %% [markdown]
# # ЛР2 · lab_04 — визуально-языковые модели, zero-shot (GPU)
#
# **Inputs:** датасет `vkr-transit-lab`, вывод `lab-00-data`.
# **Accelerator:** GPU T4 ×1. Время: 3B — ~1 ч, 7B (4 бита) — ~2 ч, 2B — ~45 мин.
# **Internet:** включён (загрузка весов с Hugging Face).
#
# Один и тот же код для трёх моделей: меняется только `MODEL_KEY` в первой ячейке.
# Для каждой модели — три варианта входа (протокол, разд. 6): только текст оповещений, только график,
# текст + график (основной). Вероятность провала — из логитов ответа Yes / No на первом токене.
# Сначала отладка на V-60 (val), затем один прогон на E-1200 (test).

# %%
MODEL_KEY = "qwen25vl_3b"          # qwen25vl_3b | qwen25vl_7b | internvl3_2b

# %%
import os, sys, time, glob, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
_kit = glob.glob("/kaggle/input/**/labkit.py", recursive=True) or ["./labkit.py"]
sys.path.insert(0, str(Path(_kit[0]).parent))
import labkit as lk
OUT = lk.results_dir()
dev = "cuda"
MODELS = {
    "qwen25vl_3b": {"id": "Qwen/Qwen2.5-VL-3B-Instruct", "quant": None, "name": "M6_qwen25vl_3b"},
    "qwen25vl_7b": {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "quant": "nf4", "name": "M7_qwen25vl_7b"},
    "internvl3_2b": {"id": "OpenGVLab/InternVL3-2B-hf", "quant": None, "name": "M8_internvl3_2b"},
}
CFG = MODELS[MODEL_KEY]
if CFG["quant"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=False)
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
print(transformers.__version__, torch.__version__, CFG)

# %% [markdown]
# ## Объекты E-1200 и V-60, тексты оповещений, графики

# %%
d = lk.load()
Y = d.y.to_numpy("float32")
A = lk.load_alerts_clean()
st_name = d.static["station_complex"].to_numpy(); st_boro = d.static["borough"].to_numpy()
E = lk.load_objects("test"); E = E[E["in_e1200"]].reset_index(drop=True)
V = lk.load_objects("val"); V = V[V["in_v60"]].reset_index(drop=True)
print(len(E), len(V))

CHART_NOTE = ("The image shows hourly entries at this station over the last 48 hours (solid blue line) and the same "
              "48 hours one week earlier (dashed grey line, the usual pattern). The red line on the right marks now.")


def prompt_text(r, variant):
    s = int(r["s"]); T = r["T"]
    name = st_name[s]
    head = (f"You help a New York City subway dispatcher. Station: {name}, {st_boro[s]}. "
            f"Now: {T:%A %d %B %Y, %H:%M}.\n")
    parts = [head]
    if variant in ("chart", "text+chart"):
        parts.append(CHART_NOTE + "\n")
    if variant in ("text", "text+chart"):
        txt = lk.format_alerts(r, A)
        parts.append("Active MTA service alerts on this station's lines (most relevant first):\n" + txt + "\n"
                     if txt else "There are no active MTA service alerts on this station's lines.\n")
    parts.append(f"Question: in the next hour ({T:%H}:00–{(T + pd.Timedelta(hours=1)):%H}:00), will the number of "
                 "people entering this station be below 75% of its usual level for this hour of the week? "
                 "Answer with one word: Yes or No.")
    return "".join(parts)


def chart(r):
    return lk.render_chart(*lk.chart_series(Y, int(r["t"]), int(r["s"])))


json.dump({"text+chart": prompt_text(E.iloc[0], "text+chart"), "text": prompt_text(E.iloc[0], "text"),
           "chart": prompt_text(E.iloc[0], "chart")}, open(OUT / f"prompts_{MODEL_KEY}.json", "w"), indent=1)
chart(E.iloc[0]).save(OUT / f"prompt_chart_{MODEL_KEY}.png")
print(prompt_text(E.iloc[0], "text+chart"))

# %% [markdown]
# ## Модель и оценка вероятности по логитам Yes / No

# %%
kw = {"torch_dtype": torch.float16, "device_map": "cuda:0"}
if CFG["quant"] == "nf4":
    kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.float16), "device_map": "cuda:0"}
t0 = time.time()
proc = AutoProcessor.from_pretrained(CFG["id"])
model = AutoModelForImageTextToText.from_pretrained(CFG["id"], **kw).eval()
load_sec = time.time() - t0
tok = proc.tokenizer


def ids_for(words):
    out = set()
    for w in words:
        for v in (w, " " + w):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                out.add(t[0])
    return sorted(out)


YES, NO = ids_for(["Yes", "yes", "YES"]), ids_for(["No", "no", "NO"])
assert not set(YES) & set(NO)
print("токены Yes:", YES, "No:", NO, f"загрузка {load_sec:.0f} c")


@torch.no_grad()
def p_yes(r, variant):
    content = []
    if variant in ("chart", "text+chart"):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text(r, variant)})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    images = [chart(r)] if variant in ("chart", "text+chart") else None
    inputs = proc(text=[text], images=images, return_tensors="pt").to(dev)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    logits = model(**inputs).logits[0, -1].float()
    ly, ln = torch.logsumexp(logits[YES], 0), torch.logsumexp(logits[NO], 0)
    top = tok.decode(int(logits.argmax()))
    return float(torch.sigmoid(ly - ln)), top


VARIANTS = ["text+chart", "text", "chart"]


def run(objs, part):
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for v in VARIANTS:
        tm = lk.Timer()
        ps, tops = [], []
        for _, r in objs.iterrows():
            with tm:
                p, top = p_yes(r, v)
            ps.append(p); tops.append(top)
        ps = np.array(ps)
        if not np.isfinite(ps).all():
            print("NaN в вероятностях:", v, int((~np.isfinite(ps)).sum())); ps = np.nan_to_num(ps, nan=0.5)
        objs[f"p_{v}"] = ps
        top_yn = np.mean([t.strip().lower() in ("yes", "no") for t in tops])
        rows.append({"variant": v, "part": part, "p50_ms": 1000 * tm.p(50), "p95_ms": 1000 * tm.p(95),
                     "top_token_yes_no_share": top_yn, "gpu_gb": lk.gpu_peak_gb(), "n": len(objs),
                     "sec_total": float(np.sum(tm.times))})
        print(part, v, f"p95 {1000 * tm.p(95):.0f} мс; первый токен Yes/No в {top_yn:.0%} ответов; "
              f"средняя p {ps.mean():.3f}")
    return pd.DataFrame(rows)


res_v = run(V, "val")
if (V[[f"p_{v}" for v in VARIANTS]] == 0.5).mean().mean() > 0.2:
    # fp16 на T4 иногда даёт NaN в визуальной части — переходим на 4 бита с вычислениями в float32
    print("много NaN в fp16 — перезагрузка: nf4, вычисления в float32 (записать в отступления)")
    del model; torch.cuda.empty_cache()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], check=False)
    kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.float32), "device_map": "cuda:0"}
    model = AutoModelForImageTextToText.from_pretrained(CFG["id"], **kw).eval()
    CFG["quant"] = "nf4-fp32-fallback"
    res_v = run(V, "val")
assert res_v["top_token_yes_no_share"].min() > 0.8, "модель отвечает не Yes/No — проверить промпт"
res_e = run(E, "test")

# %% [markdown]
# ## Метрики на E-1200 (порог macro-F1 — по V-60)

# %%
name = CFG["name"]
rows = []
for v in VARIANTS:
    thr = lk.best_threshold(V["y_drop"].to_numpy(), V[f"p_{v}"].to_numpy())
    m = lk.metrics(E["y_drop"], E[f"p_{v}"], thr)
    lo, hi = lk.block_bootstrap(E["y_drop"], E[f"p_{v}"], E["day"])
    r_e = res_e[res_e["variant"] == v].iloc[0]
    rows.append({"model": name, "weights": CFG["id"], "variant": v, **m, "pr_auc_ci_low": lo, "pr_auc_ci_high": hi,
                 "p50_ms": r_e["p50_ms"], "p95_ms": r_e["p95_ms"], "gpu_gb": r_e["gpu_gb"],
                 "cost_usd_per_1000": r_e["sec_total"] / len(E) * 1000 / 3600 * 0.35,
                 "top_token_yes_no_share": r_e["top_token_yes_no_share"]})
    for strat, g in E.groupby("stratum"):
        rows.append({"model": name, "variant": f"{v} | слой {strat}",
                     **lk.metrics(g["y_drop"], g[f"p_{v}"])})
res = pd.DataFrame(rows)
res.to_csv(OUT / f"metrics_{name}.csv", index=False)
print(res[["variant", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "auroc", "macro_f1", "brier", "p95_ms"]]
      .round(4).to_string(index=False))
keep = ["T", "station", "stratum", "y_drop"] + [f"p_{v}" for v in VARIANTS]
E[keep].to_parquet(OUT / f"{name}__e1200.parquet", index=False)
V[keep].to_parquet(OUT / f"{name}__v60.parquet", index=False)
pd.concat([res_v, res_e]).to_csv(OUT / f"timing_{name}.csv", index=False)
json.dump({"model": name, "weights": CFG["id"], "quant": CFG["quant"], "transformers": transformers.__version__,
           "load_sec": load_sec, "gpu": torch.cuda.get_device_name(0)}, open(OUT / f"run_{name}.json", "w"), indent=1)
