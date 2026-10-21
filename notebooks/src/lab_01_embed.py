# %% [markdown]
# # ЛР2 · lab_01 — эмбеддинги текста и изображений (GPU)
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA` (с `alerts/`), вывод `lab-00-data`.
# **Accelerator:** GPU T4 ×1. Время: ~30–40 мин.
#
# 1. Текст: `BAAI/bge-small-en-v1.5`, эмбеддинг каждого очищенного оповещения (header + начало description).
# 2. Изображение: `google/siglip-base-patch16-224`, эмбеддинг графика «48 ч до T на фоне недели назад»
#    для каждого объекта train / val / test / holdout. Графики рисуются на лету и не сохраняются.
#
# Выход: `alert_emb.npy` [оповещения × 384], `img_emb_<part>.npy` [объекты × 768], float16,
# порядок строк = порядок в `alerts_clean.parquet` / `objects_<part>.parquet`.

# %%
import os, sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
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
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(dev, torch.__version__)

# %% [markdown]
# ## 1. Текст

# %%
from sentence_transformers import SentenceTransformer
a = lk.load_alerts_clean()
texts = a["text"].fillna("").tolist()
st = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
t0 = time.time()
E = st.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
np.save(OUT / "alert_emb.npy", E.astype("float16"))
t_text = time.time() - t0
print(E.shape, f"{t_text:.0f} c, {len(texts) / t_text:.0f} текстов/с")
del st; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Изображения

# %%
from transformers import AutoModel, AutoProcessor
d = lk.load()
Y = d.y.to_numpy("float32")
proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
try:
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224", dtype=torch.float16)
except TypeError:                                # старые версии transformers
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16)
model = model.to(dev).eval()


class Charts(torch.utils.data.Dataset):
    def __init__(self, t, s):
        self.t, self.s = t, s

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        cur, prev = lk.chart_series(Y, int(self.t[i]), int(self.s[i]))
        return lk.render_chart(cur, prev)


def collate(imgs):
    return proc(images=imgs, return_tensors="pt")["pixel_values"]


timing = {}
for part in lk.SPLIT:
    o = pd.read_parquet(lk.find_file(f"objects_{part}.parquet"), columns=["t", "s"])
    dl = torch.utils.data.DataLoader(Charts(o["t"].to_numpy(), o["s"].to_numpy()), batch_size=256,
                                     num_workers=4, collate_fn=collate)
    out = np.zeros((len(o), model.config.vision_config.hidden_size), dtype="float16")
    t0, k = time.time(), 0
    with torch.no_grad():
        for px in dl:
            f = model.get_image_features(pixel_values=px.to(dev, torch.float16))
            if not torch.is_tensor(f):          # transformers ≥ 5 возвращает объект, а не тензор
                f = f.pooler_output
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            out[k:k + len(f)] = f.cpu().numpy().astype("float16"); k += len(f)
    np.save(OUT / f"img_emb_{part}.npy", out)
    timing[part] = (len(o), time.time() - t0)
    print(part, out.shape, f"{timing[part][1] / 60:.1f} мин")

# %%
pd.DataFrame([{"what": "text_bge_small", "n": len(texts), "sec": t_text},
              *[{"what": f"img_siglip_{p}", "n": n, "sec": s} for p, (n, s) in timing.items()]]
             ).assign(per_item_ms=lambda x: 1000 * x["sec"] / x["n"]).to_csv(OUT / "embed_timing.csv", index=False)
# пример графика — для отчёта
o = pd.read_parquet(lk.find_file("objects_test.parquet"), columns=["t", "s", "stratum", "y_drop"])
for k, r in o[(o["stratum"] == "a") & (o["y_drop"] == 1)].head(3).reset_index().iterrows():
    lk.render_chart(*lk.chart_series(Y, int(r["t"]), int(r["s"]))).save(OUT / f"chart_example_{k}.png")
