"""通用性评测：同一套默认参数跑一批差异很大的场景，用真值量还原误差。

跑法：python3 tools/matrix_eval.py

合成样本的好处是叠加前的原画留在手里，可以直接算「修复结果 vs 真实原画」的误差，
不必依赖残影相关性那种间接指标。
"""
import io
import sys
import numpy as np
import cv2
import fitz
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unwatermark import clean_pdf, CleanOptions

W, H, PAGES = 900, 620, 10
BADGE = (W - 250, H - 60, W - 60, H - 34)      # 角标区域


def make_art(kind: str, index: int) -> np.ndarray:
    """逐页变化的「原画」。每种 kind 是一类完全不同的画面。"""
    rng = np.random.default_rng(1000 + index)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "flat_light":
        art = np.full((H, W, 3), 246, np.float32)
    elif kind == "flat_dark":
        art = np.full((H, W, 3), 38, np.float32)
    elif kind == "paper":                       # 浅底 + 纸纹
        art = np.full((H, W, 3), 238, np.float32)
        art += cv2.GaussianBlur(rng.normal(0, 14, (H, W)).astype(np.float32), (0, 0), 1.1)[:, :, None]
    elif kind == "watercolor":                  # 大色块 + 纸纹，接近绘本
        art = np.stack([
            150 + 70 * np.sin((x + index * 40) / 260.0),
            160 + 60 * np.cos((y + index * 25) / 180.0),
            110 + 50 * np.sin((x + y) / 320.0),
        ], -1).astype(np.float32)
        art += cv2.GaussianBlur(rng.normal(0, 11, (H, W)).astype(np.float32), (0, 0), 1.0)[:, :, None]
    elif kind == "night":                       # 深蓝夜景
        art = np.stack([
            30 + 26 * np.sin((x + index * 50) / 300.0),
            60 + 30 * np.cos((y + index * 30) / 200.0),
            95 + 28 * np.sin((x - y) / 280.0),
        ], -1).astype(np.float32)
        art += cv2.GaussianBlur(rng.normal(0, 8, (H, W)).astype(np.float32), (0, 0), 1.0)[:, :, None]
    elif kind == "gradient":                    # 角标正好压在明暗过渡上
        art = np.repeat(np.linspace(20, 250, W, dtype=np.float32)[None, :, None], H, 0).repeat(3, 2)
    elif kind == "busy":                        # 高频细节，最难
        art = np.full((H, W, 3), 150, np.float32)
        canvas = art.astype(np.uint8)
        for _ in range(160):
            cx, cy = rng.integers(0, W), rng.integers(0, H)
            cv2.circle(canvas, (int(cx), int(cy)), int(rng.integers(4, 26)),
                       tuple(int(v) for v in rng.integers(40, 235, 3)), -1)
        art = canvas.astype(np.float32) + rng.normal(0, 6, (H, W, 3))
    else:
        raise ValueError(kind)

    # 逐页变化的正文，保证「内容在变、角标不变」这个前提成立
    canvas = np.clip(art, 0, 255).astype(np.uint8)
    for row in range(9):
        yy = 60 + row * 52
        x1 = 70 + 180 + (index * 47 + row * 83) % 420
        cv2.rectangle(canvas, (70, yy), (x1, yy + 14), (62, 66, 78), -1)
    return canvas


def badge_alpha() -> np.ndarray:
    """角标的 alpha 图：文字 + 一个圆形 logo，带抗锯齿。"""
    big = np.zeros((H * 3, W * 3), np.uint8)
    cv2.putText(big, "NotebookLM", (BADGE[0] * 3 + 90, BADGE[3] * 3 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5, cv2.LINE_AA)
    cv2.circle(big, (BADGE[0] * 3 + 45, (BADGE[1] + BADGE[3]) * 3 // 2), 30, 255, 5, cv2.LINE_AA)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


ALPHA = badge_alpha()


def composite(art: np.ndarray, colour: str, strength: float) -> np.ndarray:
    """把角标按 alpha 叠上去——和 NotebookLM 的做法一致。"""
    if colour == "auto":
        patch = art[BADGE[1]:BADGE[3], BADGE[0]:BADGE[2]]
        c = 255.0 if patch.mean() < 128 else 0.0
    elif colour == "white":
        c = 255.0
    elif colour == "black":
        c = 0.0
    else:
        c = None
    a = (ALPHA * strength)[:, :, None]
    overlay = np.full_like(art, c, np.float32) if c is not None else np.full_like(art, 0, np.float32)
    if c is None:                                # 彩色角标
        overlay[:] = np.array([232, 120, 40], np.float32)
    return np.clip(art.astype(np.float32) * (1 - a) + overlay * a, 0, 255).astype(np.uint8)


def build(path: Path, kind: str, colour: str, strength: float):
    truth = []
    doc = fitz.open()
    for i in range(PAGES):
        art = make_art(kind, i)
        truth.append(art)
        marked = composite(art, colour, strength)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
        page = doc.new_page(width=W * 0.72, height=H * 0.72)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(path, deflate=True)
    doc.close()
    return truth


def evaluate(out_path: Path, truth, marked_path: Path):
    """在角标区域内，量「修复结果」与「真实原画」的差距，并和不处理做对比。"""
    doc = fitz.open(out_path)
    errs, base_errs = [], []
    for i in range(min(PAGES, doc.page_count)):
        imgs = doc[i].get_images(full=True)
        if not imgs:
            continue
        raw = doc.extract_image(imgs[0][0])["image"]
        got = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        if got.shape[:2] != (H, W):
            continue
        region = (slice(BADGE[1] - 6, BADGE[3] + 6), slice(BADGE[0] - 6, BADGE[2] + 6))
        t = truth[i].astype(np.float32)[region]
        errs.append(float(np.abs(got[region] - t).mean()))
    doc.close()
    src = fitz.open(marked_path)
    for i in range(min(PAGES, src.page_count)):
        raw = src.extract_image(src[i].get_images(full=True)[0][0])["image"]
        got = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        region = (slice(BADGE[1] - 6, BADGE[3] + 6), slice(BADGE[0] - 6, BADGE[2] + 6))
        base_errs.append(float(np.abs(got[region] - truth[i].astype(np.float32)[region]).mean()))
    src.close()
    return (float(np.mean(errs)) if errs else float("nan"),
            float(np.mean(base_errs)) if base_errs else float("nan"))


CASES = [
    ("纯浅底 / 黑字",     "flat_light", "black", 1.0),
    ("纯深底 / 白字",     "flat_dark",  "white", 1.0),
    ("纸纹浅底 / 黑字",   "paper",      "black", 1.0),
    ("水彩 / 自动极性",   "watercolor", "auto",  1.0),
    ("夜景深底 / 白字",   "night",      "white", 1.0),
    ("明暗渐变 / 自动",   "gradient",   "auto",  1.0),
    ("高频杂画 / 黑字",   "busy",       "black", 1.0),
    ("水彩 / 半透明50%",  "watercolor", "auto",  0.5),
    ("纸纹 / 半透明30%",  "paper",      "black", 0.3),
    ("水彩 / 橙色角标",   "watercolor", "colour", 1.0),
]

print(f"{'场景':<20}{'检出':<6}{'策略':<16}{'涂改%':>7}{'残留误差':>10}{'不处理':>9}{'改善'}")
print("-" * 84)
rows = []
for name, kind, colour, strength in CASES:
    src = Path(f"/tmp/mx_{kind}_{colour}_{int(strength*100)}.pdf")
    out = Path(str(src).replace(".pdf", "_out.pdf"))
    truth = build(src, kind, colour, strength)
    r = clean_pdf(src, out, CleanOptions())
    if not r.success:
        print(f"{name:<20}{'✗':<6}{'—':<16}{'':>7}{'':>10}{'':>9}  {r.message}")
        rows.append((name, False, 0, 0))
        continue
    err, base = evaluate(out, truth, src)
    print(f"{name:<20}{'✓':<6}{r.strategy:<16}{r.area_percent:>7.3f}{err:>10.2f}{base:>9.2f}"
          f"{(1 - err / base) * 100:>8.0f}%")
    rows.append((name, True, err, base))

good = [r for r in rows if r[1]]
print("-" * 84)
print(f"检出 {len(good)}/{len(rows)} 个场景"
      + (f"，平均残留误差 {np.mean([r[2] for r in good]):.2f}（不处理 {np.mean([r[3] for r in good]):.2f}）"
         if good else ""))
