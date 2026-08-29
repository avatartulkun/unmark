from __future__ import annotations

"""去掉整页位图型 PDF 的固定位置水印（NotebookLM 导出等）。全程离线。

核心思路：水印的定义就是「内容在变、它不变」的那部分像素。
所以不靠内置模板图，而是把若干页的角落区域叠起来求交集 ——
在几乎每一页的同一坐标都成立的，才算水印。这样换个字体、
换个版本、甚至换成别的厂商的角标，都还能认出来。

只改判定出来的那一小块（默认上限是整页 0.5%），页面其余像素逐位不变：
每页修复后都在内存里强制还原 Mask 外像素并断言，写出后再抽页复核。

与 GUI 的配合：整个流程是流式的 —— 逐页解码、修复、写入新文档，解码出的位图
只在预算内缓存（见 CleanOptions.cache_budget_mb），所以内存占用有上限而不随页数
线性增长。调用方可以传 progress 显示进度、传 should_cancel 中途取消。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import fitz
import numpy as np

ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


@dataclass(frozen=True)
class CleanOptions:
    """检测与修复参数；默认值针对 NotebookLM 右下角角标调校。"""

    corner_w: float = 0.30
    """检测区域宽度占页宽比例（0.30 = 只看右侧 30%）。"""
    corner_h: float = 0.12
    """检测区域高度占页高比例（0.12 = 只看底部 12%）。"""
    dark: int = 120
    """暗像素阈值 0-255；灰度低于它算「暗」。"""
    contrast_delta: int = 18
    """回退策略用：像素与其局部背景的灰度差达到多少算「浮在背景之上」。"""
    chroma_delta: int = 12
    """第三路策略用：像素与其局部背景在 Lab 的 a/b 平面上的距离。

    前两路都只看亮度，所以「颜色不同但一样亮」的角标（例如浅灰底上的橙色字标）
    会同时从两张网里漏掉。a/b 与亮度无关，专门补这个洞。
    """
    vote: float = 0.85
    """某像素需在多大比例的取样页上都成立才算水印。"""
    dilate: int = 2
    """Mask 外扩像素，吃掉抗锯齿边。"""
    ink_kernel: int = 7
    """精修形状用的中值滤波核（奇数）。要比笔画粗、比字高细。

    定位靠高斯反差，但高斯半径大到足以站稳时会把整行字糊成一团，涂出来是个方块。
    中值滤波核只要比笔画粗，就能把笔画整根抹掉而背景原样留下，相减剩下的才是墨迹。
    """
    ink_min_ratio: float = 0.12
    """精修后至少要保留粗定位的多大比例，否则认为精修失效、退回粗掩膜。"""
    residue_sweeps: int = 2
    """修完之后再自查几轮：边缘还留着墨迹就并进掩膜重修。

    抗锯齿那圈有多宽，取决于角标颜色和底色的反差——深底上的白字比浅底上的深字
    扩散得宽得多，靠一个固定的外扩像素数盖不住。与其猜半径，不如修完看结果。
    只允许长进紧贴掩膜的一圈，不会啃到画面。
    """
    residue_ratio: float = 0.55
    """自查时的墨迹判据，取 contrast_delta 的这个比例。"""
    ink_dilate: int = 1
    """精修成功后，对字形掩膜再外扩多少像素，用来盖住抗锯齿边。

    抗锯齿那圈在不同页的底色上时隐时现，达不到跨页投票的门槛，只能靠几何补。
    起点是字形而不是方块，所以扩 1 像素不会把字母之间糊成一片；扩 0 会留下
    能读出字来的残影。
    """
    ink_weak_ratio: float = 0.45
    """滞后阈值的低档，取 contrast_delta 的这个比例。

    抗锯齿边比笔画淡，够不到高阈值。用低阈值向外生长、只保留与核心连通的部分，
    形状就沿着字母轮廓走；换成形态学外扩则是方块外扩，几像素就把字母之间粘成一片。
    """
    radius: int = 4
    """inpaint 修复半径。"""
    backdrop_ring: int = 6
    """判定底色是否纯净时，往 Mask 外取样多少像素宽的一圈；0 表示禁用纯色铺底。"""
    backdrop_tolerance: float = 4.0
    """取样圈内像素偏离中位数的平均值上限；不超过就认为是纯色底，直接铺底色。"""
    graft_texture: bool = True
    """走 inpaint 那条路时，是否把邻近的高频纹理移植到修复区。

    补的是按周围颗粒强度合成的噪声。注意不能从旁边搬真实纹理——试过，会把画面结构
    （叶子边缘等）一起复制进来，肉眼看到弧线和划痕，而高频能量这个指标反而变好看。
    """
    max_area: float = 0.005
    """涂改面积占整页的上限，超了就判定为误检并中止。"""
    coverage: float = 0.98
    """图片覆盖页面多少比例才算「满版位图页」。"""
    vote_sample: int = 48
    """参与投票的最大页数（均匀取样）；0 表示全部页。"""
    verify_sample: int = 8
    """写出后抽多少页做逐像素复核；0 表示全部处理过的页。"""
    png_compression: int = 3
    """中间 PNG 的压缩级别。无论取值都是无损，调低只是更快、文件更大。"""
    cache_budget_mb: int = 256
    """检测阶段解码出的位图最多缓存多少 MB 给修复阶段复用；超出的页在修复时重新解码。"""
    box: Optional[tuple[int, int, int, int]] = None
    """手工指定水印框 (x0, y0, x1, y1)，图像像素坐标；给了就跳过自动检测。"""


@dataclass(frozen=True)
class PageRef:
    """一个满版位图页的元信息；不持有像素，保证内存占用与页数无关。"""

    index: int
    xref: int
    width: int
    height: int


@dataclass(frozen=True)
class SkippedPage:
    index: int
    reason: str


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray
    """与页面同尺寸的 uint8 掩膜（0/255）。"""
    box: tuple[int, int, int, int]
    core_pixels: int
    pages_voted: int
    strategy: str


@dataclass(frozen=True)
class AutoCleanResult:
    """一次自动去水印的完整结论；失败时 message 说明原因。"""

    success: bool
    message: str
    pages_processed: int = 0
    box: Optional[tuple[int, int, int, int]] = None
    box_ratio: Optional[tuple[float, float, float, float]] = None
    area_percent: float = 0.0
    strategy: str = ""
    cancelled: bool = False
    output_path: Optional[Path] = None
    skipped: list[SkippedPage] = field(default_factory=list)

    def as_dict(self) -> dict:
        """兼容既有 GUI 线程的 dict 取值方式。"""
        return {
            "success": self.success,
            "message": self.message,
            "pages_processed": self.pages_processed,
            "box": self.box,
            "box_ratio": self.box_ratio,
            "area_percent": self.area_percent,
            "strategy": self.strategy,
            "cancelled": self.cancelled,
            "output": str(self.output_path) if self.output_path else None,
            "skipped": [(item.index, item.reason) for item in self.skipped],
        }


class CleanCancelled(RuntimeError):
    """调用方通过 should_cancel 主动中断。"""


# ---------------------------------------------------------------- 页面筛选

def scan_full_bleed_pages(
    document: fitz.Document, coverage: float = 0.98
) -> tuple[list[PageRef], list[SkippedPage]]:
    """挑出「整页就是一张图」的页面，只读元信息、不解码像素。

    有文字、有矢量图形、有多张图，或者图没铺满页面的，一律跳过 ——
    那种页面的水印可能是可删除的对象，不该用涂像素的办法处理。
    """
    refs: list[PageRef] = []
    skipped: list[SkippedPage] = []
    for index, page in enumerate(document):
        images = page.get_images(full=True)
        reason = ""
        if len(images) != 1:
            reason = f"本页有 {len(images)} 张图片，不是单张满版位图"
        elif page.get_text().strip():
            reason = "本页有文字层，水印可能是可直接删除的文本对象"
        elif page.get_drawings():
            reason = "本页有矢量图形"
        else:
            infos = page.get_image_info(xrefs=True)
            info = next((item for item in infos if item.get("xref") == images[0][0]), None)
            if info is None:
                reason = "取不到图片位置信息"
            else:
                x0, y0, x1, y1 = info["bbox"]
                page_area = page.rect.width * page.rect.height
                covered = abs((x1 - x0) * (y1 - y0))
                if page_area <= 0 or covered < coverage * page_area:
                    percent = 100.0 * covered / page_area if page_area > 0 else 0.0
                    reason = f"图片只覆盖页面 {percent:.0f}%，不是满版"
        if reason:
            skipped.append(SkippedPage(index, reason))
            continue
        xref, _, width, height = images[0][0], images[0][1], int(images[0][2]), int(images[0][3])
        if width <= 0 or height <= 0:
            skipped.append(SkippedPage(index, "图片尺寸信息异常"))
            continue
        refs.append(PageRef(index, xref, width, height))
    return refs, skipped


def keep_dominant_size(
    refs: Sequence[PageRef],
) -> tuple[list[PageRef], list[SkippedPage]]:
    """跨页比对要求像素尺寸一致；保留占多数的那个尺寸，其余页原样保留不处理。"""
    if not refs:
        return [], []
    counts: dict[tuple[int, int], int] = {}
    for ref in refs:
        counts[(ref.width, ref.height)] = counts.get((ref.width, ref.height), 0) + 1
    dominant = max(counts, key=lambda size: counts[size])
    kept = [ref for ref in refs if (ref.width, ref.height) == dominant]
    dropped = [
        SkippedPage(ref.index, f"尺寸 {ref.width}×{ref.height} 与主流尺寸 {dominant[0]}×{dominant[1]} 不一致")
        for ref in refs
        if (ref.width, ref.height) != dominant
    ]
    return kept, dropped


def _decode_page_image(document: fitz.Document, ref: PageRef) -> Optional[np.ndarray]:
    """把某页的嵌入位图解码成 RGB 数组；失败返回 None。"""
    raw = document.extract_image(ref.xref)
    array = cv2.imdecode(np.frombuffer(raw["image"], np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return None
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


class _PageCache:
    """检测阶段解码的位图，在预算内留给修复阶段复用。

    绘本通常几十页，全部装得下，于是每页只解码一次；遇到超大文件则自动退化成
    「用完就扔、需要时重新解码」，用一点时间换一个不会被页数撑爆的内存上限。
    """

    def __init__(self, budget_mb: int) -> None:
        self._budget = max(0, budget_mb) * 1024 * 1024
        self._used = 0
        self._images: dict[int, np.ndarray] = {}

    def put(self, index: int, image: np.ndarray) -> None:
        if self._used + image.nbytes > self._budget:
            return
        self._images[index] = image
        self._used += image.nbytes

    def pop(self, index: int) -> Optional[np.ndarray]:
        image = self._images.pop(index, None)
        if image is not None:
            self._used -= image.nbytes
        return image

    def clear(self) -> None:
        self._images.clear()
        self._used = 0


def _even_sample(refs: Sequence[PageRef], limit: int) -> list[PageRef]:
    if limit <= 0 or len(refs) <= limit:
        return list(refs)
    positions = np.linspace(0, len(refs) - 1, limit).round().astype(int)
    return [refs[int(position)] for position in dict.fromkeys(positions.tolist())]


def _check_cancel(should_cancel: CancelCallback) -> None:
    if should_cancel is not None and should_cancel():
        raise CleanCancelled()


# ---------------------------------------------------------------- 水印检测

def detect_watermark(
    document: fitz.Document,
    refs: Sequence[PageRef],
    options: CleanOptions,
    progress: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    cache: Optional["_PageCache"] = None,
) -> tuple[Optional[Detection], str]:
    """跨页求交集定位水印。返回 (Detection, "") 或 (None, 原因)。"""
    if len(refs) < 3:
        return None, f"只有 {len(refs)} 页满版位图，跨页比对不可靠；请手工框选水印区域"

    height, width = refs[0].height, refs[0].width
    # 搜索区域：默认只看右下角，避免把正文里反复出现的深色元素当水印。
    x_start = int(width * (1.0 - options.corner_w))
    y_start = int(height * (1.0 - options.corner_h))
    if x_start >= width or y_start >= height:
        return None, "检测区域比例设置有误，右下角范围为空"

    sample = _even_sample(refs, options.vote_sample)
    dark_votes = np.zeros((height - y_start, width - x_start), np.float32)
    contrast_votes = np.zeros_like(dark_votes)
    chroma_votes = np.zeros_like(dark_votes)
    fine_votes = np.zeros_like(dark_votes)
    weak_votes = np.zeros_like(dark_votes)
    sigma = max(2.0, min(8.0, min(dark_votes.shape) / 12.0))
    used = 0
    for position, ref in enumerate(sample, start=1):
        _check_cancel(should_cancel)
        if progress:
            progress(position, len(sample), f"分析第 {ref.index + 1} 页")
        rgb = _decode_page_image(document, ref)
        if rgb is None or rgb.shape[:2] != (height, width):
            continue
        if cache is not None:
            cache.put(ref.index, rgb)
        corner = rgb[y_start:, x_start:]
        gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)
        dark_votes += (gray < options.dark).astype(np.float32)
        # 第二路投票：与局部背景的反差。深色角标、浅色角标都能站住，
        # 而恒定的白边距因为「和背景一样」不会入选，不至于把整条页边当水印。
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        contrast_votes += (cv2.absdiff(gray, background) >= options.contrast_delta).astype(np.float32)
        # 第三路投票：色度反差。前两路都只看亮度，遇到「颜色不同但一样亮」的角标
        # （浅灰底上的橙色字标就是这样）会同时漏掉。a/b 通道与亮度无关，专治这种。
        lab = cv2.cvtColor(corner, cv2.COLOR_RGB2LAB).astype(np.float32)
        ab = lab[:, :, 1:]
        ab_background = cv2.GaussianBlur(ab, (0, 0), sigmaX=sigma)
        chroma_distance = np.linalg.norm(ab - ab_background, axis=2)
        chroma_votes += (chroma_distance >= options.chroma_delta).astype(np.float32)
        # 精细投票：只为了定「形状」，不负责定位。
        # 上面三路用的高斯半径要足够大才站得稳，代价是把整行字糊成一团——字母间隙
        # 也跟着超标，最后涂掉的是一个方块而不是笔画。中值滤波正好相反：核比笔画粗
        # 就能把笔画整根抹掉、背景原样留下，两者相减剩下的才是真正的墨迹。
        ink = cv2.absdiff(gray, cv2.medianBlur(gray, options.ink_kernel))
        fine_votes += (ink >= options.contrast_delta).astype(np.float32)
        weak_votes += (ink >= options.contrast_delta * options.ink_weak_ratio).astype(np.float32)
        used += 1
    if used < 3:
        return None, f"只有 {used} 页位图可解码，跨页比对不可靠"

    dark_votes /= used
    contrast_votes /= used
    chroma_votes /= used
    fine_votes /= used
    weak_votes /= used
    reasons: list[str] = []
    # 顺序即优先级：先试最保守的暗像素，再试亮度反差，最后才用色度补漏。
    for strategy, votes in (
        ("暗像素跨页交集", dark_votes),
        ("局部反差跨页交集", contrast_votes),
        ("色度跨页交集", chroma_votes),
    ):
        core = (votes >= options.vote).astype(np.uint8)
        shaped, reason = _shape_core(core, height, width, options, fine_votes, weak_votes)
        if shaped is None:
            reasons.append(f"{strategy}：{reason}")
            continue
        core_full, box, core_pixels, refined_ok = shaped
        # 精修成功时不再做形态学外扩：滞后阈值已经把抗锯齿边沿着字形收进来了，
        # 再方块外扩只会把字母之间重新粘成一片。
        grow = options.ink_dilate if refined_ok else options.dilate
        if grow:
            mask = cv2.dilate(core_full * 255, np.ones((grow * 2 + 1,) * 2, np.uint8))
        else:
            mask = core_full * 255
        return Detection(mask, box, core_pixels, used, strategy), ""
    return None, "；".join(reasons)


def _refine_to_ink(
    core_full: np.ndarray,
    core_pixels: int,
    fine_votes: np.ndarray,
    weak_votes: np.ndarray,
    options: CleanOptions,
) -> tuple[np.ndarray, int, bool]:
    """把粗定位削成笔画形状，返回 (掩膜, 像素数, 是否精修成功)。

    滞后阈值：高阈值挑出确定是墨迹的核心，低阈值向外生长，只保留与核心连通的块。
    抗锯齿边因此沿着字形被收进来，而不是像形态学外扩那样把字母之间糊成一片。

    削不动就原样返回粗掩膜——宁可多涂一点，也不能因为精修失灵而漏掉半个角标。
    """
    height, width = core_full.shape

    def lift(region: np.ndarray) -> np.ndarray:
        full = np.zeros_like(core_full)
        full[height - region.shape[0]:, width - region.shape[1]:] = region
        return full

    strong = lift((fine_votes >= options.vote).astype(np.uint8)) & core_full
    if strong.sum() == 0:
        return core_full, core_pixels, False

    weak = lift((weak_votes >= options.vote).astype(np.uint8)) & core_full
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    touched = np.unique(labels[strong > 0])
    keep = np.zeros(count, bool)
    keep[touched[touched > 0]] = True
    refined = (keep[labels] | (strong > 0)).astype(np.uint8)

    kept = int(refined.sum())
    if kept < max(8, int(core_pixels * options.ink_min_ratio)):
        return core_full, core_pixels, False
    return refined, kept, True


def _shape_core(
    core: np.ndarray,
    height: int,
    width: int,
    options: CleanOptions,
    fine_votes: Optional[np.ndarray] = None,
    weak_votes: Optional[np.ndarray] = None,
) -> tuple[Optional[tuple[np.ndarray, tuple[int, int, int, int], int, bool]], str]:
    """把投票结果收敛成一个像角标的连通块，并做几何与面积校验。

    core 是右下角搜索区域内的 0/1 图；返回的掩膜是整页尺寸。
    """
    if core.sum() == 0:
        return None, "区域内没有跨页一致的像素"

    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if count <= 1:
        return None, "区域内没有跨页一致的像素"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, top, _, block_height, _ = stats[biggest]

    # 合并同一行上的其它块：logo 和文字通常是彼此分开的连通块。
    band_low, band_high = top - block_height, top + 2 * block_height
    keep = np.zeros(count, bool)
    for label in range(1, count):
        _, label_top, _, label_height, label_area = stats[label]
        if label_area >= 8 and band_low <= label_top and label_top + label_height <= band_high:
            keep[label] = True
    merged = keep[labels].astype(np.uint8)
    if merged.sum() == 0:
        return None, "区域内没有成行的候选块"

    # 先削成字形再校验面积：粗定位的团块把字母间隙也算了进去，虚胖一倍有余，
    # 拿它去比 0.5% 的上限会把合法角标误判成正文——纯深底上尤其容易踩到。
    y_offset, x_offset = height - core.shape[0], width - core.shape[1]
    refined_ok = False
    if fine_votes is not None and weak_votes is not None:
        full = np.zeros((height, width), np.uint8)
        full[y_offset:, x_offset:] = merged
        full, _, refined_ok = _refine_to_ink(full, int(merged.sum()), fine_votes, weak_votes, options)
        if refined_ok:
            merged = full[y_offset:, x_offset:]
    if merged.sum() == 0:
        return None, "精修后没有剩下像素"

    rows, columns = np.nonzero(merged)
    box = (
        int(columns.min()) + x_offset,
        int(rows.min()) + y_offset,
        int(columns.max()) + 1 + x_offset,
        int(rows.max()) + 1 + y_offset,
    )
    core_pixels = int(merged.sum())

    ratio = core_pixels / float(height * width)
    if ratio > options.max_area:
        return None, (
            f"命中区域占整页 {ratio * 100:.3f}%，超过上限 {options.max_area * 100:.3f}%，很可能是正文"
        )
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    if box_width < 12 or box_height < 4:
        return None, f"包围盒 {box_width}×{box_height} 太小，不像角标"
    aspect = box_width / float(box_height)
    if not 1.2 <= aspect <= 40:
        return None, f"包围盒长宽比 {aspect:.1f} 不像一行角标文字"

    full = np.zeros((height, width), np.uint8)
    full[y_offset:, x_offset:] = merged
    return (full, box, core_pixels, refined_ok), ""


def box_to_mask(
    shape: tuple[int, int], box: tuple[int, int, int, int], dilate: int
) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = box
    mask[max(0, y0):y1, max(0, x0):x1] = 255
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate * 2 + 1,) * 2, np.uint8))
    return mask


def parse_box(value) -> Optional[tuple[int, int, int, int]]:
    """接受 "x0,y0,x1,y1" 字符串或四元组；None 表示走自动检测。"""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 4:
        raise ValueError("水印框需要四个数：x0,y0,x1,y1")
    x0, y0, x1, y1 = (int(item) for item in parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("水印框的右下角必须大于左上角")
    return x0, y0, x1, y1


# ---------------------------------------------------------------- 修复与写出

def _flat_backdrop(
    rgb: np.ndarray, mask: np.ndarray, ring: int, tolerance: float
) -> Optional[np.ndarray]:
    """角标压在纯色块上时，返回那个底色；底色不统一则返回 None。

    这一步是为了绕开 inpaint。TELEA 是从边界向内推测纹理的算法，纯色区域里它会拉出
    规则的竖条纹——幅度不大（几个灰阶），但因为有结构，人眼一眼就看出「这块被涂过」。
    纯色底压根不需要推测：把底色铺回去就是像素级完美。
    """
    if ring <= 0:
        return None
    kernel = np.ones((ring * 2 + 1,) * 2, np.uint8)
    outer = cv2.dilate(mask, kernel)
    band = (outer > 0) & (mask == 0)
    samples = rgb[band]
    if samples.size == 0:
        return None
    # 用中位数而不是均值：万一取样圈蹭到了正文的笔画，中位数不会被带偏
    center = np.median(samples, axis=0)
    spread = np.abs(samples.astype(np.int16) - center).mean(axis=0)
    if spread.max() > tolerance:
        return None
    return center.round().astype(np.uint8)


def _graft_texture(
    repaired: np.ndarray, rgb: np.ndarray, mask: np.ndarray, sigma: float = 1.6
) -> np.ndarray:
    """给修复区补回颗粒感，别让补丁光滑得可疑。

    inpaint 靠扩散填色：低频颜色接得住，高频全被抹平。于是在水彩、纸纹这类有颗粒的
    画面上，修复区是一块异常光滑的矩形——放大看一眼就锁定。

    补的是**合成噪声**，不是从旁边搬来的真实纹理。搬真实纹理试过，会把邻近的画面结构
    （叶子边缘之类）一起复制进来，视觉上更糟；量化指标反而变好看，所以这里只信眼睛。
    噪声的强度按周围一圈的高频能量定，空间尺度用一次轻微模糊对上纸纹的粗细。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return repaired
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    # 参照区：Mask 外扩一圈里的真实像素，用来量「这块画面本身有多少颗粒」
    band = (cv2.dilate(mask, np.ones((13, 13), np.uint8)) > 0) & (mask == 0)
    if not band.any():
        return repaired
    detail = rgb.astype(np.float32) - cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma).astype(np.float32)
    # 用中位绝对偏差而不是标准差：取样圈常常蹭到装饰线、星光这类高对比元素，
    # std 会被少数极端值拉高好几倍（实测 8 倍），照它注入噪声就是一片黑白麻点。
    # 1.4826 是把 MAD 换算成正态分布标准差的常数。
    amplitude = 1.4826 * float(np.median(np.abs(detail[band])))
    amplitude = min(amplitude, 8.0)          # 再兜一道：颗粒感不该盖过画面本身
    if amplitude < 0.5:                      # 周围本来就平滑，补噪声反而是画蛇添足
        return repaired

    height_, width_ = y1 - y0, x1 - x0
    rng = np.random.default_rng(0)           # 固定种子：同一份输入每次结果一致
    # 单通道再广播到 RGB：纸纹是亮度上的颗粒。三个通道各自随机会产生彩色噪点，
    # 放大看是一片红绿斑，比原来那块光滑区域更扎眼。
    mono = rng.standard_normal((height_, width_)).astype(np.float32)
    mono = cv2.GaussianBlur(mono, (0, 0), sigmaX=sigma * 0.6)
    noise = np.repeat(mono[:, :, None], 3, axis=2)
    spread = noise.std()
    if spread <= 1e-6:
        return repaired
    noise *= amplitude / spread

    patch = repaired[y0:y1, x0:x1].astype(np.float32) + noise
    out = repaired.copy()
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def _fill_once(
    rgb: np.ndarray, mask: np.ndarray, radius: int, ring: int, flat_tolerance: float, graft: bool
) -> np.ndarray:
    backdrop = _flat_backdrop(rgb, mask, ring, flat_tolerance)
    if backdrop is not None:
        filled = rgb.copy()
        filled[mask > 0] = backdrop
        return filled
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.cvtColor(cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    if graft:
        filled = _graft_texture(filled, rgb, mask)
    return filled


def _residue_ring(
    filled: np.ndarray,
    mask: np.ndarray,
    ink_kernel: int,
    threshold: float,
    bounds: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """修完之后，紧贴掩膜的一圈里还有没有没擦干净的墨迹。

    只看这一圈，而且不许越过检测框：角标按定义就在框内，框外的一律是画面。
    实测踩过——一条奶油色的装饰线只伸进框边 5 个像素，被当成残留吃掉之后，
    整条线看起来就断了。按颜色区分不可靠（那条线色度只有 29，比画面里别的东西还低），
    按几何边界区分才是干净的判据。
    """
    band = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & (mask == 0)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        inside_box = np.zeros_like(band)
        inside_box[y0:y1, x0:x1] = True
        band &= inside_box
    if not band.any():
        return np.zeros_like(mask)
    gray = cv2.cvtColor(filled, cv2.COLOR_RGB2GRAY)
    ink = cv2.absdiff(gray, cv2.medianBlur(gray, ink_kernel))
    return ((ink >= threshold) & band).astype(np.uint8) * 255


def _repair(
    rgb: np.ndarray,
    mask: np.ndarray,
    radius: int,
    ring: int = 6,
    flat_tolerance: float = 4.0,
    graft: bool = True,
    sweeps: int = 2,
    ink_kernel: int = 7,
    residue_threshold: float = 10.0,
    bounds: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """修复 Mask 内像素，返回 (修复后的图, 实际生效的掩膜)。

    自查会让掩膜比传进来的略大，所以必须把生效的那张传回去——否则调用方拿旧掩膜
    去复核，会把自己刚补涂的像素判成越界。
    """
    work = mask.copy()
    repaired = _fill_once(rgb, work, radius, ring, flat_tolerance, graft)
    for _ in range(max(0, sweeps)):
        extra = _residue_ring(repaired, work, ink_kernel, residue_threshold, bounds)
        if not extra.any():
            break
        work = np.maximum(work, extra)
        repaired = _fill_once(rgb, work, radius, ring, flat_tolerance, graft)
    outside = work == 0
    repaired[outside] = rgb[outside]
    if not np.array_equal(repaired[outside], rgb[outside]):
        raise RuntimeError("安全检查失败：Mask 外像素发生变化，已拒绝写出。")
    return repaired, work


def _encode_png(rgb: np.ndarray, compression: int) -> bytes:
    ok, buffer = cv2.imencode(
        ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return buffer.tobytes()


def _rebuild_page(destination: fitz.Document, page: fitz.Page, image: bytes) -> None:
    """用修复后的位图重建一页。

    位图页整页就是一张图，重建不丢东西；而直接改流要处理 Flate 预测器、
    颜色空间、SMask 一堆情况，反而更容易出错。
    尺寸取未旋转的 mediabox，插图后再设回旋转角，避免旋转页被拉变形。
    """
    box = page.mediabox
    new_page = destination.new_page(width=box.width, height=box.height)
    new_page.insert_image(new_page.rect, stream=image)
    if page.rotation:
        new_page.set_rotation(page.rotation)


def _verify(
    source: Path, destination: Path, mask: np.ndarray, indexes: Sequence[int], sample: int
) -> list[str]:
    """写出后复核：页数一致，且抽查页的 Mask 之外像素逐位未变。"""
    problems: list[str] = []
    with fitz.open(source) as before, fitz.open(destination) as after:
        if before.page_count != after.page_count:
            return [f"页数变了：{before.page_count} -> {after.page_count}"]
        checked = list(indexes)
        if 0 < sample < len(checked):
            positions = np.linspace(0, len(checked) - 1, sample).round().astype(int)
            checked = [checked[int(position)] for position in dict.fromkeys(positions.tolist())]
        outside = mask == 0
        for index in checked:
            try:
                pair = []
                for document in (before, after):
                    xref = document[index].get_images(full=True)[0][0]
                    raw = document.extract_image(xref)["image"]
                    pair.append(
                        cv2.cvtColor(
                            cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB,
                        )
                    )
            except Exception as exc:  # pragma: no cover - 取决于具体 PDF 结构
                problems.append(f"第 {index + 1} 页读回失败：{exc}")
                continue
            original, cleaned = pair
            if original.shape != cleaned.shape:
                problems.append(f"第 {index + 1} 页尺寸变了：{original.shape} -> {cleaned.shape}")
                continue
            changed = int((np.any(original != cleaned, axis=2) & outside).sum())
            if changed:
                problems.append(f"第 {index + 1} 页 Mask 之外有 {changed} 个像素被改动")
    return problems


# ---------------------------------------------------------------- 对外接口

def clean_pdf(
    source,
    destination,
    options: Optional[CleanOptions] = None,
    progress: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> AutoCleanResult:
    """自动去掉固定位置角标水印，写出一个副本；原始 PDF 始终不被修改。"""
    options = options or CleanOptions()
    source, destination = Path(source), Path(destination)
    if destination.exists() and destination.resolve() == source.resolve():
        raise ValueError("输出会覆盖源文件，已中止。请换一个输出路径。")

    document = fitz.open(source)
    cleaned: Optional[fitz.Document] = None
    try:
        refs, skipped = scan_full_bleed_pages(document, options.coverage)
        refs, mismatched = keep_dominant_size(refs)
        skipped = sorted(skipped + mismatched, key=lambda item: item.index)
        if not refs:
            return AutoCleanResult(
                success=False,
                message=f"没有可处理的满版位图页（{len(skipped)} 页被跳过），原文件未改动",
                skipped=skipped,
            )

        height, width = refs[0].height, refs[0].width
        cache = _PageCache(options.cache_budget_mb)
        if options.box:
            box = options.box
            mask = box_to_mask((height, width), box, options.dilate)
            strategy = "手工指定水印框"
            if not mask.any():
                return AutoCleanResult(False, "手工指定的水印框落在图像之外", skipped=skipped)
        else:
            detection, reason = detect_watermark(document, refs, options, progress, should_cancel, cache)
            if detection is None:
                return AutoCleanResult(False, f"未检测到水印：{reason}", skipped=skipped)
            mask, box, strategy = detection.mask, detection.box, detection.strategy

        painted = int((mask > 0).sum())
        painted_mask = mask.copy()          # 自查会逐页扩，这里累计实际涂过的并集
        area_percent = 100.0 * painted / (height * width)
        box_ratio = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)

        cleaned = fitz.open()
        processed: list[int] = []
        by_index = {ref.index: ref for ref in refs}
        verified_roundtrip = False
        pending_start: Optional[int] = None
        for index in range(document.page_count):
            _check_cancel(should_cancel)
            ref = by_index.get(index)
            rgb = None
            if ref is not None:
                rgb = cache.pop(index)
                if rgb is None:
                    rgb = _decode_page_image(document, ref)
            if rgb is None or rgb.shape[:2] != (height, width):
                # 非位图页、解码失败页原样搬过去；连续的整段一次性拷贝更快。
                if pending_start is None:
                    pending_start = index
                if ref is not None:
                    skipped.append(SkippedPage(index, "位图解码失败，本页原样保留"))
                continue
            if pending_start is not None:
                cleaned.insert_pdf(document, from_page=pending_start, to_page=index - 1)
                pending_start = None
            repaired, page_mask = _repair(rgb, mask, options.radius,
                               options.backdrop_ring, options.backdrop_tolerance,
                               options.graft_texture, options.residue_sweeps,
                               options.ink_kernel,
                               options.contrast_delta * options.residue_ratio, box)
            painted_mask = np.maximum(painted_mask, page_mask)
            image = _encode_png(repaired, options.png_compression)
            if not verified_roundtrip:
                # 只在第一页做一次编解码往返核对，确认 PNG 这条路确实无损。
                decoded = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                if not np.array_equal(decoded, repaired):
                    raise RuntimeError("安全检查失败：中间图像编码不是无损的，已拒绝写出。")
                verified_roundtrip = True
            _rebuild_page(cleaned, document[index], image)
            processed.append(index)
            if progress:
                progress(len(processed), len(refs), f"修复第 {index + 1} 页")
        if pending_start is not None:
            cleaned.insert_pdf(document, from_page=pending_start, to_page=document.page_count - 1)

        cache.clear()
        cleaned.set_metadata(document.metadata or {})
        destination.parent.mkdir(parents=True, exist_ok=True)
        cleaned.save(str(destination), garbage=4, deflate=True)
    except CleanCancelled:
        return AutoCleanResult(False, "已取消，原文件未改动", cancelled=True)
    finally:
        if cleaned is not None:
            cleaned.close()
        document.close()

    skipped = sorted(skipped, key=lambda item: item.index)
    if not processed:
        destination.unlink(missing_ok=True)
        return AutoCleanResult(False, "所有满版位图页都无法解码，原文件未改动", skipped=skipped)

    # 自查可能让每页多涂一点，复核必须拿这些页实际用过的掩膜的并集，
    # 否则要么误报越界，要么放过真正的越界。
    if int(painted_mask.sum()) > 4 * max(1, int(mask.sum())):
        return AutoCleanResult(
            success=False,
            message="自查阶段掩膜异常膨胀，已中止（可能把画面误判为残留）",
            skipped=skipped,
        )
    problems = _verify(source, destination, painted_mask, processed, options.verify_sample)
    if problems:
        return AutoCleanResult(
            success=False,
            message="已写出但复核未通过：" + "；".join(problems),
            pages_processed=len(processed),
            box=box,
            box_ratio=box_ratio,
            area_percent=area_percent,
            strategy=strategy,
            output_path=destination,
            skipped=skipped,
        )
    return AutoCleanResult(
        success=True,
        message=(
            f"{strategy}命中 x{box[0]}-{box[2]} y{box[1]}-{box[3]}，"
            f"涂改 {area_percent:.3f}%，{len(processed)} 页已处理"
        ),
        pages_processed=len(processed),
        box=box,
        box_ratio=box_ratio,
        area_percent=area_percent,
        strategy=strategy,
        output_path=destination,
        skipped=skipped,
    )


def auto_clean_pdf(
    input_path: str,
    output_path: str,
    corner_w: float = 0.30,
    corner_h: float = 0.12,
    dark: int = 120,
    vote: float = 0.85,
    dilate: int = 2,
    radius: int = 4,
    max_area: float = 0.005,
    box=None,
    progress: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> dict:
    """clean_pdf 的 dict 版封装，供 GUI 线程直接消费。"""
    options = CleanOptions(
        corner_w=corner_w,
        corner_h=corner_h,
        dark=dark,
        vote=vote,
        dilate=dilate,
        radius=radius,
        max_area=max_area,
        box=parse_box(box),
    )
    return clean_pdf(input_path, output_path, options, progress, should_cancel).as_dict()
