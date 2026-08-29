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
    """inpaint 修复半径（仅 TELEA 回退路径用）。"""
    fill_quality: str = "telea"
    """填充算法：best / fast / telea。

    默认 TELEA。best/fast 走 OpenCV contrib 的 FSR（频率选择重建），需要额外装
    opencv-contrib-python，没装会自动退回 TELEA。

    FSR 在合成基准上大胜（「细线穿过角标」场景误差 68.93 → 8.96，平均 16.81 → 4.60），
    但在两份真实绘本上**明显更差**：深蓝夜景页冒出成片绿色伪影，还凭空补出亮线
    （金线像素 33 → 45，是在编造内容）。裁剪到角标周围再跑也没有改善。
    所以默认不用它——合成基准和真实文件给出了相反的结论，以真实文件为准。
    """
    backdrop_ring: int = 6
    """判定底色是否纯净时，往 Mask 外取样多少像素宽的一圈；0 表示禁用纯色铺底。"""
    backdrop_tolerance: float = 4.0
    """取样圈内像素偏离中位数的平均值上限；不超过就认为是纯色底，直接铺底色。"""
    surface_tolerance: float = 3.0
    """渐变曲面拟合的验收线：拟合后的中位残差超过它就判定「这不是渐变」，退回 inpaint。

    定得太松，高频画面也会被硬拟合成一张平滑曲面，反而不如 inpaint。
    """
    defrost: bool = True
    """是否还原角标背后那层磨砂衬底。

    NotebookLM 给标签垫的**不是一层色板，是把底下的画面做了高斯模糊**（实测 σ≈4）。
    这一条是量出来的：拿板外没被盖住的画面按 σ=4 模糊，去预测板内实测值，
    在关键几行只差 0.5～2.6 个灰阶。

    这解释了 `remove_panel` 那条路为什么走不通——根本没有一层白色可以减掉。
    模糊是丢信息的操作，板下的细节不在文件里，任何算法都变不出来；能做的是
    **从板外把结构续进去**，再把模糊没解释掉的部分按原样叠回来。见 `_defrost`。
    """
    defrost_tolerance: float = 3.0
    """磨砂模型的验收线：按模型预测板内像素，中位误差超过它就整块放弃。

    这是这条路上最关键的一道闸门，而且是自证的：模型成立才动手，不成立就不动。
    画面越花，模型越解释不了，误差自然越大——不需要另外去判断「画面复不复杂」。
    """
    defrost_coherence: float = 3.0
    """板外取样带在水平方向的一致性上限（中位绝对偏差，灰阶）。

    重建靠的是把板外一条竖带的逐行剖面横向续进板内。前提是那一带的画面本来就
    横向连贯（横线、平底色、横向渐变）。花哨画面上硬续会拉出条纹，所以先量后做。
    """
    defrost_max_lift: float = 24.0
    """磨砂之外允许的整体亮度偏移上限。超过说明这不是磨砂，是别的东西。"""
    defrost_max_area: float = 0.02
    """磨砂衬底允许占整页的最大面积。"""
    panel_strength: float = 8.0
    """衬底反解的启动线：板内相对本页背景的亮度偏移超过它才动手。

    NotebookLM 会按背景深浅决定衬底强弱——浅色页几乎没有，深色页会加一层明显的
    亮色板。它**不是跨页固定的**，跨页交集看不见，只能逐页判断，所以判据比别处弱，
    闸门也就设得更严：偏移够大、覆盖够广、alpha 图够平滑、反解后对比度确实回升，
    四条全过才落地，任何一条不满足就整页放弃。
    """
    panel_max_alpha: float = 0.55
    """反解允许的最大 alpha。太高说明底下的原画所剩无几，除以 (1-a) 会放大噪声。"""
    panel_max_area: float = 0.02
    """衬底反解之后，涂改总面积占整页的天花板。超过就判定失控并中止。"""
    panel_box: Optional[tuple[int, int, int, int]] = None
    """用户框定的衬底范围 (x0, y0, x1, y1)，图像像素坐标。**目前不要用。**

    本以为「范围由人指定」就能绕开自动猜板的不可靠，实测仍然不成立——但当时归因
    也错了：曾判定「那片区域比背景更暗」，其实是拿来当参照的区域里混进了角标白字，
    均值被拉高。

    真正的原因见 `defrost`：衬底根本不是一层可减的颜色，是把画面模糊了。
    按「半透明叠加」去解，无论范围由谁指定都不可能对。接口留着不删，是为了
    记住这个教训的形状。
    """
    remove_panel: bool = False
    """是否启用衬底反解。**默认关闭——实测这条路走不通。**

    动机是真的：NotebookLM 会按背景深浅加一层自适应衬底，深色页上明显，
    而它不是跨页固定的，跨页交集看不见，只能逐页判断。

    但逐页判断依赖「从板外拟合背景、外推到板内」，而这个外推的误差和要测的
    衬底效应是同一量级。实测在一处根本没有衬底的地方（板外背景 74.5，板内
    74.8，本来就一致），它硬解出 alpha 并把那里压暗到 57.1——凭空造出一块
    暗色矩形。加了边缘闸门和羽化窗也没救。

    更要命的是「反解后对比度必须回升」那道闸门被骗过了：人为造出的硬边缘
    反而让对比度指标变好看。又一次指标骗人、图不骗人。

    **后来查清了根因**：衬底不是一层色板，是把底下的画面做了高斯模糊。
    没有颜色可减，所以整个 alpha 模型从一开始就套错了对象，加多少道闸门都白搭。
    正确的做法在 `defrost`。留着开关和这段记录，是为了别再走一遍。"""
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
    # 用中位绝对偏差，不是均值：取样圈常蹭到装饰线、星光这类高对比元素，
    # 均值会被少数极端值拉高，于是一块本来平整的纯色底被误判成「有纹理」，
    # 白白掉进 inpaint——而纯色底直接铺底色本可以做到像素级完美。
    deviation = np.abs(samples.astype(np.int16) - center)
    spread = np.median(deviation, axis=0)
    if spread.max() > tolerance:
        return None
    # 但也不能只看中位数：真有大片区域不同色时得拒绝，否则会把画面涂成一块死色。
    if float((deviation.max(axis=1) > tolerance * 4).mean()) > 0.25:
        return None
    return center.round().astype(np.uint8)


def _smooth_backdrop(
    rgb: np.ndarray, mask: np.ndarray, ring: int, tolerance: float
) -> Optional[np.ndarray]:
    """底色是平滑渐变时，用二次曲面从周围外推出掩膜内的底色。

    介于「纯色铺底」（零次，常数）和 inpaint 之间。渐变是可建模的，拟合出来的值
    是算的不是猜的；inpaint 则是从边界扩散平均值，在渐变上会拉出一块死板的过渡。

    拟合必须抗干扰：取样圈里常有装饰线、星光这类高对比元素，直接最小二乘会被
    带偏。这里迭代剔除离群点，再用中位残差判断拟合是否可信——判不可信就返回
    None，交给 inpaint，不硬来。
    """
    if ring <= 0:
        return None
    kernel = np.ones((ring * 4 + 1,) * 2, np.uint8)      # 比纯色判定取宽一点，才拟合得稳
    band = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    rows, columns = np.nonzero(band)
    if len(rows) < 80:
        return None

    target_rows, target_columns = np.nonzero(mask)
    if len(target_rows) == 0:
        return None

    # 坐标归一化到 [-1, 1]，避免二次项把矩阵条件数搞坏
    all_rows = np.concatenate([rows, target_rows])
    all_columns = np.concatenate([columns, target_columns])
    row_mid, row_span = all_rows.mean(), max(float(np.ptp(all_rows)) / 2.0, 1.0)
    col_mid, col_span = all_columns.mean(), max(float(np.ptp(all_columns)) / 2.0, 1.0)

    def design(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        u = (r - row_mid) / row_span
        v = (c - col_mid) / col_span
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)

    source = design(rows.astype(np.float64), columns.astype(np.float64))
    target = design(target_rows.astype(np.float64), target_columns.astype(np.float64))
    samples = rgb[band].astype(np.float64)

    predicted = np.zeros((len(target_rows), 3))
    for channel in range(3):
        values = samples[:, channel]
        weights = np.ones(len(values), bool)
        for _ in range(3):                            # 迭代剔除离群点
            coefficients, *_ = np.linalg.lstsq(source[weights], values[weights], rcond=None)
            residual = np.abs(values - source @ coefficients)
            cutoff = max(2.5 * np.median(residual[weights]), 2.0)
            updated = residual <= cutoff
            if updated.sum() < 60 or np.array_equal(updated, weights):
                break
            weights = updated
        # 中位残差就是「这块底色有多平滑」；不够平滑说明不是渐变，别硬拟合
        if float(np.median(np.abs(values - source @ coefficients)[weights])) > tolerance:
            return None
        predicted[:, channel] = target @ coefficients

    # 外推不许跑出取样圈见过的范围——二次曲面在边界外很容易失控
    low = samples.min(axis=0) - 6.0
    high = samples.max(axis=0) + 6.0
    if (predicted < low - 12).any() or (predicted > high + 12).any():
        return None
    return np.clip(predicted, low, high).round().astype(np.uint8)


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


_FSR_ALGOS = {}
if hasattr(cv2, "xphoto"):                       # opencv-contrib 才有
    for _key, _attr in (("best", "INPAINT_FSR_BEST"), ("fast", "INPAINT_FSR_FAST")):
        if hasattr(cv2.xphoto, _attr):
            _FSR_ALGOS[_key] = getattr(cv2.xphoto, _attr)


def _inpaint(rgb: np.ndarray, mask: np.ndarray, radius: int, quality: str) -> np.ndarray:
    """补上掩膜内的像素。优先用 FSR，它会续接穿过掩膜的结构。

    TELEA 是扩散法：从边界往里推平均值，一条穿过掩膜的细线会被直接抹断。
    FSR 在频域外推，能把线接上。实测「细线穿过角标」场景误差 68.93 → 8.96。
    """
    algo = _FSR_ALGOS.get(quality)
    if algo is not None:
        rows, columns = np.nonzero(mask)
        if len(rows):
            # 只在角标周围一小块上跑。FSR 是频域外推，整页送进去会被页面别处的
            # 高对比结构干扰——实测在深蓝夜景页上冒出成片绿色伪影，还凭空补出亮线。
            margin = 48
            y0 = max(0, int(rows.min()) - margin)
            y1 = min(mask.shape[0], int(rows.max()) + 1 + margin)
            x0 = max(0, int(columns.min()) - margin)
            x1 = min(mask.shape[1], int(columns.max()) + 1 + margin)
            patch, patch_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            # xphoto 的掩膜语义相反：255 = 已知，0 = 待补
            known = np.where(patch_mask > 0, 0, 255).astype(np.uint8)
            destination = np.zeros_like(patch)
            try:
                cv2.xphoto.inpaint(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR), known, destination, algo)
                filled = rgb.copy()
                filled[y0:y1, x0:x1] = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)
                return filled
            except cv2.error:
                pass                              # 退回 TELEA，不因为一张图失败就整本作废
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)


def _fill_once(
    rgb: np.ndarray,
    mask: np.ndarray,
    radius: int,
    ring: int,
    flat_tolerance: float,
    graft: bool,
    quality: str = "best",
    surface_tolerance: float = 3.0,
) -> np.ndarray:
    backdrop = _flat_backdrop(rgb, mask, ring, flat_tolerance)
    if backdrop is not None:
        filled = rgb.copy()
        filled[mask > 0] = backdrop
        return filled
    # 纯色不成立，再试渐变曲面；都不行才交给 inpaint 去猜
    surface = _smooth_backdrop(rgb, mask, ring, surface_tolerance)
    if surface is not None:
        filled = rgb.copy()
        filled[mask > 0] = surface
        if graft:
            filled = _graft_texture(filled, rgb, mask)
        return filled
    filled = _inpaint(rgb, mask, radius, quality)
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


def _remove_panel(
    rgb: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    strength: float,
    max_alpha: float,
    panel_rect: Optional[tuple[int, int, int, int]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """去掉角标背后那层自适应衬底，返回 (修好的图, 动过的区域)。

    NotebookLM 会按背景深浅决定衬底强弱：浅色页几乎没有，深色页会加一层明显的
    亮色板保证文字可读。**它不是跨页固定的**，所以跨页求交集看不见它——只能逐页判断。

    模型和角标本身一致：纯白或纯黑按 alpha 叠加。
        观察值 = (1-a)·原画 + a·C
    原画用板外一圈拟合的二次曲面外推，a 由此解出，再反解回原画。这是算的不是猜的。

    逐页统计的依据比跨页交集弱得多，所以每一步都设了闸门，任何一条不满足就返回
    (None, None) 整页放弃——宁可留着那层雾，也不能在画面上留一块变暗的方斑。
    """
    height, width = mask.shape
    x0, y0, x1, y1 = box
    if panel_rect is not None:
        # 用户框定的范围：不再猜几何。自动猜板的位置正是这条路走不通的根源——
        # 猜宽了会把干净背景也算进去，背景外推的误差就被当成衬底减掉。
        px0, py0, px1, py1 = panel_rect
        px0, py0 = max(0, int(px0)), max(0, int(py0))
        px1, py1 = min(width, int(px1)), min(height, int(py1))
    else:
        box_height = max(y1 - y0, 1)
        py0 = max(0, y0 - int(box_height * 1.8))
        py1 = min(height, y1 + int(box_height * 1.8))
        px0 = max(0, x0 - int((x1 - x0) * 0.35))
        px1 = min(width, x1 + int((x1 - x0) * 0.12))
    if py1 - py0 < 8 or px1 - px0 < 24:
        return None, None

    panel = np.zeros((height, width), bool)
    panel[py0:py1, px0:px1] = True
    # 板外一圈用来拟合「本页真实背景」，必须离板足够远，免得把衬底本身当成背景
    outer = np.zeros((height, width), bool)
    outer[max(0, py0 - 26):min(height, py1 + 26), max(0, px0 - 90):min(width, px1 + 26)] = True
    ring = outer & ~panel
    if ring.sum() < 400:
        return None, None

    rows, columns = np.nonzero(ring)
    target_rows, target_columns = np.nonzero(panel)
    all_rows = np.concatenate([rows, target_rows])
    all_columns = np.concatenate([columns, target_columns])
    row_mid, row_span = all_rows.mean(), max(float(np.ptp(all_rows)) / 2.0, 1.0)
    col_mid, col_span = all_columns.mean(), max(float(np.ptp(all_columns)) / 2.0, 1.0)

    def design(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        u = (r - row_mid) / row_span
        v = (c - col_mid) / col_span
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)

    source = design(rows.astype(np.float64), columns.astype(np.float64))
    target = design(target_rows.astype(np.float64), target_columns.astype(np.float64))
    ring_values = rgb[ring].astype(np.float64)

    background = np.zeros((len(target_rows), 3))
    for channel in range(3):
        values = ring_values[:, channel]
        keep = np.ones(len(values), bool)
        for _ in range(3):
            coefficients, *_ = np.linalg.lstsq(source[keep], values[keep], rcond=None)
            residual = np.abs(values - source @ coefficients)
            cutoff = max(2.5 * np.median(residual[keep]), 2.0)
            updated = residual <= cutoff
            if updated.sum() < 300 or np.array_equal(updated, keep):
                break
            keep = updated
        # 闸门一：本页背景本身就杂乱，拟合不出可信的参照，放弃
        if float(np.median(np.abs(values - source @ coefficients)[keep])) > 6.0:
            return None, None
        background[:, channel] = target @ coefficients

    observed = rgb[panel].astype(np.float64)
    offset = observed.mean(axis=1) - background.mean(axis=1)
    # 闸门二：衬底太弱、或者覆盖面太小，就别动——收益抵不上误判的风险
    lifted = np.abs(offset) > strength
    # 覆盖面这条闸门只在自动猜板时才严：用户亲手框的范围，由他负责
    if lifted.mean() < (0.02 if panel_rect is not None else 0.10):
        return None, None

    overlay = 255.0 if float(np.median(offset[lifted])) > 0 else 0.0
    denominator = overlay - background.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(np.abs(denominator) > 12.0, offset / denominator, 0.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0, max_alpha)
    # 没到阈值的地方先清零再平滑。否则文字附近的高 alpha 会被高斯抹到旁边
    # 本来干净的区域，把那里平白压暗——框内也不能乱动。
    alpha = np.where(lifted, alpha, 0.0)

    # alpha 图必须平滑——真实画面的亮度起伏不会像一块板那样规则
    alpha_map = np.zeros((height, width), np.float32)
    alpha_map[panel] = alpha
    smoothed = cv2.GaussianBlur(alpha_map, (0, 0), 3.0)
    roughness = float(np.abs(alpha_map[panel] - smoothed[panel]).mean())
    # 闸门三：alpha 不平滑，说明抓到的是画面而不是衬底
    if roughness > 0.05:
        return None, None
    # 闸门四：板边缘的 alpha 必须本就接近 0。不接近说明背景拟合被衬底本身污染了，
    # 硬减下去会在板的边界留一道台阶——实测就是一块边缘锐利的暗色矩形。
    # 这条只对「自动猜板」有意义：猜出来的板边缘本应落在衬底之外，不然就是猜宽了。
    # 用户亲手框定时不查——衬底本来就可能一直延伸到框边，靠下面的羽化窗防台阶。
    if panel_rect is None:
        edge = np.zeros((height, width), bool)
        edge[py0:py0 + 3, px0:px1] = True
        edge[py1 - 3:py1, px0:px1] = True
        edge[py0:py1, px0:px0 + 3] = True
        if float(np.abs(smoothed[edge]).mean()) > 0.02:
            return None, None

    # 再乘一个到边界归零的窗，杜绝任何残余台阶
    window = np.zeros((height, width), np.float32)
    window[py0:py1, px0:px1] = 1.0
    feather = min(14, max(4, int(min(py1 - py0, px1 - px0) * 0.15)))
    window = cv2.GaussianBlur(window, (0, 0), feather / 2.0)
    alpha = np.clip(smoothed[panel] * window[panel], 0.0, max_alpha)

    restored = (observed - alpha[:, None] * overlay) / np.maximum(1.0 - alpha[:, None], 0.2)
    restored = np.clip(restored, 0.0, 255.0)

    result = rgb.copy()
    result[panel] = restored.round().astype(np.uint8)
    # 掩膜内的像素由填充负责，衬底这一步不许覆盖它
    result[mask > 0] = rgb[mask > 0]

    touched = panel & (mask == 0)
    changed = np.zeros((height, width), bool)
    changed[touched] = (np.abs(result[touched].astype(np.int16) - rgb[touched].astype(np.int16))
                        .max(axis=1) > 0)
    if not changed.any():
        return None, None

    # 闸门四：反解应当把对比度还回来。如果修完反而更平了，说明模型不成立。
    def detail(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image[py0:py1, px0:px1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float((gray - cv2.GaussianBlur(gray, (0, 0), 2.0)).std())

    if detail(result) < detail(rgb) * 0.98:
        return None, None
    return result, changed


def _band_profile(
    rgb: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> tuple[Optional[np.ndarray], float]:
    """取一条竖带的逐行中位剖面，并量它在水平方向有多一致。

    返回 (剖面, 中位绝对偏差)。偏差小说明这一带的画面本来就横向连贯
    （横线、平底色、横向渐变），把剖面续到旁边去才站得住。
    """
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, float("inf")
    band = rgb[y0:y1, x0:x1].astype(np.float32)
    profile = np.median(band, axis=1)
    return profile, float(np.median(np.abs(band - profile[:, None, :])))


def _panel_edge(
    inside: np.ndarray,
    outside: Optional[np.ndarray],
    minimum: float,
    outermost: str = "first",
    consistency: float = 0.75,
    contrast: float = 0.4,
) -> Optional[tuple[int, float]]:
    """找衬底的一条直边，返回 (位置, 台阶高度)。沿 axis0 扫，两个参数是同样的扫描
    方向、不同的取样范围：`inside` 跨过板、`outside` 完全在板外。

    只按「台阶最大」找是不行的：搜索窗里最强的边几乎总是画面自己的（一条装饰横线
    的上下沿轻松有二十几个灰阶，而衬底才五六个）。衬底边的特征不是强，是**到板就
    没了**——同一个位置，板外那段不该有这个台阶。这一条把画面的边全筛掉了。

    台阶沿 axis1 取中位数，再要求大部分列同号：衬底的边是一整条直线，每列都有；
    斜穿过去的装饰线只占少数列，抬不动中位数。

    合格的位置取**最外侧**那个（`outermost` 指明扫描方向是由外向内还是相反），
    不取台阶最大的那个：板内被模糊摊开的亮结构，其上升沿同样是一道「板外没有的
    台阶」，而且往往比板边本身更陡——按最大值挑会把边界定到板内去。
    """
    length = inside.shape[0]
    # 带子短的时候（板底紧贴页面下缘就会这样）固定取 3 会把最后几行扫不到
    thickness = max(1, min(3, length // 4))
    best: Optional[tuple[int, float]] = None
    for index in range(thickness, length - thickness):
        column = (inside[index:index + thickness].mean(axis=0)
                  - inside[index - thickness:index].mean(axis=0))
        step = float(np.median(column))
        if abs(step) < minimum:
            continue
        if float((np.sign(column) == np.sign(step)).mean()) < consistency:
            continue
        if outside is not None and outside.shape[1] >= 6:
            beyond = (outside[index:index + thickness].mean(axis=0)
                      - outside[index - thickness:index].mean(axis=0))
            if abs(float(np.median(beyond))) > abs(step) * contrast:
                continue                     # 板外也有这道台阶，那是画面的边
        if best is None or outermost == "last":
            best = (index, step)
        if outermost == "first":
            break
    return best


_FROST_SIGMAS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0)
_FROST_MIN_STEP = 2.5


def _frost_panel(
    repaired: np.ndarray, box: tuple[int, int, int, int]
) -> Optional[tuple[int, int, int, int]]:
    """找出磨砂衬底的矩形范围，靠的是它四条边上的台阶。

    衬底是深是浅取决于它底下画的是什么，所以不能问「亮了还是暗了」，只能问
    「四条边的台阶是不是同向、是不是一整条」。三条边（上、下、左）都得对上，
    右边允许一直延伸到页面边缘——实测它就是这么画的。

    也试过按区域找：先按「颗粒消失」（模糊必然抹平高频），实测板外那圈平滑背景
    颗粒 1.06、板内 0.2～1.9，分不开；再按「横向续接解释不了的地方」，结果把
    板外斜穿上去的装饰弧线一起圈了进来，包围盒被拉大一倍。边界才是衬底的定义。
    """
    height, width = repaired.shape[:2]
    x0, y0, x1, y1 = box
    box_height, box_width = max(y1 - y0, 1), max(x1 - x0, 1)
    # 衬底总是贴着角标长，搜索范围按角标自身尺寸放大，不去扫整页
    wy0, wy1 = max(0, y0 - int(box_height * 2.5)), min(height, y1 + int(box_height * 2.5))
    wx0, wx1 = max(0, x0 - int(box_width * 0.6)), min(width, x1 + int(box_width * 0.3))
    if y0 - wy0 < 8 or wy1 - y1 < 8 or x0 - wx0 < 16:
        return None

    gray = cv2.cvtColor(repaired, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 上下边：板内那段（角标的横向跨度）有台阶，而它左边那段没有
    beyond = gray[:, wx0:x0]
    top = _panel_edge(gray[wy0:y0, x0:x1], beyond[wy0:y0], _FROST_MIN_STEP, "first")
    if top is None:
        return None
    sign = np.sign(top[1])                   # 衬底相对画面是抬亮还是压暗，四条边得一致
    py0 = wy0 + top[0]

    bottom = _panel_edge(gray[y1:wy1, x0:x1], beyond[y1:wy1], _FROST_MIN_STEP, "last")
    if bottom is None or np.sign(-bottom[1]) != sign:
        return None
    py1 = y1 + bottom[0]

    # 左边界可能落在角标框内侧一点点，扫描范围往右多给两成；
    # 「板外」这次取板上方那几行——竖直的板边在那里同样不该存在
    limit = min(width, x0 + int(box_width * 0.2))
    left = _panel_edge(gray[py0:py1, wx0:limit].T,
                       gray[max(0, py0 - 12):py0, wx0:limit].T, _FROST_MIN_STEP, "first")
    if left is None or np.sign(left[1]) != sign:
        return None
    px0 = wx0 + left[0]

    # 右边界经常就是页面边缘，找不到台阶不算失败
    right_from = max(x1, px0 + 8)
    right = _panel_edge(gray[py0:py1, right_from:wx1].T,
                        gray[max(0, py0 - 12):py0, right_from:wx1].T, _FROST_MIN_STEP, "last")
    px1 = right_from + right[0] if right is not None and np.sign(-right[1]) == sign else wx1

    if px0 > x0 + int(box_width * 0.2) or px1 < x1 or py0 > y0 or py1 < y1:
        return None                          # 板得把角标整个包住，否则认错了
    if py1 - py0 < box_height or px1 - px0 < box_width:
        return None
    return px0, py0, px1, py1


def _defrost(
    repaired: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    tolerance: float,
    coherence: float,
    max_lift: float,
    max_area: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """还原被磨砂衬底模糊掉的画面，返回 (修好的图, 动过的区域)。

    模型是量出来的，不是猜的——NotebookLM 给标签垫的不是一层色板，是把底下的
    画面做了高斯模糊：

        观察值 = 高斯模糊(原画, σ) + 整体偏移

    拿板外没被盖住的画面按 σ=4 模糊去预测板内实测值，在关键几行只差 0.5～2.6 个
    灰阶。σ 和偏移都从本页数据里拟合，不写死。

    原画在板内是未知的，但衬底两侧的画面还在，而这类页面在衬底那一带本来就是横向
    连贯的（横线、平底色、横向渐变），所以把板外一条竖带的逐行剖面横向续进来，
    就是对原画的一个可用估计。记它为 T、真实原画为 T + Δ，剩下的是代数：

        观察值 - 偏移 = 模糊(T) + 模糊(Δ)
        还原值 = T + (观察值 - 偏移 - 模糊(T)) = T + 模糊(Δ)

    也就是：**知道的那部分结构按锐利的补回去，不知道的那部分（Δ，比如斜穿过去的
    装饰弧线）保持它模糊的样子叠在上面。** 两头都不极端——既不抹掉弧线，也不去
    编造它的锐利形状；而这里本来就没有磨砂时（σ→0、偏移→0）式子退化成
    「还原值 = 观察值」，一个像素都不会动。

    模糊丢掉的高频是真找不回来了。这里做的是**从板外把结构续进去**，不是从模糊里
    解出原画——后者不可能。
    """
    height, width = mask.shape
    found = _frost_panel(repaired, box)
    if found is None:
        return None, None
    px0, py0, px1, py1 = found
    if (py1 - py0) * (px1 - px0) > max_area * height * width:
        return None, None

    strip, pad = 24, 24                      # 剖面要向上下各多取一段，模糊才有料可卷
    qy0, qy1 = max(0, py0 - pad), min(height, py1 + pad)
    left, left_spread = _band_profile(repaired, qy0, qy1, max(0, px0 - strip), px0)
    right, right_spread = _band_profile(repaired, qy0, qy1, px1, min(width, px1 + strip))
    use_left = left is not None and left_spread <= coherence
    use_right = right is not None and right_spread <= coherence
    if not (use_left or use_right):
        return None, None                    # 两侧的画面横向都不连贯，剖面续不过去

    positions = np.arange(px0, px1, dtype=np.float32)
    if use_left and use_right:
        # 两侧都能用就横向线性过渡，板宽一点也不会在另一头对不上
        weight = ((px1 - 1 - positions) / max(px1 - 1 - px0, 1))[None, :, None]
        base = left[:, None, :] * weight + right[:, None, :] * (1.0 - weight)
    else:
        base = np.repeat((left if use_left else right)[:, None, :], px1 - px0, axis=1)
    base = np.ascontiguousarray(base, dtype=np.float32)

    observed = repaired[py0:py1, px0:px1].astype(np.float32)
    known = mask[py0:py1, px0:px1] == 0      # 角标压着的地方，底下是什么已经没了
    if int(known.sum()) < 200 or int(known.sum(axis=1).min()) < 8:
        return None, None
    inside = slice(py0 - qy0, py1 - qy0)

    # 板内的逐行中位剖面。角标是字形掩膜，字与字之间留着的都是真的板内像素，
    # 所以即使在文字那几行也取得到足够样本。
    columns = np.ma.masked_array(observed, ~np.repeat(known[:, :, None], 3, axis=2))
    profile_in = np.ma.median(columns, axis=1).filled(np.nan)
    if not np.isfinite(profile_in).all():
        return None, None

    # 闸门：板内剖面必须能被「板外剖面模糊一下再整体平移」解释。这是个一维拟合，
    # 只有二三十行，比逐像素拟合稳得多，而它问的正是「这块到底是不是磨砂」。
    base_profile = np.ascontiguousarray(np.median(base, axis=1), dtype=np.float32)
    best: Optional[tuple[float, np.ndarray]] = None
    for sigma in _FROST_SIGMAS:
        predicted = cv2.GaussianBlur(base_profile, (1, 0), sigmaX=0, sigmaY=sigma)[inside]
        lift = np.median(profile_in - predicted, axis=0)
        error = float(np.median(np.abs(profile_in - predicted - lift)))
        if best is None or error < best[0]:
            best = (error, lift)
    error, lift = best
    # 自证的闸门：模型解释得了才动手。画面越花，横向续接越不成立，误差越大，
    # 自动就退出了——不必另外去猜「这页画面复不复杂」。
    if error > tolerance or float(np.abs(lift).max()) > max_lift:
        return None, None

    # 真正的重建只用一条恒等式，不依赖上面拟合出来的 σ 和偏移：
    #
    #     还原值 = 板内像素 + （续过来的剖面 - 板内该行的中位数）
    #
    # 括号里是逐行的常数，等价于「把板内这一行的整体水平抬回它本该有的位置」。
    # σ、偏移拟合的误差同样是逐行常数，正好在这一步被抵消掉——一开始把拟合结果
    # 直接用进重建，σ 差一点就在线的上方压出一道暗带。
    deviation = observed - profile_in[:, None, :]
    correction = base[inside] - profile_in[:, None, :]

    # 但「这一行整体抬多少」只对横向连贯的地方成立。板内横向起伏大的地方——斜穿
    # 过去的装饰弧线、以及横线拐弯离开之后的那一段——续接本来就不该生效，
    # 否则会把一条本该拐上去的线直愣愣地画到页边。所以逐像素按起伏大小退让。
    # 起伏要先平滑再算权重，而且**只能数角标之外的像素**：
    #  · 不平滑，权重会跟着颗粒抖，而它乘的是一个逐行常数，结果整块斑驳；
    #  · 把角标那片算进来，填充留下的起伏会顺着滤波窗漏到相邻几行，
    #    把线上那几行的权重压到 0.6~0.8，重建出来的线只有真实亮度的八成。
    valid = known.astype(np.float32)
    total = cv2.boxFilter(valid, -1, (5, 5), normalize=False)
    swing = cv2.boxFilter(np.abs(deviation).max(axis=2) * valid, -1, (5, 5), normalize=False)
    swing = np.where(total > 0.5, swing / np.maximum(total, 1e-3), 0.0)
    # 权重要以「本页板内典型起伏」为基准，不能拿 0 当基准：颗粒本身就有两三个灰阶，
    # 按 1 - 起伏/上限 算的话，平平无奇的地方权重就掉到 0.75，线只能重建出八成亮度。
    typical = float(np.median(swing[known]))
    floor, limit = max(2.0, 2.0 * typical), max(10.0, 6.0 * typical)
    weight = cv2.GaussianBlur(
        np.clip((limit - swing) / (limit - floor), 0.0, 1.0).astype(np.float32), (0, 0), 1.5)
    restored = observed + correction * weight[:, :, None]

    # 角标压着的那片没有任何可信信息，整片换成续过来的剖面；羽化免得留边
    hole = np.clip(cv2.GaussianBlur((mask[py0:py1, px0:px1] > 0).astype(np.float32),
                                    (0, 0), 1.5), 0.0, 1.0)[:, :, None]
    restored = restored * (1.0 - hole) + base[inside] * hole

    # 兜底：重建值不许超出取样带见过的范围，杜绝外推跑飞
    stack = np.concatenate([p for p, ok in ((left, use_left), (right, use_right)) if ok])
    restored = np.clip(restored, stack.min(axis=0) - 6.0, stack.max(axis=0) + 6.0)

    # 板边界向内羽化两像素。边界处重建值本就等于板外画面，这一步只是兜底不留台阶
    feather = np.zeros((height, width), np.float32)
    feather[py0:py1, px0:px1] = 1.0
    feather = cv2.GaussianBlur(feather, (0, 0), 1.2)[py0:py1, px0:px1][:, :, None]

    result = repaired.copy()
    result[py0:py1, px0:px1] = np.clip(
        observed * (1.0 - feather) + restored * feather, 0.0, 255.0
    ).round().astype(np.uint8)

    # 这里**不**补颗粒。模糊确实把颗粒也抹掉了（板外 1.68、板内 0.23），补完
    # 量化上更接近（1.98 vs 1.68），但放大看是一片比原来那层雾还扎眼的斑块——
    # 合成噪声是相关的团块，和纸纹那种细密颗粒不是一回事。又一次指标骗人。
    # 何况板内的平滑本来就是原件里的样子，去掉染色、把结构续回去是还原，
    # 凭空造颗粒不是。
    panel = np.zeros((height, width), np.uint8)
    panel[py0:py1, px0:px1] = 255

    changed = np.abs(result.astype(np.int16) - repaired.astype(np.int16)).max(axis=2) > 0
    touched = (panel > 0) & changed
    if not touched.any():
        return None, None
    return result, touched


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
    quality: str = "best",
    surface_tolerance: float = 3.0,
    panel: bool = True,
    panel_strength: float = 8.0,
    panel_max_alpha: float = 0.55,
    panel_rect: Optional[tuple[int, int, int, int]] = None,
    bounds: Optional[tuple[int, int, int, int]] = None,
    defrost: bool = True,
    defrost_tolerance: float = 3.0,
    defrost_coherence: float = 3.0,
    defrost_max_lift: float = 24.0,
    defrost_max_area: float = 0.02,
) -> np.ndarray:
    """修复 Mask 内像素，返回 (修复后的图, 实际生效的掩膜)。

    自查会让掩膜比传进来的略大，所以必须把生效的那张传回去——否则调用方拿旧掩膜
    去复核，会把自己刚补涂的像素判成越界。
    """
    work = mask.copy()
    # 自查阶段只是「找哪里还没擦干净」，用最便宜的填充就够；高质量填充留到最后一次，
    # 否则 FSR 会被重复跑三遍，每页多花两秒而结果完全一样。
    probe = _fill_once(rgb, work, radius, ring, flat_tolerance, False, "telea", surface_tolerance)
    for _ in range(max(0, sweeps)):
        extra = _residue_ring(probe, work, ink_kernel, residue_threshold, bounds)
        if not extra.any():
            break
        work = np.maximum(work, extra)
        probe = _fill_once(rgb, work, radius, ring, flat_tolerance, False, "telea", surface_tolerance)
    repaired = _fill_once(rgb, work, radius, ring, flat_tolerance, graft, quality, surface_tolerance)
    if defrost and bounds is not None:
        # 先把掩膜外的像素还原。_fill_once 会波及掩膜之外（inpaint 的扩散边、
        # 补颗粒时按包围盒整块加噪声），这些本来都由函数末尾统一还原；但磨砂那步
        # 要从板外取剖面、要量板内起伏，喂给它一张被污染的图，取样带和板内统计
        # 全都会偏——实测最后一页因此过不了模型闸门。
        repaired[work == 0] = rgb[work == 0]
        # 必须在填充之后：那时角标已被抹平，板内才只剩磨砂本身的痕迹
        thawed, touched = _defrost(repaired, rgb, work, bounds, defrost_tolerance,
                                   defrost_coherence, defrost_max_lift, defrost_max_area)
        if thawed is not None:
            repaired = thawed
            work = np.maximum(work, (touched * 255).astype(np.uint8))
    if (panel or panel_rect is not None) and bounds is not None:
        restored, touched = _remove_panel(repaired, work, bounds, panel_strength,
                                          panel_max_alpha, panel_rect)
        if restored is not None:
            repaired = restored
            work = np.maximum(work, (touched * 255).astype(np.uint8))

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
                               options.contrast_delta * options.residue_ratio,
                               options.fill_quality, options.surface_tolerance,
                               options.remove_panel, options.panel_strength,
                               options.panel_max_alpha, options.panel_box, box,
                               options.defrost, options.defrost_tolerance,
                               options.defrost_coherence, options.defrost_max_lift,
                               options.defrost_max_area)
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
    # 两道上限：自查最多把掩膜撑到 4 倍；磨砂衬底那一步会合法地多动一整块板
    # （实测约整页 0.27%，远超角标本身的 4 倍），所以再给一个按整页面积算的天花板。
    # 两条都超才算失控——只超前者多半只是衬底那步生效了。
    painted_total = int((painted_mask > 0).sum())   # 数像素个数，不是把 255 加起来
    # 天花板：自动模式按整页比例；用户框定衬底时按他画的框算——能动多少由那个框决定，
    # 再多给两成余量容纳羽化边。这样闸门仍能拦住失控，又不会否掉用户的正当选择。
    ceiling = max(options.panel_max_area, options.defrost_max_area) * height * width
    if options.panel_box is not None:
        bx0, by0, bx1, by1 = options.panel_box
        ceiling = max(ceiling, abs(bx1 - bx0) * abs(by1 - by0) * 1.2 + (mask > 0).sum())
    if (painted_total > 4 * max(1, int((mask > 0).sum()))
            and painted_total > ceiling):
        return AutoCleanResult(
            success=False,
            message=(f"修复区异常膨胀到整页 {100.0 * painted_total / (height * width):.2f}%，"
                     "已中止（可能把画面误判为水印或衬底）"),
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
