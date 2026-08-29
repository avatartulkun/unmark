from __future__ import annotations

"""去掉 PPTX 里跨页固定的角标水印（NotebookLM 导出等）。全程离线。

和 PDF 那条路同一个判据——**内容在变、它不变**——但做法完全不同，也好得多：

PPTX 里的水印不是像素，是一个形状对象。所以这里不涂、不填、不重编码，
**直接把那个对象删掉**。后果是：

  · 无损。其余每一个字、每一张图，二进制原样保留
  · 文件不会变大（PDF 那条路因为要无损重编码位图，输出常常涨到几倍）
  · 没有「修复得像不像」这种问题，也就没有磨砂衬底那一堆麻烦

母版和版式上的水印也算数：版式上的一个形状会出现在所有用它的页面上，
所以按「它实际覆盖到多少页」来算证据，和逐页放置的水印同一个标准。

安全保证与 PDF 那条路一致：原文件永不改写，结果写到新文件；写出后重新打开，
逐页核对「该留的一个没少、该删的一个没剩」。
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from pptx import Presentation
from pptx.util import Emu

ProgressCallback = Optional[Callable[[int, int, str], None]]

_WHITESPACE = re.compile(r"\s+")
_XML_ID = re.compile(rb'\sid="\d+"')
_XML_NAME = re.compile(rb'\sname="[^"]*"')


@dataclass(frozen=True)
class PptxOptions:
    """检测参数；默认值针对导出工具加在角落的那种角标。"""

    vote: float = 0.85
    """一个形状要覆盖到多大比例的页面，才算「跨页固定」。

    和 PDF 那条路取同一个值。留出余量是因为封面、封底常常用不同的版式，
    水印在那几页可能确实没有。
    """
    min_slides: int = 3
    """少于这么多页就不做自动判定——两页之间「都有」不足以说明什么。"""
    max_area: float = 0.08
    """自动模式下，一个候选最多能占页面多大面积。角标是小东西。"""
    margin_w: float = 0.25
    margin_h: float = 0.15
    """页面中央「正文区」的边距比例。自动模式只动完全落在这块之外的形状。

    水印待在页边，正文待在中间。压到正文区的东西就算跨页固定，也更可能是
    版式元素（标题栏、装饰块），不该替用户做主删掉。
    """
    outer_only: bool = True
    """自动模式是否只考虑落在正文区之外的形状。关掉就只剩「跨页固定 + 够小」。"""


@dataclass(frozen=True)
class Mark:
    """一处跨页固定的元素。`key` 用来在调用方和引擎之间指认它。"""

    key: str
    kind: str
    """text / picture / shape。"""
    label: str
    """给人看的名字：文字内容，或「图片 12.4 KB」。"""
    home: str
    """它长在哪儿：slide（逐页放置）/ layout（版式）/ master（母版）。"""
    hits: int
    """实际覆盖到多少页。"""
    total: int
    rect: tuple[int, int, int, int]
    """位置和尺寸，EMU（1 英寸 = 914400）。"""
    area_percent: float
    outer: bool
    """是否完全落在正文区之外。"""
    eligible: bool
    """自动模式会不会动它。"""
    reason: str
    """不合格时说明原因，合格时为空。"""


@dataclass(frozen=True)
class PptxResult:
    success: bool
    message: str
    slides: int = 0
    removed: int = 0
    marks: list[Mark] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    output_path: Optional[Path] = None

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "slides": self.slides,
            "removed": self.removed,
            "removed_keys": list(self.removed_keys),
            "marks": [
                {
                    "key": m.key, "kind": m.kind, "label": m.label, "home": m.home,
                    "hits": m.hits, "total": m.total, "area_percent": m.area_percent,
                    "outer": m.outer, "eligible": m.eligible, "reason": m.reason,
                }
                for m in self.marks
            ],
            "output": str(self.output_path) if self.output_path else None,
        }


# ---------------------------------------------------------------- 形状指纹

def _text_of(shape) -> str:
    if not getattr(shape, "has_text_frame", False):
        return ""
    try:
        return _WHITESPACE.sub(" ", shape.text_frame.text or "").strip()
    except Exception:
        return ""


def _content_key(shape) -> tuple[str, str]:
    """返回 (类别, 内容指纹)。指纹要跨页稳定，所以不能带对象 id。"""
    text = _text_of(shape)
    if text:
        return "text", hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    try:                                       # 图片按像素内容认，改名换 id 都躲不掉
        blob = shape.image.blob
        return "picture", hashlib.sha1(blob).hexdigest()[:16]
    except Exception:
        pass
    try:
        # 兜底：拿形状自己的 XML 当指纹，但要抹掉逐页递增的 id 和自动生成的 name，
        # 否则同一个水印在每页的指纹都不一样，跨页交集永远为空。
        xml = shape._element.xml.encode("utf-8")
        xml = _XML_NAME.sub(b"", _XML_ID.sub(b"", xml))
        return "shape", hashlib.sha1(xml).hexdigest()[:16]
    except Exception:
        return "shape", "unknown"


def _rect_of(shape) -> Optional[tuple[int, int, int, int]]:
    """位置四元组，取整到百分之一英寸——同一个水印在各页的坐标本该完全相同，
    取整只是为了容忍导出工具偶尔差一两个 EMU。"""
    try:
        values = (shape.left, shape.top, shape.width, shape.height)
    except Exception:
        return None
    if any(v is None for v in values):
        return None                            # 继承来的占位符，没有自己的坐标
    grid = 9144                                # 1/100 英寸
    return tuple(int(round(int(v) / grid)) * grid for v in values)


def _signature(shape) -> Optional[tuple]:
    rect = _rect_of(shape)
    if rect is None:
        return None
    kind, content = _content_key(shape)
    return (kind, content, rect)


def _label_of(shape, kind: str) -> str:
    text = _text_of(shape)
    if text:
        return text if len(text) <= 60 else text[:57] + "…"
    if kind == "picture":
        try:
            return f"图片 {len(shape.image.blob) / 1024:.1f} KB"
        except Exception:
            return "图片"
    name = (getattr(shape, "name", "") or "").strip()
    return f"图形「{name}」" if name else "图形"


# ---------------------------------------------------------------- 扫描

def _visible_shapes(container) -> list:
    """容器里会真正画出来的顶层形状。

    版式和母版上的占位符是「提示框」，不填内容就不出现在页面上，所以排除；
    页面自己的占位符已经有内容，照常参与。
    """
    out = []
    for shape in container.shapes:
        try:
            if getattr(shape, "is_placeholder", False):
                continue
        except Exception:
            pass
        out.append(shape)
    return out


def _shows_master(part) -> bool:
    """版式/页面有没有关掉「显示母版图形」。"""
    element = getattr(part, "element", None)
    if element is None:
        return True
    return element.get("showMasterSp", "1") not in ("0", "false")


def scan_pptx(source: Path | str, options: PptxOptions = PptxOptions()) -> tuple[list[Mark], int, str]:
    """列出这份 PPTX 里所有跨页固定的元素。返回 (候选表, 页数, 出错原因)。

    候选表按「覆盖页数 → 面积从小到大」排，最像角标的排在前面。
    """
    presentation = Presentation(str(source))
    slides = list(presentation.slides)
    total = len(slides)
    if total == 0:
        return [], 0, "这份文件里没有幻灯片"

    width = int(presentation.slide_width or 0)
    height = int(presentation.slide_height or 0)
    if width <= 0 or height <= 0:
        return [], total, "取不到页面尺寸"

    # 每个指纹记：覆盖到哪些页、它长在哪儿、一个代表形状
    hits: dict[tuple, set[int]] = {}
    home: dict[tuple, str] = {}
    sample: dict[tuple, object] = {}

    def note(signature, index: int, where: str, shape) -> None:
        if signature is None:
            return
        hits.setdefault(signature, set()).add(index)
        # 逐页放置的优先级最低，母版最高——同一个指纹如果两处都有，
        # 删源头（母版/版式）才是一次性解决。
        rank = {"slide": 0, "layout": 1, "master": 2}
        if signature not in home or rank[where] > rank[home[signature]]:
            home[signature] = where
            sample[signature] = shape
        sample.setdefault(signature, shape)

    for index, slide in enumerate(slides):
        for shape in _visible_shapes(slide):
            note(_signature(shape), index, "slide", shape)
        layout = getattr(slide, "slide_layout", None)
        if layout is None:
            continue
        for shape in _visible_shapes(layout):
            note(_signature(shape), index, "layout", shape)
        if not (_shows_master(layout) and _shows_master(slide)):
            continue
        master = getattr(layout, "slide_master", None)
        if master is None:
            continue
        for shape in _visible_shapes(master):
            note(_signature(shape), index, "master", shape)

    # 正文区：中间那一块。水印待在页边，正文待在中间。
    body = (width * options.margin_w, height * options.margin_h,
            width * (1 - options.margin_w), height * (1 - options.margin_h))

    marks: list[Mark] = []
    for signature, pages in hits.items():
        count = len(pages)
        if count < max(2, int(total * options.vote + 0.5)):
            continue
        kind, content, rect = signature
        left, top, shape_w, shape_h = rect
        area = 100.0 * (shape_w * shape_h) / float(width * height)
        outer = not (left < body[2] and left + shape_w > body[0]
                     and top < body[3] and top + shape_h > body[1])

        reason = ""
        if total < options.min_slides:
            reason = f"只有 {total} 页，跨页比对不可靠"
        elif area > options.max_area * 100:
            reason = f"占了页面 {area:.1f}%，比角标大得多"
        elif options.outer_only and not outer:
            reason = "压在正文区里，更像版式元素"
        marks.append(Mark(
            key=f"{kind}:{content}:{left}:{top}",
            kind=kind,
            label=_label_of(sample[signature], kind),
            home=home[signature],
            hits=count,
            total=total,
            rect=rect,
            area_percent=area,
            outer=outer,
            eligible=not reason,
            reason=reason,
        ))

    marks.sort(key=lambda m: (-m.hits, m.area_percent))
    return marks, total, ""


# ---------------------------------------------------------------- 清理

def _drop(shape) -> None:
    parent = shape._element.getparent()
    if parent is not None:
        parent.remove(shape._element)


def _inventory(path: Path) -> tuple[int, list[set[tuple]], set[str]]:
    """把一份 PPTX 的内容摊平成可比对的形式，用来做写出后的复核。"""
    presentation = Presentation(str(path))
    per_slide: list[set[tuple]] = []
    texts: set[str] = set()
    for slide in presentation.slides:
        signatures = set()
        for shape in _visible_shapes(slide):
            signature = _signature(shape)
            if signature is not None:
                signatures.add(signature)
            text = _text_of(shape)
            if text:
                texts.add(text)
        per_slide.append(signatures)
    for master in presentation.slide_masters:
        for container in [master] + list(master.slide_layouts):
            for shape in _visible_shapes(container):
                text = _text_of(shape)
                if text:
                    texts.add(text)
    return len(per_slide), per_slide, texts


def clean_pptx(
    source: Path | str,
    destination: Path | str,
    options: PptxOptions = PptxOptions(),
    select: Optional[Sequence[str]] = None,
    progress: ProgressCallback = None,
) -> PptxResult:
    """删掉跨页固定的角标，写出新文件。原文件永不改写。

    `select` 给定要删的 `Mark.key` 列表；不给就用自动模式——只删 `eligible` 的那些。
    """
    source, destination = Path(source), Path(destination)
    if progress:
        progress(0, 1, "分析各页")
    marks, total, why = scan_pptx(source, options)
    if why:
        return PptxResult(False, why, slides=total)

    if select is None:
        wanted = {m.key for m in marks if m.eligible}
    else:
        wanted = set(select)
        unknown = wanted - {m.key for m in marks}
        if unknown:
            return PptxResult(False, f"指定的对象不在候选里：{sorted(unknown)[:3]}",
                              slides=total, marks=marks)
    if not wanted:
        hint = "；".join(f"「{m.label}」{m.reason}" for m in marks[:3] if m.reason)
        return PptxResult(
            False,
            "没找到跨页固定的角标" + (f"（最接近的：{hint}）" if hint else ""),
            slides=total, marks=marks,
        )

    chosen = {m.key: m for m in marks if m.key in wanted}
    before_digest = hashlib.sha1(source.read_bytes()).hexdigest()

    presentation = Presentation(str(source))
    slides = list(presentation.slides)
    removed = 0

    def sweep(container, where: str) -> None:
        nonlocal removed
        for shape in list(_visible_shapes(container)):
            signature = _signature(shape)
            if signature is None:
                continue
            kind, content, rect = signature
            key = f"{kind}:{content}:{rect[0]}:{rect[1]}"
            mark = chosen.get(key)
            if mark is not None and mark.home == where:
                _drop(shape)
                removed += 1

    for index, slide in enumerate(slides):
        if progress:
            progress(index + 1, len(slides), f"清理第 {index + 1} 页")
        sweep(slide, "slide")
    for master in presentation.slide_masters:
        sweep(master, "master")
        for layout in master.slide_layouts:
            sweep(layout, "layout")

    if removed == 0:
        return PptxResult(False, "没有对象被删除，原文件未改动", slides=total, marks=marks)

    destination.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(destination))

    if hashlib.sha1(source.read_bytes()).hexdigest() != before_digest:
        raise RuntimeError("安全检查失败：原文件被改动了。")

    if progress:
        progress(len(slides), len(slides), "复核")
    problems = _verify(source, destination, chosen.values(), options)
    if problems:
        destination.unlink(missing_ok=True)
        return PptxResult(False, "已中止：" + "；".join(problems), slides=total, marks=marks)

    where = "、".join(sorted({{"slide": "逐页", "layout": "版式", "master": "母版"}[m.home]
                             for m in chosen.values()}))
    listed = "、".join(f"「{m.label}」" for m in list(chosen.values())[:3])
    return PptxResult(
        True,
        f"删掉 {len(chosen)} 个跨页固定对象（{where}）：{listed}，共 {removed} 处，{total} 页已处理",
        slides=total, removed=removed, marks=marks,
        removed_keys=sorted(chosen), output_path=destination,
    )


def _verify(source: Path, destination: Path, chosen, options: PptxOptions) -> list[str]:
    """写出后重新打开核对：该留的一个没少，该删的一个没剩。

    删除是很容易做过头的操作——选错指纹就会把正文里的东西一起带走，而且不像
    涂像素那样一眼看得出来。所以这里不信内存里的状态，只信重新读回来的文件。
    """
    problems: list[str] = []
    src_count, src_slides, _src_texts = _inventory(source)
    out_count, out_slides, out_texts = _inventory(destination)
    if src_count != out_count:
        return [f"页数从 {src_count} 变成了 {out_count}"]

    targets = {m.key for m in chosen}
    for index, (before, after) in enumerate(zip(src_slides, out_slides)):
        expected = {s for s in before
                    if f"{s[0]}:{s[1]}:{s[2][0]}:{s[2][1]}" not in targets}
        if after != expected:
            lost = len(expected - after)
            extra = len(after - expected)
            problems.append(f"第 {index + 1} 页少了 {lost} 个、多了 {extra} 个对象")
            break

    for mark in chosen:
        if mark.kind == "text" and mark.label.rstrip("…") and not mark.label.endswith("…"):
            if mark.label in out_texts:
                problems.append(f"「{mark.label}」在清理后仍然存在")
                break
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="去掉 PPTX 里跨页固定的角标水印")
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--list", action="store_true", help="只列出候选，不改动")
    parser.add_argument("--select", action="append", default=None, help="指定要删的 key，可重复")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if args.list:
        marks, total, why = scan_pptx(source)
        if why:
            print(why)
            return 1
        print(f"{total} 页，找到 {len(marks)} 个跨页固定对象：")
        for mark in marks:
            flag = "✓" if mark.eligible else "·"
            print(f"  {flag} [{mark.kind}] {mark.label}  {mark.hits}/{mark.total} 页 "
                  f"{mark.home} {mark.area_percent:.2f}%"
                  + (f"  （不动：{mark.reason}）" if mark.reason else ""))
            print(f"      key={mark.key}")
        return 0

    destination = Path(args.destination) if args.destination else \
        source.with_name(f"{source.stem}_已去水印.pptx")
    result = clean_pptx(source, destination, select=args.select)
    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
