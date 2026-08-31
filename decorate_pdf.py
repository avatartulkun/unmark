from __future__ import annotations

"""给已清理的 PDF 叠加自定义 Logo 或文字，不改动原结果。"""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageOps


POSITIONS = {"top-left", "top-right", "bottom-left", "bottom-right", "center"}
FONTS = {"sans": "helv", "serif": "tiro", "mono": "cour", "cjk": "china-s"}


def _box(page: fitz.Page, width: float, height: float, position: str, margin: float,
         x_ratio: float | None = None, y_ratio: float | None = None) -> fitz.Rect:
    if position not in POSITIONS:
        raise ValueError("位置参数无效")
    pw, ph = page.rect.width, page.rect.height
    if x_ratio is not None and y_ratio is not None:
        x = max(0, min(pw - width, pw * x_ratio - width / 2))
        y = max(0, min(ph - height, ph * y_ratio - height / 2))
    else:
        x = (pw - width) / 2 if position == "center" else (margin if position.endswith("left") else pw - margin - width)
        y = (ph - height) / 2 if position == "center" else (margin if position.startswith("top") else ph - margin - height)
    return fitz.Rect(x, y, x + width, y + height)


def _save(document: fitz.Document, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    document.save(destination, garbage=3, deflate=True)
    with fitz.open(destination) as check:
        if check.page_count != document.page_count:
            raise RuntimeError("导出复核失败")


def add_logo(source: Path, destination: Path, image_bytes: bytes, *, position: str,
             width_ratio: float, opacity: float, margin_ratio: float,
             x_ratio: float | None = None, y_ratio: float | None = None) -> None:
    with Image.open(BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        if image.width * image.height > 16_000_000:
            raise ValueError("Logo 图片不能超过 1600 万像素")
        alpha = image.getchannel("A").point(lambda value: round(value * opacity))
        image.putalpha(alpha)
        blob = BytesIO()
        image.save(blob, "PNG", optimize=True)
        logo = blob.getvalue()
        aspect = image.height / image.width

    with fitz.open(source) as document:
        for page in document:
            width = page.rect.width * width_ratio
            height = width * aspect
            max_height = page.rect.height * .5
            if height > max_height:
                height, width = max_height, max_height / aspect
            margin = min(page.rect.width, page.rect.height) * margin_ratio
            page.insert_image(_box(page, width, height, position, margin, x_ratio, y_ratio), stream=logo,
                              keep_proportion=True, overlay=True)
        _save(document, destination)


def add_text(source: Path, destination: Path, text: str, *, position: str,
             size_ratio: float, opacity: float, margin_ratio: float,
             x_ratio: float | None = None, y_ratio: float | None = None,
             font: str = "cjk", color: tuple[float, float, float] = (.12, .12, .16),
             background: tuple[float, float, float] | None = None) -> None:
    text = " ".join(text.split())
    if not text or len(text) > 120:
        raise ValueError("文字需为 1–120 个字符")
    with fitz.open(source) as document:
        for page in document:
            font_size = max(8, min(72, page.rect.width * size_ratio))
            # 中文强制使用 MuPDF 内置中文字体；英文才应用所选字形，避免中文变方框。
            fontname = "china-s" if any(ord(char) > 127 for char in text) else FONTS.get(font, "china-s")
            raw_width = fitz.get_text_length(text, fontname=fontname, fontsize=font_size)
            max_text_width = page.rect.width * .8
            if raw_width > max_text_width:
                font_size = max(8, font_size * max_text_width / raw_width)
            padding = font_size * .45
            text_width = min(page.rect.width * .8, max(font_size * 2,
                             fitz.get_text_length(text, fontname=fontname, fontsize=font_size) * 1.08))
            width = min(page.rect.width * .9, text_width + padding * 2)
            # Base14 衬线 / 等宽字体的行框比中文字体更高，预留足够高度，
            # 否则 insert_textbox 会因差几个点而整段不写入。
            height = font_size * 2.2 + padding
            margin = min(page.rect.width, page.rect.height) * margin_ratio
            rect = _box(page, width, height, position, margin, x_ratio, y_ratio)
            if background is not None:
                page.draw_rect(rect, color=None, fill=background, fill_opacity=opacity, overlay=True)
            text_rect = fitz.Rect(rect.x0 + padding, rect.y0 + padding * .2,
                                  rect.x1 - padding, rect.y1 - padding * .1)
            page.insert_textbox(text_rect, text, fontname=fontname, fontsize=font_size,
                                color=color, align=fitz.TEXT_ALIGN_CENTER,
                                fill_opacity=opacity, overlay=True)
        _save(document, destination)
