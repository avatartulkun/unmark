from __future__ import annotations

"""给已清理的 PDF 叠加自定义 Logo 或文字，不改动原结果。"""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageOps


POSITIONS = {"top-left", "top-right", "bottom-left", "bottom-right", "center"}


def _box(page: fitz.Page, width: float, height: float, position: str, margin: float) -> fitz.Rect:
    if position not in POSITIONS:
        raise ValueError("位置参数无效")
    pw, ph = page.rect.width, page.rect.height
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
             width_ratio: float, opacity: float, margin_ratio: float) -> None:
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
            page.insert_image(_box(page, width, height, position, margin), stream=logo,
                              keep_proportion=True, overlay=True)
        _save(document, destination)


def add_text(source: Path, destination: Path, text: str, *, position: str,
             size_ratio: float, opacity: float, margin_ratio: float) -> None:
    text = " ".join(text.split())
    if not text or len(text) > 120:
        raise ValueError("文字需为 1–120 个字符")
    with fitz.open(source) as document:
        for page in document:
            font_size = max(8, min(72, page.rect.width * size_ratio))
            # china-s 是 MuPDF 内置中文字体，同时覆盖英文；无需依赖服务器系统字体。
            text_width = min(page.rect.width * .8, max(font_size * 2, fitz.get_text_length(text, fontname="china-s", fontsize=font_size)))
            height = font_size * 1.65
            margin = min(page.rect.width, page.rect.height) * margin_ratio
            rect = _box(page, text_width, height, position, margin)
            page.insert_textbox(rect, text, fontname="china-s", fontsize=font_size,
                                color=(.12, .12, .16), align=fitz.TEXT_ALIGN_CENTER,
                                fill_opacity=opacity, overlay=True)
        _save(document, destination)
