from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from decorate_pdf import add_logo, add_text


def _pdf(path: Path) -> None:
    document = fitz.open()
    for _ in range(2):
        page = document.new_page(width=600, height=800)
        page.draw_rect(page.rect, color=(1, 1, 1), fill=(1, 1, 1))
    document.save(path)


def _logo() -> bytes:
    image = Image.new("RGBA", (240, 100), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 239, 99), fill=(220, 30, 50, 255))
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_add_logo_to_every_page(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "logo.pdf"
    _pdf(source)
    add_logo(source, output, _logo(), position="bottom-right", width_ratio=.2,
             opacity=.6, margin_ratio=.03)
    with fitz.open(output) as document:
        assert document.page_count == 2
        assert all(page.get_images() for page in document)


def test_add_chinese_text_to_every_page(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pdf", tmp_path / "text.pdf"
    _pdf(source)
    add_text(source, output, "小产品实验室 Tiny Lab", position="center", size_ratio=.04,
             opacity=.5, margin_ratio=.03)
    with fitz.open(output) as document:
        assert document.page_count == 2
        assert all("小产品实验室" in page.get_text() for page in document)
