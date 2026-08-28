"""算法层 + 网页层的自包含测试。

跑法：cd 到本工具目录，执行  python3 -m pytest -q
"""

import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw

from unwatermark import CleanOptions, auto_clean_pdf, clean_pdf

PAGE_WIDTH, PAGE_HEIGHT = 400, 300
BADGE = (320, 276, 380, 284)  # 固定在右下角的假角标：60×8 px，占整页 0.4%


def _build_deck(path: Path, page_count: int = 6, badge: bool = True) -> None:
    """造一本「整页就是一张图」的 PDF：内容逐页在变，角标每页都在同一位置。"""
    document = fitz.open()
    for index in range(page_count):
        image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "#F2EFE6")
        draw = ImageDraw.Draw(image)
        offset = index * 23
        draw.rectangle((20 + offset, 40, 90 + offset, 150), fill=(30, 30, 30))
        draw.ellipse((150, 60 + offset % 40, 260, 170 + offset % 40), fill=(200, 90, 40))
        if badge:
            draw.rectangle(BADGE, fill=(20, 20, 20))
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        buffer = path.parent / f"_page_{index}.png"
        image.save(buffer, "PNG")
        page.insert_image(page.rect, filename=str(buffer))
        buffer.unlink()
    document.save(path)
    document.close()


def _page_pixels(path: Path, index: int = 0) -> np.ndarray:
    with fitz.open(path) as document:
        pixmap = document[index].get_pixmap(alpha=False)
        return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3)


def _badge_area(image: np.ndarray) -> float:
    return image[BADGE[1]:BADGE[3], BADGE[0]:BADGE[2]].mean()


# ------------------------------------------------------------------ 算法层

def test_removes_badge_and_never_touches_the_source(tmp_path: Path) -> None:
    source, destination = tmp_path / "deck.pdf", tmp_path / "cleaned.pdf"
    _build_deck(source)
    original_bytes = source.read_bytes()

    result = clean_pdf(source, destination)

    assert result.success, result.message
    assert result.pages_processed == 6
    x0, y0, x1, y1 = result.box
    assert BADGE[0] - 3 <= x0 <= BADGE[0] + 3 and BADGE[2] - 3 <= x1 <= BADGE[2] + 3
    assert BADGE[1] - 3 <= y0 <= BADGE[1] + 3 and BADGE[3] - 3 <= y1 <= BADGE[3] + 3
    assert result.area_percent < 1.0

    before, after = _page_pixels(source), _page_pixels(destination)
    assert _badge_area(before) < 60 and _badge_area(after) > 150
    # 角标之外的正文逐位不变；原始 PDF 一个字节都没动。
    assert np.array_equal(before[:250], after[:250])
    assert source.read_bytes() == original_bytes


def test_progress_and_cancel(tmp_path: Path) -> None:
    source = tmp_path / "deck.pdf"
    _build_deck(source)

    seen: list[tuple] = []
    assert clean_pdf(source, tmp_path / "a.pdf", progress=lambda *a: seen.append(a)).success
    assert seen and all(current <= total for current, total, _ in seen)

    cancelled = clean_pdf(source, tmp_path / "b.pdf", should_cancel=lambda: True)
    assert cancelled.cancelled and not cancelled.success
    assert not (tmp_path / "b.pdf").exists()


def test_refuses_when_nothing_is_consistently_marked(tmp_path: Path) -> None:
    source, destination = tmp_path / "clean.pdf", tmp_path / "out.pdf"
    _build_deck(source, badge=False)

    result = clean_pdf(source, destination)

    assert not result.success and "未检测到水印" in result.message
    assert not destination.exists()


def test_manual_box_and_dict_wrapper(tmp_path: Path) -> None:
    source, destination = tmp_path / "deck.pdf", tmp_path / "cleaned.pdf"
    _build_deck(source)

    payload = auto_clean_pdf(str(source), str(destination), box="318,274,382,286")

    assert payload["success"], payload["message"]
    assert payload["box"] == (318, 274, 382, 286)
    assert _badge_area(_page_pixels(destination)) > 150


def test_pages_with_a_text_layer_are_skipped(tmp_path: Path) -> None:
    """有文字层说明水印可能是可直接删除的文本对象，不该用涂像素的办法处理。"""
    source = tmp_path / "text.pdf"
    _build_deck(source, page_count=4)
    document = fitz.open(source)
    for page in document:
        page.insert_text((40, 90), "Story body", fontsize=14)
    document.saveIncr()
    document.close()

    result = clean_pdf(source, tmp_path / "out.pdf", CleanOptions())

    assert not result.success
    assert len(result.skipped) == 4 and "文字层" in result.skipped[0].reason


def test_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "deck.pdf"
    _build_deck(source)
    with pytest.raises(ValueError):
        clean_pdf(source, source)


# ------------------------------------------------------------------ 网页层

uvicorn = pytest.importorskip("uvicorn")
from server import create_app  # noqa: E402  （依赖检查之后再导入）


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    """起一个真的 uvicorn，测的就是用户实际访问的那条路径。"""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "网页服务没能启动"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read()


def _post_pdf(base_url: str, path: Path, name: str = "deck.pdf") -> str:
    request = urllib.request.Request(
        f"{base_url}/api/jobs", data=path.read_bytes(), method="POST",
        headers={"content-type": "application/octet-stream", "x-filename": name},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["id"]


def _wait(base_url: str, job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = json.loads(_get(f"{base_url}/api/jobs/{job_id}")[1])
        if state["status"] != "running":
            return state
        time.sleep(0.2)
    raise AssertionError("任务超时未完成")


def test_upload_preview_and_download(base_url, tmp_path: Path) -> None:
    source = tmp_path / "deck.pdf"
    _build_deck(source)

    state = _wait(base_url, _post_pdf(base_url, source))
    assert state["status"] == "done", state
    result = state["result"]
    assert result["success"] and result["pages_processed"] == 6
    assert result["box_ratio"] and len(result["box_ratio"]) == 4

    job_id = state["id"]
    # 放大预览：处理前应有深色角标，处理后应干净 —— 这是页面上唯一能让人一眼确认的证据。
    patches = {}
    for side in ("before", "after"):
        status, body = _get(f"{base_url}/api/jobs/{job_id}/preview?page=1&side={side}&zoom=1")
        assert status == 200
        patches[side] = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
    assert patches["before"].min() < 60
    assert patches["after"].min() > 150

    status, body = _get(f"{base_url}/api/jobs/{job_id}/download")
    assert status == 200 and body.startswith(b"%PDF")

    urllib.request.urlopen(
        urllib.request.Request(f"{base_url}/api/jobs/{job_id}", method="DELETE"), timeout=10
    ).close()
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base_url}/api/jobs/{job_id}")
    assert excinfo.value.code == 404


def test_pdf_without_watermark_is_reported_not_crashed(base_url, tmp_path: Path) -> None:
    source = tmp_path / "clean.pdf"
    _build_deck(source, badge=False)

    state = _wait(base_url, _post_pdf(base_url, source, "clean.pdf"))

    assert state["status"] == "failed" and "未检测到水印" in state["error"]
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base_url}/api/jobs/{state['id']}/download")
    assert excinfo.value.code == 409


def test_non_pdf_upload_is_rejected(base_url) -> None:
    request = urllib.request.Request(
        f"{base_url}/api/jobs", data=b"hello world", method="POST",
        headers={"content-type": "application/octet-stream", "x-filename": "fake.pdf"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 400


def test_index_page_is_served(base_url) -> None:
    status, body = _get(base_url + "/")
    assert status == 200
    assert "去水印" in body.decode("utf-8")
