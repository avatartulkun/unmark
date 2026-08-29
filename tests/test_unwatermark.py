"""算法层 + 网页层的自包含测试。

跑法：cd 到本工具目录，执行  python3 -m pytest -q
"""

import io
import json
import socket
import threading
import time
import types
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


# ------------------------------------------------------- 公网部署用的限制

import server as server_module  # noqa: E402


def _post_raw(base_url: str, data: bytes, name: str = "deck.pdf"):
    request = urllib.request.Request(
        f"{base_url}/api/jobs", data=data, method="POST",
        headers={"content-type": "application/octet-stream", "x-filename": name},
    )
    return urllib.request.urlopen(request, timeout=30)


def test_oversized_upload_is_rejected(base_url, monkeypatch) -> None:
    """超限的请求要被挡掉，而不是先整个收进内存再说。"""
    monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 2048)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_raw(base_url, b"%PDF-1.7\n" + b"0" * 5000)
    assert excinfo.value.code == 413
    assert "MB" in excinfo.value.read().decode("utf-8")


def test_rate_limit_blocks_flood(base_url, monkeypatch) -> None:
    monkeypatch.setattr(server_module, "RATE_LIMIT_JOBS", 3)
    with server_module.RATE_LOCK:
        server_module.RATE_HITS.clear()
    try:
        codes = []
        for _ in range(5):
            try:
                # 内容故意不合法：限流发生在解析之前，所以 400 也算「放行了」。
                _post_raw(base_url, b"not a pdf at all")
            except urllib.error.HTTPError as error:
                codes.append(error.code)
        assert codes[:3] == [400, 400, 400], codes
        assert codes[3:] == [429, 429], codes
    finally:
        with server_module.RATE_LOCK:
            server_module.RATE_HITS.clear()


def test_healthz_reports_liveness(base_url) -> None:
    status, body = _get(base_url + "/healthz")
    assert status == 200
    assert json.loads(body)["ok"] is True


@pytest.mark.parametrize("raw, expected", [
    ("../../etc/passwd.pdf", "passwd"),
    ("C:\\Users\\me\\秘密.pdf", "秘密"),
    ("....pdf", "document"),   # 纯点号的主干会被清空，回落到默认名
    ("", "document"),
    ("/", "document"),
    ("a" * 200 + ".pdf", "a" * 80),
])
def test_safe_stem_keeps_files_inside_the_job_directory(raw, expected) -> None:
    stem = server_module._safe_stem(raw)
    assert stem == expected
    assert "/" not in stem and "\\" not in stem


@pytest.mark.parametrize("raw, expected", [
    (None, 0.30),
    ("0.4", 0.40),
    ("垃圾", 0.30),
    ("nan", 0.30),
    ("1.0", 0.60),     # 夹到上限：整页扫描会让 CPU/内存翻好几倍
    ("-5", 0.05),
])
def test_ratio_param_is_clamped(raw, expected) -> None:
    params = {} if raw is None else {"corner_w": raw}
    request = types.SimpleNamespace(query_params=params)
    assert server_module._ratio_param(request, "corner_w", 0.30) == pytest.approx(expected)


def test_index_privacy_copy_matches_deployment_mode(monkeypatch) -> None:
    """本机版说「不上传」，公开版必须说清文件确实传到服务器——不能两边都挂着。"""
    monkeypatch.setattr(server_module, "PUBLIC_MODE", False)
    local = server_module._render_index()
    assert "不上传任何服务器" in local and "全程离线" in local
    assert "上传到服务器" not in local

    monkeypatch.setattr(server_module, "PUBLIC_MODE", True)
    public = server_module._render_index()
    assert "上传到服务器" in public
    assert "不上传任何服务器" not in public and "全程离线" not in public
    assert "自动删除" in public          # 说了传上去，就得说什么时候删
    assert "下载到本地运行" in public     # 给在意隐私的人一条退路

    for html in (local, public):
        assert "<!--SUB_NOTE-->" not in html and "<!--FOOT_NOTE-->" not in html


# ------------------------------------------------- 修复质量与颜色覆盖

def _solid_badge(bg, fg, box=(430, 150, 660, 182), size=(700, 200)):
    """一块纯色底 + 一个角标，走一遍 JPEG 贴近真实导出。"""
    W, H = size
    image = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(image)
    draw.rectangle([box[0] + 8, box[1] + 9, box[0] + 150, box[1] + 22], fill=fg)
    draw.ellipse([box[0] + 170, box[1] + 8, box[0] + 196, box[1] + 26], fill=fg)
    array = np.asarray(image)
    import cv2
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("bg, fg", [
    ((198, 198, 198), (60, 62, 70)),      # 深灰字 / 浅灰底，NotebookLM 属于这类
    ((252, 252, 252), (26, 90, 220)),     # 蓝字 / 白底
    ((200, 200, 200), (210, 45, 40)),     # 红字 / 浅灰底
])
def test_solid_backdrop_is_filled_exactly_not_inpainted(bg, fg) -> None:
    """纯色底上不该用 inpaint。

    TELEA 是从边界向内推测纹理的算法，在纯色区会拉出规则条纹——幅度只有几个灰阶，
    但因为有结构，人眼一眼就看得出「这块被涂过」。纯色底直接铺底色才是正解。
    """
    from unwatermark import _repair, box_to_mask
    box = (430, 150, 660, 182)
    rgb = _solid_badge(bg, fg)
    mask = box_to_mask((200, 700), box, dilate=2)
    inside = mask > 0

    filled, filled_mask = _repair(rgb, mask, radius=4)          # 默认启用纯色铺底
    inpainted, _ = _repair(rgb, mask, radius=4, ring=0)          # ring=0 强制走旧路径
    inside = filled_mask > 0

    truth = np.array(bg, float)
    filled_dev = np.abs(filled[inside].astype(float) - truth).max()
    inpaint_dev = np.abs(inpainted[inside].astype(float) - truth).max()

    assert filled_dev <= 2, f"铺底后仍偏离底色 {filled_dev}"
    assert filled_dev < inpaint_dev, "纯色底上铺底色没有比 inpaint 更干净"
    # 铺进去的必须是一个常量，不能有残留结构
    assert len(np.unique(filled[inside].reshape(-1, 3), axis=0)) == 1


def test_textured_backdrop_still_uses_inpaint() -> None:
    """底色有纹理时不能乱铺——那会糊掉背景，必须退回 inpaint。"""
    from unwatermark import _flat_backdrop, box_to_mask
    rng = np.random.default_rng(11)
    noisy = rng.integers(0, 255, (200, 700, 3), dtype=np.uint8)
    mask = box_to_mask((200, 700), (430, 150, 660, 182), dilate=2)
    assert _flat_backdrop(noisy, mask, ring=6, tolerance=4.0) is None


def _color_deck(path: Path, badge_rgb, pages: int = 6) -> None:
    """整页位图 + 固定角标；底色浅灰，正文逐页变化。"""
    import cv2
    W, H = 800, 1000
    document = fitz.open()
    rng = np.random.default_rng(5)
    for index in range(pages):
        array = np.full((H, W, 3), 200, np.uint8) - rng.integers(0, 4, (H, W, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        for row in range(14):
            y = 70 + row * 60
            draw.rectangle([60, y, 60 + 200 + (index * 43 + row * 71) % 380, y + 15],
                           fill=(70, 74, 88))
        draw.rectangle([W - 220, H - 64, W - 90, H - 46], fill=badge_rgb)
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
        page = document.new_page(width=W * 0.72, height=H * 0.72)
        page.insert_image(page.rect, stream=buffer.tobytes())
    document.save(path, deflate=True)
    document.close()


@pytest.mark.parametrize("name, badge", [
    ("深色字标", (55, 58, 66)),
    ("等亮度橙色", (255, 190, 120)),   # 灰度 201 ≈ 底色 200，亮度上等于隐形
    ("等亮度青色", (120, 215, 205)),
    ("比底更浅的灰", (232, 232, 232)),
    ("蓝紫色", (108, 99, 255)),
])
def test_badges_of_any_color_are_found(tmp_path: Path, name, badge) -> None:
    """角标不一定是黑的。等亮度的彩色角标躲得过前两路投票，必须有色度那一路兜住。"""
    source = tmp_path / "deck.pdf"
    _color_deck(source, badge)

    result = clean_pdf(source, tmp_path / "out.pdf")

    assert result.success, f"{name} 未被检出：{result.message}"
    x0, y0, x1, y1 = result.box
    assert 560 <= x0 <= 600 and 700 <= x1 <= 730, f"{name} 定位偏了：{result.box}"
    assert 920 <= y0 <= 945 and 945 <= y1 <= 970, f"{name} 定位偏了：{result.box}"


def test_grain_added_to_repair_is_monochrome() -> None:
    """补回的颗粒必须是亮度噪声。

    三个通道各自随机会产生彩色噪点，放大看是一片红绿斑，比原来那块光滑区域更扎眼——
    这是实测踩过的坑。纸纹本身是亮度上的颗粒，所以合成噪声也必须单色。
    """
    import cv2
    from unwatermark import _graft_texture, box_to_mask

    rng = np.random.default_rng(3)
    H, W = 120, 300
    base = np.full((H, W, 3), 150, np.uint8)
    grain = rng.normal(0, 9, (H, W, 1)).repeat(3, axis=2)      # 单色纸纹
    textured = np.clip(base + grain, 0, 255).astype(np.uint8)
    mask = box_to_mask((H, W), (110, 50, 200, 70), dilate=2)

    smooth = textured.copy()
    smooth[mask > 0] = 150                                      # 模拟 inpaint 抹平后的样子
    grained = _graft_texture(smooth, textured, mask)

    inside = mask > 0
    # 颗粒补回来了
    assert grained[inside].std() > smooth[inside].std() + 2
    # 且是单色的：同一像素三通道的偏差应当极小
    channel_spread = grained[inside].astype(float)
    assert np.abs(channel_spread - channel_spread.mean(axis=1, keepdims=True)).max() <= 1.5
    # Mask 外一个像素都不许动
    assert np.array_equal(grained[~inside], smooth[~inside])


def test_grain_is_skipped_on_smooth_surroundings() -> None:
    """周围本来就平滑时不该硬加噪声——那是画蛇添足。"""
    from unwatermark import _graft_texture, box_to_mask
    flat = np.full((120, 300, 3), 200, np.uint8)
    mask = box_to_mask((120, 300), (110, 50, 200, 70), dilate=2)
    assert np.array_equal(_graft_texture(flat.copy(), flat, mask), flat)


def test_mask_follows_the_glyphs_not_the_bounding_block(tmp_path: Path) -> None:
    """涂改要贴着字形走，不能糊成一个方块。

    定位那几路用的高斯半径必须足够大才站得稳，代价是把整行字糊成一团，字母间隙
    也跟着超标——涂出来就是个方块，画面被抹掉的比水印本身大得多。中值滤波精修
    把它削回字形。

    但也不能削过头：只涂笔画核心会留下能读出字来的残影（抗锯齿那圈仍勾着字形），
    所以精修之后仍要小幅外扩。这两头都踩过，用覆盖率把区间钉住。
    """
    import cv2
    import unwatermark as U

    # 角标必须画成真实文字：实心方块本来就该 100% 覆盖，测不出精修有没有生效。
    source = tmp_path / "deck.pdf"
    W, H = 800, 1000
    document = fitz.open()
    rng = np.random.default_rng(5)
    for index in range(8):
        array = np.full((H, W, 3), 205, np.uint8) - rng.integers(0, 4, (H, W, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        for row in range(14):
            y = 70 + row * 60
            draw.rectangle([60, y, 60 + 200 + (index * 43 + row * 71) % 380, y + 15],
                           fill=(70, 74, 88))
        draw.text((W - 210, H - 62), "NotebookLM", fill=(40, 42, 50))
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 92])
        page = document.new_page(width=W * 0.72, height=H * 0.72)
        page.insert_image(page.rect, stream=buffer.tobytes())
    document.save(source, deflate=True)
    document.close()

    document = fitz.open(source)
    refs, _ = U.scan_full_bleed_pages(document)
    refs, _ = U.keep_dominant_size(refs)

    refined, _ = U.detect_watermark(document, refs, CleanOptions())
    coarse, _ = U.detect_watermark(document, refs, CleanOptions(ink_min_ratio=9.9))
    document.close()

    def coverage(detection) -> float:
        x0, y0, x1, y1 = detection.box
        return float((detection.mask[y0:y1, x0:x1] > 0).mean())

    assert coverage(refined) < coverage(coarse) - 0.05, "精修没有比粗掩膜更贴合字形"
    assert coverage(refined) > 0.15, "削得太狠会留下能读出字的残影"


def test_residue_sweep_grows_mask_and_reports_it() -> None:
    """自查扩大掩膜后必须把生效的那张传回来。

    否则调用方拿旧掩膜去复核，会把自己刚补涂的像素判成越界——这个坑踩过，
    表现是「已写出但复核未通过：Mask 之外有 N 个像素被改动」。
    """
    import cv2
    from unwatermark import _repair, box_to_mask

    rng = np.random.default_rng(4)
    H, W = 90, 260
    art = np.clip(np.full((H, W, 3), 60, np.int16)
                  + rng.normal(0, 5, (H, W, 3)), 0, 255).astype(np.uint8)
    marked = art.copy()
    cv2.putText(marked, "NotebookLM", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    tight = box_to_mask((H, W), (62, 40, 200, 60), dilate=0)
    repaired, effective = _repair(marked, tight, radius=4, sweeps=2)

    assert effective.sum() >= tight.sum(), "自查只会让掩膜变大或不变"
    assert effective.sum() <= tight.sum() * 4, "掩膜膨胀失控就会啃到画面"
    # 返回的掩膜必须真的能通过「掩膜外逐位不变」这条复核
    outside = effective == 0
    assert np.array_equal(repaired[outside], marked[outside])


def test_area_check_runs_after_the_mask_is_refined(tmp_path: Path) -> None:
    """面积上限要拿削好的字形去比，不能拿粗定位的团块。

    粗团块把字母间隙也算进去，虚胖一倍有余。纯深底配白字标时它会涨到 0.65%，
    超过 0.5% 的上限，于是一个完全合法的角标被判成「很可能是正文」而拒绝处理。
    削形之后同一个角标只有 0.35%。
    """
    import cv2
    source = tmp_path / "dark.pdf"
    W, H = 900, 620
    document = fitz.open()
    for index in range(8):
        art = np.full((H, W, 3), 38, np.uint8)
        for row in range(9):
            y = 60 + row * 52
            cv2.rectangle(art, (70, y), (70 + 180 + (index * 47 + row * 83) % 420, y + 14),
                          (150, 154, 168), -1)
        cv2.putText(art, "NotebookLM", (W - 240, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(art, cv2.COLOR_RGB2BGR))
        page = document.new_page(width=W * 0.72, height=H * 0.72)
        page.insert_image(page.rect, stream=buffer.tobytes())
    document.save(source, deflate=True)
    document.close()

    result = clean_pdf(source, tmp_path / "out.pdf")

    assert result.success, f"纯深底白字标被误拒：{result.message}"
    assert result.area_percent < 0.5, "削形之后仍然虚胖"


def test_grain_amplitude_ignores_bright_neighbours() -> None:
    """颗粒强度要用中位绝对偏差估，不能用标准差。

    取样圈常常蹭到装饰线、星光这类高对比元素。实测一份真实绘本里，圈内 |细节| 的
    中位数只有 2，标准差却被少数极端值拉到 16——照标准差注入噪声，修复区就是一片
    黑白麻点，比不修还难看。
    """
    import cv2
    from unwatermark import _graft_texture, box_to_mask

    rng = np.random.default_rng(7)
    H, W = 120, 320
    base = np.clip(np.full((H, W, 3), 40, np.float32)
                   + rng.normal(0, 2, (H, W, 1)), 0, 255).astype(np.uint8)
    # 紧贴掩膜上方画一条高亮装饰线，模拟真实版面
    cv2.line(base, (0, 44), (W, 44), (250, 246, 220), 2)

    mask = box_to_mask((H, W), (120, 56, 250, 74), dilate=1)
    smooth = base.copy()
    smooth[mask > 0] = np.array([40, 40, 40], np.uint8)

    grained = _graft_texture(smooth, base, mask)

    inside = mask > 0
    injected = grained[inside].astype(float).std()
    assert injected < 9, f"注入的噪声幅度 {injected:.1f} 过大，会变成黑白麻点"


def test_residue_sweep_never_leaves_the_detection_box() -> None:
    """自查不许越过检测框：框外一律是画面。

    实测踩过——一条奶油色装饰线只伸进框边几个像素，被当成「没擦干净的墨迹」吃掉之后，
    整条线看起来就断了。按颜色区分不可靠（那条线色度只有 29，比画面里别的东西还低），
    按几何边界区分才干净。
    """
    import cv2
    from unwatermark import _repair, box_to_mask

    H, W = 120, 320
    art = np.full((H, W, 3), 30, np.uint8)
    cv2.line(art, (0, 62), (W, 62), (237, 229, 206), 2)      # 贯穿全图的装饰线
    marked = art.copy()
    cv2.putText(marked, "Gemini", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    box = (140, 52, 240, 78)
    mask = box_to_mask((H, W), (150, 58, 230, 74), dilate=1)
    _, effective = _repair(marked, mask, radius=4, sweeps=2, bounds=box)

    ys, xs = np.nonzero(effective)
    assert xs.min() >= box[0] and xs.max() < box[2], "自查横向越界了"
    assert ys.min() >= box[1] and ys.max() < box[3], "自查纵向越界了"
    # 框外的装饰线必须一根汗毛都没动
    outside_line = art[62, :box[0]]
    assert np.array_equal(marked[62, :box[0]], outside_line)


def test_flat_backdrop_survives_a_bright_neighbour() -> None:
    """判断底色纯不纯要用中位数，别被取样圈里的亮线拉偏。

    真实绘本里角标旁边常有装饰线、星光。用均值判断时，少数极端值会把「离散度」
    抬高，一块本来平整的纯色底被误判成有纹理，白白掉进 inpaint——而纯色底
    直接铺底色本可以做到像素级完美。
    """
    import cv2
    from unwatermark import _flat_backdrop, _repair, box_to_mask

    rng = np.random.default_rng(12)
    H, W = 120, 320
    flat = np.clip(np.full((H, W, 3), (13, 36, 67), np.float32)
                   + rng.normal(0, 1.5, (H, W, 1)), 0, 255).astype(np.uint8)
    cv2.line(flat, (0, 44), (W, 44), (237, 229, 206), 2)      # 紧贴掩膜上方的亮线
    mask = box_to_mask((H, W), (120, 56, 250, 74), dilate=1)

    backdrop = _flat_backdrop(flat, mask, ring=6, tolerance=4.0)
    assert backdrop is not None, "有亮线在旁边就认不出纯色底了"
    assert abs(int(backdrop[2]) - 67) <= 3, f"解出的底色偏了：{backdrop}"

    # 真有大片异色时仍要拒绝，否则会把画面涂成一块死色
    half = flat.copy()
    half[:, :W // 2] = (200, 190, 120)
    assert _flat_backdrop(half, mask, ring=6, tolerance=4.0) is None


def test_smooth_backdrop_fits_a_gradient() -> None:
    """底色是平滑渐变时，用曲面外推，别交给 inpaint 去猜。

    渐变是可建模的：从掩膜外一圈拟合二次曲面，算出来的值比扩散平均值准得多。
    合成基准里「明暗渐变」场景因此从残留误差 0.20 降到 0.00。
    """
    from unwatermark import _smooth_backdrop, box_to_mask

    H, W = 140, 360
    ramp = np.linspace(20, 240, W, dtype=np.float32)
    art = np.repeat(np.repeat(ramp[None, :, None], H, 0), 3, 2).astype(np.uint8)
    mask = box_to_mask((H, W), (140, 60, 240, 84), dilate=1)

    surface = _smooth_backdrop(art, mask, ring=6, tolerance=3.0)
    assert surface is not None, "平滑渐变没被认出来"
    truth = art[mask > 0].astype(float)
    assert np.abs(surface.astype(float) - truth).max() <= 4, "曲面外推偏差过大"


def test_smooth_backdrop_refuses_busy_content() -> None:
    """高频画面不能硬拟合成一张平滑曲面——那比 inpaint 还差。

    验收线定得太松就会踩这个坑：实测阈值从 3.0 放到 9.0，高频场景的残留误差
    从 2.11 涨到 2.92。
    """
    from unwatermark import _smooth_backdrop, box_to_mask
    import cv2

    rng = np.random.default_rng(21)
    H, W = 140, 360
    art = np.full((H, W, 3), 150, np.uint8)
    for _ in range(120):
        cv2.circle(art, (int(rng.integers(0, W)), int(rng.integers(0, H))),
                   int(rng.integers(3, 18)), tuple(int(v) for v in rng.integers(30, 240, 3)), -1)
    mask = box_to_mask((H, W), (140, 60, 240, 84), dilate=1)

    assert _smooth_backdrop(art, mask, ring=6, tolerance=3.0) is None


def test_panel_removal_is_off_by_default() -> None:
    """衬底反解默认关闭——实测它会凭空造出暗色矩形，见 CleanOptions 里的说明。"""
    assert CleanOptions().remove_panel is False


def test_area_guard_counts_pixels_not_values(tmp_path: Path) -> None:
    """面积闸门要数像素个数，不能把掩膜里的 255 加起来。

    掩膜是 0/255 的 uint8，`mask.sum()` 得到的是值的总和。拿它去和「整页像素数」
    比，会算出「涂改了整页的 171%」这种荒谬结论，把正常任务误判成失控中止。
    """
    source = tmp_path / "deck.pdf"
    _build_deck(source)

    result = clean_pdf(source, tmp_path / "out.pdf")

    assert result.success, result.message
    assert 0 < result.area_percent < 100, f"面积算错了：{result.area_percent}"


# ---------------------------------------------------------------- 磨砂衬底

def _frosted_page(sigma: float = 4.0, lift: float = 6.0, busy: bool = False):
    """造一张「深底 + 一条横向亮线」的画面，再按磨砂衬底的做法盖一块板。

    磨砂 = 把板内的画面高斯模糊再整体抬一点亮，这是从真实导出件里量出来的模型
    （σ≈4、抬亮 5~6 个灰阶）。返回 (原画, 盖了板的图, 板的范围)。
    """
    import cv2

    H, W = 120, 400
    rng = np.random.default_rng(7)
    art = np.full((H, W, 3), 40, np.float32)
    if busy:                                  # 横向不连贯：剖面续接不该生效
        for _ in range(90):
            cv2.circle(art, (int(rng.integers(0, W)), int(rng.integers(0, H))),
                       int(rng.integers(4, 16)), tuple(float(v) for v in rng.integers(20, 220, 3)), -1)
    else:
        art[58:60] = 224.0                    # 一条贯穿整幅的横线
        art[60:62] = 150.0
    art += rng.normal(0, 1.5, (H, W, 1))
    art = np.clip(art, 0, 255)

    panel = (240, 50, W, 100)                 # x0, y0, x1, y1
    px0, py0, px1, py1 = panel
    blurred = cv2.GaussianBlur(art, (0, 0), sigmaX=0.01, sigmaY=sigma)
    marked = art.copy()
    marked[py0:py1, px0:px1] = np.clip(blurred[py0:py1, px0:px1] + lift, 0, 255)
    return art.astype(np.uint8), marked.astype(np.uint8), panel


def test_frost_panel_is_found_by_its_edges() -> None:
    """磨砂板要靠四条边的台阶找，不能靠「颗粒消失」。

    颗粒判据在真实页面上不可用：实测板外那圈平滑背景颗粒 1.06、板内 0.2~1.9，
    两者根本分不开。边界才是衬底的定义。
    """
    from unwatermark import _frost_panel

    _art, marked, panel = _frosted_page()
    found = _frost_panel(marked, (250, 70, 380, 88))
    assert found is not None, "合成的磨砂板没被认出来"
    for got, want in zip(found, panel):
        assert abs(got - want) <= 3, f"板的范围偏差过大：{found} vs {panel}"


def test_defrost_restores_the_line_across_the_panel() -> None:
    """板内那条被模糊掉的横线要按板外的亮度续回来。

    这是整条路的目的：模糊丢掉的高频找不回来，但结构可以从板外续进去。
    """
    from unwatermark import _defrost

    art, marked, panel = _frosted_page()
    mask = np.zeros(marked.shape[:2], np.uint8)      # 这里只验衬底，不掺角标
    result, touched = _defrost(marked, marked, mask, (250, 70, 380, 88),
                               tolerance=3.0, coherence=3.0, max_lift=24.0, max_area=0.5)
    assert result is not None, "磨砂模型没被接受"
    px0, py0, px1, py1 = panel
    line_truth = float(art[58:60, px0 + 20:px1 - 20].mean())
    line_before = float(marked[58:60, px0 + 20:px1 - 20].mean())
    line_after = float(result[58:60, px0 + 20:px1 - 20].mean())
    assert line_before < line_truth * 0.6, "样例没造对：线本该被模糊压掉"
    assert line_after > line_truth * 0.85, f"线只重建出 {line_after / line_truth:.0%}"

    flat_truth = float(art[py1 - 8:py1, px0 + 20:px1 - 20].mean())
    flat_after = float(result[py1 - 8:py1, px0 + 20:px1 - 20].mean())
    assert abs(flat_after - flat_truth) <= 3.0, "平坦处的整体抬亮没去干净"
    assert not touched[:, :px0 - 2].any(), "板外的像素被动了"


def test_defrost_refuses_horizontally_busy_art() -> None:
    """画面横向不连贯时必须整块放弃——硬续剖面会拉出条纹。"""
    from unwatermark import _defrost

    _art, marked, _panel = _frosted_page(busy=True)
    mask = np.zeros(marked.shape[:2], np.uint8)
    result, _touched = _defrost(marked, marked, mask, (250, 70, 380, 88),
                                tolerance=3.0, coherence=3.0, max_lift=24.0, max_area=0.5)
    assert result is None


def test_defrost_does_nothing_without_a_panel() -> None:
    """没有磨砂衬底的页面，一个像素都不许动。"""
    from unwatermark import _defrost

    art, _marked, _panel = _frosted_page()
    mask = np.zeros(art.shape[:2], np.uint8)
    result, _touched = _defrost(art, art, mask, (250, 70, 380, 88),
                                tolerance=3.0, coherence=3.0, max_lift=24.0, max_area=0.5)
    assert result is None


def test_defrost_adds_no_synthetic_grain() -> None:
    """板内不补颗粒。

    模糊确实连颗粒一起抹掉了，补回去在量化指标上更接近（1.98 vs 板外真值 1.68），
    但放大看是一片比原来那层雾还扎眼的团块——合成噪声是相关的，和纸纹不是一回事。
    板内的平滑本来就是原件里的样子，去掉染色、把结构续回来是还原，造颗粒不是。
    """
    import cv2
    from unwatermark import _defrost

    _art, marked, panel = _frosted_page()
    mask = np.zeros(marked.shape[:2], np.uint8)
    result, _touched = _defrost(marked, marked, mask, (250, 70, 380, 88),
                                tolerance=3.0, coherence=3.0, max_lift=24.0, max_area=0.5)
    assert result is not None
    px0, _py0, px1, py1 = panel
    patch = result[py1 - 12:py1, px0 + 20:px1 - 20].astype(np.float32)
    detail = patch - cv2.GaussianBlur(patch, (0, 0), 1.6)
    assert 1.4826 * float(np.median(np.abs(detail))) < 1.0, "板内被注入了合成颗粒"


def test_defrost_sees_the_image_with_overspill_already_restored(tmp_path: Path) -> None:
    """磨砂那步必须拿到「掩膜外已还原」的图。

    _fill_once 会波及掩膜之外（inpaint 的扩散边、补颗粒时按包围盒整块加噪声），
    这些本由 _repair 末尾统一还原。先前把未还原的图喂给磨砂那步，取样带和板内
    统计都被污染，实测真实样本的最后一页因此过不了模型闸门。
    """
    import unwatermark

    seen: list[bool] = []
    original = unwatermark._defrost

    def spy(repaired, rgb, mask, *args, **kwargs):
        outside = mask == 0
        seen.append(bool(np.array_equal(repaired[outside], rgb[outside])))
        return original(repaired, rgb, mask, *args, **kwargs)

    deck = tmp_path / "deck.pdf"
    _build_deck(deck)
    unwatermark._defrost = spy
    try:
        clean_pdf(deck, tmp_path / "out.pdf", CleanOptions())
    finally:
        unwatermark._defrost = original
    assert seen, "磨砂那步没被调用"
    assert all(seen), "磨砂那步拿到的图里，掩膜外的像素还带着填充的溢出"


def test_defrost_is_on_by_default() -> None:
    """和 remove_panel 相反，这条路是量出来的，默认开启。"""
    assert CleanOptions().defrost is True
