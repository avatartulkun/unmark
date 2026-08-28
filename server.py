#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NotebookLM 去水印 · 网页版后端。

浏览器只负责交互，算法（unwatermark.clean_pdf）跑在本机。
服务只监听 127.0.0.1，PDF 不出这台电脑。

上传走 application/octet-stream 而不是表单，省掉 python-multipart 这个依赖。

启动：python3 server.py            （或双击 启动去水印网页.command）
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import fitz
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image

from unwatermark import CleanOptions, clean_pdf

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_ROOT = Path(tempfile.gettempdir()) / "NotebookLMUnwatermark"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# 公网部署时这些值全部可以用环境变量压低；本机自用保持默认即可。
MAX_UPLOAD_BYTES = _env_int("UNMARK_MAX_UPLOAD_MB", 60) * 1024 * 1024
"""单个上传的大小上限。整页位图的 PDF 很占地方，60 MB 大约是两三百页。"""

MAX_CONCURRENT_JOBS = _env_int("UNMARK_MAX_CONCURRENT", 2)
"""同时真正在跑的任务数。超出的排队等待——限制的是内存峰值，不是请求数。"""

MAX_LIVE_JOBS = _env_int("UNMARK_MAX_LIVE_JOBS", 24)
"""内存里同时保留的任务数（含已完成待下载的）。每个任务在磁盘上占两份 PDF。"""

RATE_LIMIT_JOBS = _env_int("UNMARK_RATE_LIMIT", 12)
"""同一来源在时间窗内最多能发起多少次任务。"""

RATE_LIMIT_WINDOW = timedelta(minutes=_env_int("UNMARK_RATE_WINDOW_MIN", 60))

JOB_TTL = timedelta(minutes=_env_int("UNMARK_JOB_TTL_MIN", 120))
"""任务连同临时文件的保留时长；到点清掉。"""

TRUST_PROXY = os.environ.get("UNMARK_TRUST_PROXY") == "1"
"""放在 Cloudflare / 负载均衡后面时置 1，才按 X-Forwarded-For 认来源 IP。"""

RUN_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


def render_page(path: Path, page_index: int, scale: float = 1.4) -> Image.Image:
    """把 PDF 的某一页渲染成图片，只读，不改文件。"""
    with fitz.open(path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ValueError("页面编号超出范围。")
        pixmap = document[page_index].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


@dataclass
class Job:
    """一次去水印任务的全部状态；只活在内存里，进程退出即消失。"""

    id: str
    name: str
    source: Path
    destination: Path
    status: str = "running"           # running / done / failed / cancelled
    stage: str = "正在读取 PDF…"
    current: int = 0
    total: int = 0
    result: Optional[dict] = None
    error: str = ""
    created: datetime = field(default_factory=datetime.now)
    cancelled: bool = False

    @property
    def directory(self) -> Path:
        return self.source.parent

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

RATE_HITS: dict[str, deque] = {}
RATE_LOCK = threading.Lock()


def _client_key(request: Request) -> str:
    """限流用的来源标识。只有明确声明在反向代理后面时才信任转发头。"""
    if TRUST_PROXY:
        forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> None:
    """滑动窗口限流；顺手把过期的来源整条丢掉，字典不会无限长大。"""
    now = time.monotonic()
    window = RATE_LIMIT_WINDOW.total_seconds()
    with RATE_LOCK:
        for stale_key in [k for k, hits in RATE_HITS.items() if not hits or now - hits[-1] > window]:
            if stale_key != key:
                RATE_HITS.pop(stale_key, None)
        hits = RATE_HITS.setdefault(key, deque())
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_JOBS:
            wait = int((window - (now - hits[0])) / 60) + 1
            raise HTTPException(
                status_code=429,
                detail=f"提交太频繁了，请约 {wait} 分钟后再试",
            )
        hits.append(now)


def _sweep_expired() -> None:
    """过期任务连同临时文件一起清掉，别让临时目录无限长大。"""
    deadline = datetime.now() - JOB_TTL
    with JOBS_LOCK:
        stale = [job for job in JOBS.values() if job.created < deadline and job.status != "running"]
        for job in stale:
            JOBS.pop(job.id, None)
    for job in stale:
        shutil.rmtree(job.directory, ignore_errors=True)


def _evict_overflow() -> None:
    """任务数超过上限时，从最旧的已结束任务开始丢，保护磁盘。"""
    with JOBS_LOCK:
        if len(JOBS) <= MAX_LIVE_JOBS:
            return
        finished = sorted(
            (job for job in JOBS.values() if job.status != "running"),
            key=lambda job: job.created,
        )
        dropped = finished[: len(JOBS) - MAX_LIVE_JOBS]
        for job in dropped:
            JOBS.pop(job.id, None)
    for job in dropped:
        shutil.rmtree(job.directory, ignore_errors=True)


def _run_job(job: Job, options: CleanOptions) -> None:
    def progress(current: int, total: int, note: str) -> None:
        job.current, job.total, job.stage = current, total, note

    # 并发闸门：同时只让 MAX_CONCURRENT_JOBS 个任务真正开跑，其余在这里等。
    # 限的是内存峰值——每个任务解码整页位图，一起跑很容易把小内存实例打爆。
    if not RUN_SLOTS.acquire(blocking=False):
        job.stage = "排队中，等前面的任务跑完…"
        RUN_SLOTS.acquire()
    if job.cancelled:
        RUN_SLOTS.release()
        job.status, job.stage = "cancelled", "已取消"
        return
    try:
        result = clean_pdf(job.source, job.destination, options, progress, lambda: job.cancelled)
        job.result = result.as_dict()
        if result.cancelled:
            job.status, job.stage = "cancelled", "已取消"
        elif result.success:
            job.status, job.stage = "done", "完成"
        else:
            job.status, job.stage = "failed", "未处理"
            job.error = result.message
    except Exception as exc:  # pragma: no cover - 兜底，避免线程静默死掉
        job.status, job.stage, job.error = "failed", "出错", str(exc)
    finally:
        RUN_SLOTS.release()


def _get_job(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return job


def _png(image: Image.Image, box_ratio: Optional[list] = None, pad: float = 1.2) -> Response:
    """按需把图裁到水印附近再输出，让「处理前/后」的差别一眼可见。"""
    if box_ratio:
        x0, y0, x1, y1 = box_ratio
        width, height = image.width, image.height
        cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
        half_w = max(60.0, (x1 - x0) * width * pad)
        half_h = max(24.0, (y1 - y0) * height * pad * 3)
        image = image.crop((
            max(0, int(cx - half_w)), max(0, int(cy - half_h)),
            min(width, int(cx + half_w)), min(height, int(cy + half_h)),
        ))
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return Response(buffer.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


def _safe_stem(name: str) -> str:
    """文件名来自客户端请求头，只拿它的主干，且只保留能安全落盘的字符。"""
    stem = Path(name.replace("\\", "/")).stem
    stem = "".join(ch for ch in stem if ch.isprintable() and ch not in '/\\:*?"<>|')
    stem = stem.strip(" .")[:80]
    return stem or "document"


def _ratio_param(request: Request, key: str, default: float) -> float:
    """检测范围来自查询串。非法值退回默认，合法值也夹在合理区间里。

    夹紧不只是防崩：corner_w=1 意味着拿整页去做跨页交集，CPU 和内存都会翻好几倍。
    """
    raw = request.query_params.get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value != value:  # NaN
        return default
    return min(max(value, 0.05), 0.60)


async def _read_capped_body(request: Request) -> bytes:
    """边收边数，超限立刻断开。

    不能先 await request.body() 再判断大小——那等于把任意大的请求体先收进内存，
    公网上足够拿来打垮进程。Content-Length 只是快速拒绝，不能当真，所以流式那一路
    仍然逐块累计。
    """
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    too_big = HTTPException(status_code=413, detail=f"文件超过 {limit_mb} MB")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise too_big

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_UPLOAD_BYTES:
            raise too_big
        chunks.append(chunk)
    return b"".join(chunks)


def create_app() -> FastAPI:
    app = FastAPI(title="NotebookLM 去水印", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """给部署平台探活用。"""
        with JOBS_LOCK:
            running = sum(1 for job in JOBS.values() if job.status == "running")
        return JSONResponse({"ok": True, "jobs": len(JOBS), "running": running})

    @app.post("/api/jobs")
    async def create_job(request: Request) -> JSONResponse:
        _sweep_expired()
        _check_rate_limit(_client_key(request))
        name = unquote(request.headers.get("x-filename", "") or "").strip() or "document.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="只接受 PDF 文件")
        payload = await _read_capped_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="没有收到文件内容")
        if not payload.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="这不是一个 PDF 文件")

        job_id = uuid.uuid4().hex[:12]
        directory = WORK_ROOT / job_id
        directory.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(name)
        source = directory / f"{stem}.pdf"
        source.write_bytes(payload)
        job = Job(job_id, name, source, directory / f"{stem}_已去水印.pdf")

        options = CleanOptions(
            corner_w=_ratio_param(request, "corner_w", 0.30),
            corner_h=_ratio_param(request, "corner_h", 0.12),
        )
        with JOBS_LOCK:
            JOBS[job_id] = job
        _evict_overflow()
        threading.Thread(target=_run_job, args=(job, options), daemon=True).start()
        return JSONResponse({"id": job_id})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        return JSONResponse(_get_job(job_id).snapshot())

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        job.cancelled = True
        return JSONResponse({"ok": True})

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str, page: int = 1, side: str = "after", zoom: int = 0) -> Response:
        job = _get_job(job_id)
        path = job.source if side == "before" else job.destination
        if not path.exists():
            raise HTTPException(status_code=404, detail="该版本尚未生成")
        try:
            image = render_page(path, page - 1)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        box_ratio = (job.result or {}).get("box_ratio") if zoom else None
        return _png(image, box_ratio)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        job = _get_job(job_id)
        if job.status != "done" or not job.destination.exists():
            raise HTTPException(status_code=409, detail="任务尚未成功完成")
        return FileResponse(job.destination, media_type="application/pdf",
                            filename=job.destination.name)

    @app.delete("/api/jobs/{job_id}")
    def drop(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        shutil.rmtree(job.directory, ignore_errors=True)
        return JSONResponse({"ok": True})

    return app


app = create_app()


def main() -> None:
    import argparse
    import webbrowser

    import uvicorn

    parser = argparse.ArgumentParser(description="NotebookLM 去水印网页版")
    parser.add_argument("--port", type=int, default=_env_int("PORT", 8823))
    parser.add_argument("--host", default=os.environ.get("UNMARK_HOST", "127.0.0.1"),
                        help="监听地址。默认只有本机能访问；公网部署用 0.0.0.0")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    local_only = args.host in ("127.0.0.1", "localhost", "::1")
    url = f"http://127.0.0.1:{args.port}/"
    if local_only:
        print(f"NotebookLM 去水印  →  {url}\n按 Control-C 退出。PDF 全程只在本机处理。")
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    else:
        print(
            f"NotebookLM 去水印  →  对外监听 {args.host}:{args.port}\n"
            f"上限 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB／并发 {MAX_CONCURRENT_JOBS}／"
            f"限流 {RATE_LIMIT_JOBS} 次每 {int(RATE_LIMIT_WINDOW.total_seconds() // 60)} 分钟／"
            f"任务保留 {int(JOB_TTL.total_seconds() // 60)} 分钟"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
