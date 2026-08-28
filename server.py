#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NotebookLM 去水印 · 网页版后端。

浏览器只负责交互，算法（unwatermark.clean_pdf）跑在本机。
服务只监听 127.0.0.1，PDF 不出这台电脑。

上传走 application/octet-stream 而不是表单，省掉 python-multipart 这个依赖。

启动：python3 server.py            （或双击 启动去水印网页.command）
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
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
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
JOB_TTL = timedelta(hours=6)


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


def _sweep_expired() -> None:
    """过期任务连同临时文件一起清掉，别让临时目录无限长大。"""
    deadline = datetime.now() - JOB_TTL
    with JOBS_LOCK:
        stale = [job for job in JOBS.values() if job.created < deadline and job.status != "running"]
        for job in stale:
            JOBS.pop(job.id, None)
    for job in stale:
        shutil.rmtree(job.directory, ignore_errors=True)


def _run_job(job: Job, options: CleanOptions) -> None:
    def progress(current: int, total: int, note: str) -> None:
        job.current, job.total, job.stage = current, total, note

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


def create_app() -> FastAPI:
    app = FastAPI(title="NotebookLM 去水印", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/api/jobs")
    async def create_job(request: Request) -> JSONResponse:
        _sweep_expired()
        name = unquote(request.headers.get("x-filename", "") or "").strip() or "document.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="只接受 PDF 文件")
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="没有收到文件内容")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 512 MB")
        if not payload.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="这不是一个 PDF 文件")

        job_id = uuid.uuid4().hex[:12]
        directory = WORK_ROOT / job_id
        directory.mkdir(parents=True, exist_ok=True)
        stem = Path(name).stem
        source = directory / f"{stem}.pdf"
        source.write_bytes(payload)
        job = Job(job_id, name, source, directory / f"{stem}_已去水印.pdf")

        options = CleanOptions(
            corner_w=float(request.query_params.get("corner_w", 0.30)),
            corner_h=float(request.query_params.get("corner_h", 0.12)),
        )
        with JOBS_LOCK:
            JOBS[job_id] = job
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

    parser = argparse.ArgumentParser(description="NotebookLM 去水印网页版（仅本机可访问）")
    parser.add_argument("--port", type=int, default=8823)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"NotebookLM 去水印  →  {url}\n按 Control-C 退出。PDF 全程只在本机处理。")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
