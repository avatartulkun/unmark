# 公网部署用的镜像。本机自用不需要它——双击 启动去水印网页.command 即可。
FROM python:3.12-slim

# opencv 用的是 headless 轮子，所以不需要 libGL 那一套系统库。
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY unwatermark.py server.py ./
COPY static ./static

# 非 root 运行；临时目录要可写（任务文件都落在 TMPDIR 下）。
RUN useradd --create-home --uid 10001 unmark \
    && mkdir -p /tmp/NotebookLMUnwatermark \
    && chown -R unmark:unmark /app /tmp/NotebookLMUnwatermark
USER unmark

# 对外监听；具体上限交给环境变量，见 server.py 顶部。
# PUBLIC_MODE 让页面如实说明「文件会上传到服务器」，别挂着本机版的隐私说法。
ENV UNMARK_HOST=0.0.0.0 \
    UNMARK_PUBLIC_MODE=1 \
    PORT=8823

EXPOSE 8823

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8823')+'/healthz',timeout=4)"

CMD ["python", "server.py"]
