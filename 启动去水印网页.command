#!/bin/zsh
# NotebookLM 去水印 · 网页版
cd "$(dirname "$0")"
exec python3 -u server.py "$@"
