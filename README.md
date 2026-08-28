# NotebookLM 去水印

去掉整页位图型 PDF 右下角的固定角标水印（NotebookLM 导出等）。浏览器界面，全程离线。

## 用法

双击 **启动去水印网页.command**，浏览器会自动打开 `http://127.0.0.1:8823/`。
把 PDF 拖进页面 → 等进度跑完 → 看「处理前 / 处理后」的放大对比 → 下载。

首次使用需要装依赖：

```bash
python3 -m pip install -r requirements.txt
```

也可以用命令行启动，`--port` 换端口，`--no-browser` 不自动开浏览器：

```bash
python3 server.py --port 8823
```

## 它是怎么找水印的

水印的定义就是「内容在变、它不变」的那部分像素。所以不靠内置模板图，而是把若干页的
右下角叠起来求交集——在几乎每一页的同一坐标都成立的，才算水印。换字体、换版本、
甚至换成别家的角标，都还能认出来。

两路投票，先暗后浅：

1. **暗像素跨页交集** —— 深色角标（NotebookLM 属于这类）
2. **局部反差跨页交集** —— 第一路没结果时才启用，能抓浅色/半透明角标。
   判据是「和自己周围不一样」，所以恒定的白页边不会被误当成水印

命中区域超过整页 0.5%，或者包围盒不像一行角标（太小、长宽比不对），一律判为误检并中止，
宁可不处理也不乱涂。

## 安全保证

- **原始 PDF 永不改写**，结果写到一个新文件
- 每页修复后强制还原 Mask 外像素并断言；中间图用 PNG 无损编码，并做一次编解码往返核对
- 写出后重新打开，抽页逐像素复核 Mask 之外确实没变
- 有文字层的页会跳过——那种水印多半是可直接删除的文本对象，不该用涂像素的办法处理
- 服务只监听 `127.0.0.1`，PDF 不出这台电脑

## 部署成公开服务

默认只监听 `127.0.0.1`。要放到公网上给别人用，把监听地址改掉即可：

```bash
python3 server.py --host 0.0.0.0
# 或者用镜像
docker build -t unmark . && docker run -p 8823:8823 unmark
```

仓库里附了 `Dockerfile` 和 `fly.toml`（Fly.io）。**注意这时 PDF 会上传到服务器**，
本机版那条「文件不出这台电脑」的保证不再成立，界面上要如实说明。

各项上限都走环境变量，默认值针对小内存实例调过：

| 变量 | 默认 | 作用 |
|---|---|---|
| `UNMARK_MAX_UPLOAD_MB` | `60` | 单个文件大小上限 |
| `UNMARK_MAX_CONCURRENT` | `2` | 同时真正在跑的任务数，直接决定内存峰值 |
| `UNMARK_MAX_LIVE_JOBS` | `24` | 内存中保留的任务数，超出从最旧的已结束任务开始丢 |
| `UNMARK_RATE_LIMIT` | `12` | 每个来源在窗口内可发起的任务数 |
| `UNMARK_RATE_WINDOW_MIN` | `60` | 限流窗口（分钟） |
| `UNMARK_JOB_TTL_MIN` | `120` | 任务连同临时文件的保留时长 |
| `UNMARK_TRUST_PROXY` | 未设 | 置 `1` 才按 `CF-Connecting-IP` / `X-Forwarded-For` 认来源 |
| `UNMARK_HOST` | `127.0.0.1` | 监听地址 |

上传是边收边计数的，超限立刻断开——不会先把请求体整个读进内存再判断。
探活接口是 `GET /healthz`。

## 文件

| 文件 | 作用 |
|---|---|
| `unwatermark.py` | 检测与修复算法，可单独 import 使用 |
| `server.py` | 网页后端（FastAPI） |
| `static/index.html` | 页面 |
| `site/index.html` | 介绍页（unmark.tinylabpro.com） |
| `启动去水印网页.command` | 双击启动 |
| `Dockerfile` / `fly.toml` | 公开部署用 |
| `tests/` | 算法层 + 网页层测试 |

当脚本用：

```python
from unwatermark import clean_pdf
result = clean_pdf("输入.pdf", "输出.pdf")
print(result.success, result.message)
```

## 测试

```bash
python3 -m pytest -q
```

## 已知情况

输出 PDF 通常比原件大（例：15.9 MB → 19.3 MB）。因为中间图用 PNG 无损重编码，
而 NotebookLM 导出的原件是 JPEG。要的是像素不失真，代价是体积。
