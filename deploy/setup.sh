#!/usr/bin/env bash
#
# unmark 一键部署（Debian / Ubuntu）
#
#   curl -fsSL https://raw.githubusercontent.com/avatartulkun/unmark/main/deploy/setup.sh | bash
#
# 或者先克隆再跑 bash deploy/setup.sh。脚本可以反复执行：已经装好的不会重装，
# 已经在跑的会重新构建并平滑替换。
#
# 需要一个 Cloudflare Tunnel 的 token，见下方提示或 README。

set -euo pipefail

REPO="https://github.com/avatartulkun/unmark.git"
DIR="${UNMARK_DIR:-$HOME/unmark}"

info()  { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()   { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 环境检查

info "检查环境"

[ "$(uname -s)" = "Linux" ] || die "这个脚本只支持 Linux。"

if ! command -v apt-get >/dev/null 2>&1; then
    die "没找到 apt-get。这个脚本针对 Debian / Ubuntu；其他发行版请手动装好 Docker 后直接跑 docker compose up -d --build。"
fi

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|aarch64|arm64) ok "架构 $ARCH（有现成的 Python 轮子，不用编译）" ;;
    *) die "架构 $ARCH 上 opencv / PyMuPDF 没有预编译轮子，装起来会非常痛苦。" ;;
esac

MEM_MB="$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)"
if   [ "$MEM_MB" -lt 1800 ]; then
    warn "内存只有 ${MEM_MB} MB。实测最轻的任务就要约 580 MB，大文件会被 OOM 杀掉。"
    warn "脚本会把并发压到 1、上传上限降到 30 MB，但仍然建议换台机器。"
    LOW_MEM=1
elif [ "$MEM_MB" -lt 3500 ]; then
    ok "内存 ${MEM_MB} MB —— 够用，并发保持 1"
    LOW_MEM=1
else
    ok "内存 ${MEM_MB} MB —— 宽裕，并发 2"
    LOW_MEM=0
fi

DISK_GB="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')"
if [ "${DISK_GB:-0}" -ge 10 ]; then
    ok "可用磁盘 ${DISK_GB} GB"
else
    warn "可用磁盘只有 ${DISK_GB} GB。任务文件会占地方，建议留 20 GB 以上。"
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "不是 root 且没有 sudo，装不了 Docker。"
    SUDO="sudo"
fi

# ---------------------------------------------------------------- 装 Docker

info "准备 Docker"

if command -v docker >/dev/null 2>&1; then
    ok "Docker 已安装（$(docker --version | cut -d, -f1)）"
else
    warn "未安装，开始装（用 Docker 官方脚本，几分钟）"
    curl -fsSL https://get.docker.com | $SUDO sh
    ok "Docker 装好了"
fi

$SUDO systemctl enable --now docker >/dev/null 2>&1 || true

# 用数组存命令：compose 可能是 "docker compose"（两个词）也可能是 "docker-compose"，
# 靠不加引号来分词太脆，数组是唯一稳妥的写法。
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
else
    warn "没有 compose 插件，安装 docker-compose-plugin"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq docker-compose-plugin
    COMPOSE_CMD=(docker compose)
fi
COMPOSE_SHOWN="${COMPOSE_CMD[*]}"
ok "编排工具：$COMPOSE_SHOWN"

# 非 root 用户加进 docker 组，省掉每条命令都要 sudo。
if [ -n "$SUDO" ] && ! id -nG "$USER" | grep -qw docker; then
    $SUDO usermod -aG docker "$USER"
    warn "已把 $USER 加入 docker 组，但本次会话还没生效，脚本内继续用 sudo。"
    COMPOSE_CMD=("$SUDO" "${COMPOSE_CMD[@]}")
fi

# ---------------------------------------------------------------- 取代码

info "取代码"

if command -v git >/dev/null 2>&1; then :; else
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git
fi

if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only && ok "已更新到最新 ($(git -C "$DIR" rev-parse --short HEAD))"
else
    git clone --depth 1 "$REPO" "$DIR" && ok "已克隆到 $DIR"
fi
cd "$DIR"

# ---------------------------------------------------------------- 隧道 token

info "Cloudflare Tunnel"

if [ -f .env ] && grep -q '^TUNNEL_TOKEN=.\+' .env; then
    ok "已有 .env，沿用其中的 token"
else
    if [ -n "${TUNNEL_TOKEN:-}" ]; then
        printf 'TUNNEL_TOKEN=%s\n' "$TUNNEL_TOKEN" > .env
        ok "已从环境变量写入 .env"
    else
        cat <<'HOWTO'

  还没有隧道 token。去拿一个（两分钟）：

    1. 打开 https://one.dash.cloudflare.com
    2. Networks → Tunnels → Create a tunnel → 选 Cloudflared
    3. 随便起个名字（比如 unmark），Save
    4. 在 "Install and run a connector" 那一步，找到形如
         cloudflared service install eyJhIjoi....
       复制 eyJ 开头那一长串，就是 token
    5. 先不要按它给的命令装，直接回来这里

HOWTO
        printf '  把 token 粘在这里（不会回显）: '
        read -rs TOKEN_INPUT
        echo
        [ -n "$TOKEN_INPUT" ] || die "没有输入 token，中止。"
        printf 'TUNNEL_TOKEN=%s\n' "$TOKEN_INPUT" > .env
        ok "已写入 .env"
    fi
fi
chmod 600 .env

# ---------------------------------------------------------------- 低内存调整

if [ "$LOW_MEM" -eq 1 ]; then
    info "低内存机器，写一份覆盖配置"
    cat > docker-compose.override.yml <<'OVERRIDE'
# 由 deploy/setup.sh 按本机内存自动生成。想改就直接编辑，脚本不会覆盖已存在的文件。
services:
  app:
    environment:
      UNMARK_MAX_CONCURRENT: "1"    # 同时只跑一个任务
      UNMARK_MAX_UPLOAD_MB: "30"    # 大文件在小内存机器上会被 OOM 杀掉
      UNMARK_MAX_LIVE_JOBS: "8"
    deploy:
      resources:
        limits:
          memory: 1500m
OVERRIDE
    ok "已生成 docker-compose.override.yml（并发 1、上限 30 MB）"
fi

# ---------------------------------------------------------------- 起服务

info "构建并启动（第一次要装依赖，可能几分钟）"

"${COMPOSE_CMD[@]}" up -d --build

info "等待健康检查"

for i in $(seq 1 40); do
    STATE="$("${COMPOSE_CMD[@]}" ps --format json app 2>/dev/null | tr -d '\n' || true)"
    if printf '%s' "$STATE" | grep -q '"Health":"healthy"'; then
        ok "应用已就绪"
        break
    fi
    if printf '%s' "$STATE" | grep -q '"State":"exited"'; then
        "${COMPOSE_CMD[@]}" logs --tail 40 app
        die "应用容器退出了，日志见上。"
    fi
    if [ "$i" -eq 40 ]; then
        "${COMPOSE_CMD[@]}" logs --tail 40 app
        die "等了 200 秒还没健康，日志见上。"
    fi
    sleep 5
done

echo
"${COMPOSE_CMD[@]}" ps

cat <<NEXT

────────────────────────────────────────────────────────────
  容器起来了。还剩最后一步，在 Cloudflare 后台点：

    one.dash.cloudflare.com → Networks → Tunnels → 你那条隧道
    → Public Hostname → Add a public hostname

      Subdomain : unmark
      Domain    : tinylabpro.com
      Type      : HTTP
      URL       : app:8823        ← 注意是 app 不是 localhost

  注意：unmark.tinylabpro.com 现在指向介绍页（Cloudflare Pages），
  保存时 Cloudflare 会提示覆盖那条 DNS 记录 —— 确认覆盖即可，
  之后这个网址就是能直接用的工具本身。

  常用命令（都在 $DIR 目录下）：
    $COMPOSE_SHOWN logs -f app      看日志
    $COMPOSE_SHOWN restart app      重启
    $COMPOSE_SHOWN up -d --build    更新代码后重新部署
    $COMPOSE_SHOWN down             停掉
────────────────────────────────────────────────────────────

NEXT
