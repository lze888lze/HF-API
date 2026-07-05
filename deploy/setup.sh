#!/bin/bash
# ============================================================
# HF-API 一键部署脚本 - Oracle Cloud (Ubuntu ARM)
# 用法: chmod +x setup.sh && sudo ./setup.sh
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  HF-API 部署到 Oracle Cloud${NC}"
echo -e "${GREEN}========================================${NC}"

# ---- 1. 安装 Docker ----
echo -e "\n${YELLOW}[1/5] 安装 Docker...${NC}"
if ! command -v docker &> /dev/null; then
    apt-get update && apt-get install -y docker.io
    systemctl enable docker && systemctl start docker
    echo -e "${GREEN}Docker 安装完成${NC}"
else
    echo -e "${GREEN}Docker 已安装，跳过${NC}"
fi

# ---- 2. 安装 Docker Compose ----
echo -e "\n${YELLOW}[2/5] 安装 Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    apt-get install -y docker-compose
    echo -e "${GREEN}Docker Compose 安装完成${NC}"
else
    echo -e "${GREEN}Docker Compose 已安装，跳过${NC}"
fi

# ---- 3. 安装 Nginx ----
echo -e "\n${YELLOW}[3/5] 安装 Nginx...${NC}"
if ! command -v nginx &> /dev/null; then
    apt-get install -y nginx
    systemctl enable nginx
    echo -e "${GREEN}Nginx 安装完成${NC}"
else
    echo -e "${GREEN}Nginx 已安装，跳过${NC}"
fi

# ---- 4. 配置 Nginx ----
echo -e "\n${YELLOW}[4/5] 配置 Nginx...${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NGINX_CONF="${SCRIPT_DIR}/nginx/hf-api.conf"

if [ -f "$NGINX_CONF" ]; then
    ln -sf "$NGINX_CONF" /etc/nginx/sites-available/hf-api
    ln -sf /etc/nginx/sites-available/hf-api /etc/nginx/sites-enabled/hf-api
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl restart nginx
    echo -e "${GREEN}Nginx 配置完成${NC}"
else
    echo -e "${RED}未找到 ${NGINX_CONF}${NC}"
    exit 1
fi

# ---- 5. 构建 & 启动 ----
echo -e "\n${YELLOW}[5/5] 构建镜像并启动...${NC}"

cd "$SCRIPT_DIR"

# 如果没有 .env 文件，从 .env.example 复制一份
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}已从 .env.example 创建 .env，请编辑填入 R2 配置后重新运行${NC}"
fi

docker-compose build --no-cache
docker-compose up -d

# 防火墙
iptables -I INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || true
iptables -I INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "验证: curl http://localhost:7860/health"
echo ""
echo "后续步骤:"
echo "  1. Oracle 控制台 → VCN → Security List 放行 80/443"
echo "  2. 域名 A 记录指向甲骨文 IP（DNS Only，关掉橙色云朵）"
echo "  3. 签 HTTPS: sudo certbot --nginx -d hf-api.lze.cc.cd"
echo "  4. 保活: crontab -e 加 */5 * * * * curl -s http://localhost:7860/health > /dev/null"
