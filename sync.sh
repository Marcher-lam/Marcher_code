#!/bin/bash

# 自动同步脚本：自动 add、commit、pull、push

# 获取当前目录名作为默认 commit 信息
DIR_NAME=${PWD##*/}
TIME=$(date +"%Y-%m-%d %H:%M:%S")
DEFAULT_MSG="Auto-sync: $DIR_NAME at $TIME"

echo "===== Git Auto Sync ====="
echo "当前目录: $DIR_NAME"
echo "时间: $TIME"
echo

# 检查是否为 git 仓库
if [ ! -d ".git" ]; then
    echo "❌ 这里不是 Git 仓库！"
    exit 1
fi

git add .

# 是否有改动？
if git diff --cached --quiet; then
    echo "✔ 无文件改动，无需提交。"
else
    echo "✔ 已暂存修改。"
    git commit -m "$DEFAULT_MSG"
    echo "✔ 已提交: $DEFAULT_MSG"
fi

echo "✔ 正在从远程拉取更新..."
git pull --rebase

echo "✔ 正在推送到 GitHub..."
git push

echo "===== 完成同步 ====="

