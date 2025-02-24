#!/bin/bash

# 定义源目录和目标目录
SOURCE_DIR="/root/autodl-pub/CULane"
DEST_DIR="/root/autodl-tmp/CULane"

# 遍历源目录下所有的 .tar.gz 文件
for file in "$SOURCE_DIR"/*.tar.gz; do
  # 检查文件是否存在
  if [ -f "$file" ]; then
    # 解压文件到目标目录
    tar -xzvf "$file" -C "$DEST_DIR"
  fi
done