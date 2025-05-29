#!/bin/bash

# 7-Scenes数据集语义引导位姿估计评估脚本

# 设置变量
OUTPUT_DIR="/home/hri3090/lsy/pose_estimate/results/7scenes_$(date +%Y%m%d_%H%M%S)"
FRAME_INTERVAL=1
MAX_PAIRS=100
MAX_ROT_ERR=4.0
MAX_TRANS_ERR=2.0

# 为室内场景设置合适的文本提示
TEXT_PROMPTS=(
    "furniture . objects . indoor items"
    "table . chair . cabinet"
    "monitor . computer . desk"
)

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始执行7-Scenes数据集评估..."
echo "输出目录: $OUTPUT_DIR"
echo "每个数据集最大处理图像对数: $MAX_PAIRS"
echo "帧间隔: $FRAME_INTERVAL"

# 运行评估脚本
python pose_estimate/run_all_7scenes.py \
    --datasets all \
    --output-dir "$OUTPUT_DIR" \
    --frame-interval $FRAME_INTERVAL \
    --max-pairs $MAX_PAIRS \
    --max-rot-err $MAX_ROT_ERR \
    --max-trans-err $MAX_TRANS_ERR \
    --text-prompts "${TEXT_PROMPTS[@]}"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "评估完成！"
    # 查找生成的汇总文件
    SUMMARY_FILE=$(find "$OUTPUT_DIR" -name "7scenes_summary_*.txt" | sort | tail -n 1)
    if [ -n "$SUMMARY_FILE" ]; then
        echo "汇总结果文件: $SUMMARY_FILE"
        # 显示所有数据集汇总结果部分
        echo "================ 评估结果摘要 ================"
        sed -n '/所有数据集汇总结果:/,$ p' "$SUMMARY_FILE"
    fi
else
    echo "评估过程中出错，请检查日志"
fi 