# 语义引导SIFT位姿估计

本项目实现了基于Grounding-DINO语义分割的SIFT特征点匹配和位姿估计方法。通过语义引导，可以有效提高特征点匹配的质量和位姿估计的准确性。

## 功能特点

- 使用Grounding-DINO进行基于文本提示的目标检测
- 使用SAM (Segment Anything Model) 生成高质量语义掩码
- 将语义掩码应用于SIFT特征提取，提高匹配点质量
- 支持标准SIFT和语义引导SIFT的对比实验
- 提供位姿估计的误差分析和可视化

## 环境要求

- Python 3.7+
- CUDA (如需GPU加速)
- 依赖库：请见 requirements.txt

## 安装步骤

1. 克隆项目

```bash
git clone <仓库地址>
cd EVL-MRCC
```

2. 安装依赖

```bash
pip install -r pose_estimate/requirements.txt
```

3. 配置模型

请将以下模型下载到正确的位置：
- Grounding-DINO模型：放置于 weights/groundingdino_swint_ogc.pth
- SAM模型：放置于 weights/sam_vit_h_4b8939.pth

模型下载链接：
- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM](https://github.com/facebookresearch/segment-anything)

## 使用方法

### 基本用法

```bash
cd pose_estimate
python main.py
```

### 命令行参数

现在支持通过命令行参数指定图像路径、模型路径和其他选项：

```bash
python main.py --img1 /path/to/image1.png --img2 /path/to/image2.png \
               --dino-config /path/to/GroundingDINO_SwinT_OGC.py \
               --dino-weights /path/to/groundingdino_swint_ogc.pth \
               --sam-weights /path/to/sam_vit_h_4b8939.pth \
               --text-prompt "walls . furniture . fixtures" \
               --device cuda \
               --output-dir results_folder
```

#### 可用的命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--img1` | 第一张图像的路径 | /Users/shiyu/mycode/EVL-MRCC/images/41069021_305.377.png |
| `--img2` | 第二张图像的路径 | /Users/shiyu/mycode/EVL-MRCC/images/41069021_306.360.png |
| `--dino-config` | Grounding-DINO配置文件路径 | GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py |
| `--dino-weights` | Grounding-DINO模型权重路径 | weights/groundingdino_swint_ogc.pth |
| `--sam-weights` | SAM模型权重路径 | weights/sam_vit_h_4b8939.pth |
| `--text-prompt` | 用于生成语义掩码的文本提示 | "walls . furniture . fixtures" |
| `--device` | 运行模型的设备 | 自动选择cuda或cpu |
| `--output-dir` | 结果保存的目录路径 | results |

### 文本提示示例

根据不同场景类型，可以使用不同的文本提示：

室内场景：
```
--text-prompt "walls . furniture . fixtures"
```

室外场景：
```
--text-prompt "buildings . trees . landmarks"
```

车辆场景：
```
--text-prompt "cars . roads . traffic signs"
```

## 结果说明

程序运行后会将所有结果保存到指定的输出目录（默认为`results/`），并在控制台输出相关信息：

1. 语义掩码图像 (`image1_with_mask.png` 和 `image2_with_mask.png`)
2. 标准SIFT特征匹配结果图 (`standard_sift_matches.png`)
3. 语义引导SIFT特征匹配结果图 (`semantic_guided_sift_matches.png`)
4. 包含位姿估计详细结果、误差分析和性能对比的文本文件 (`results.txt`)

在无GUI或远程服务器环境中，所有可视化结果都会保存为PNG图像文件，而不会尝试进行图形界面显示，避免了X11相关错误。

## 参考文献

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [OpenCV-Python](https://docs.opencv.org/master/) 