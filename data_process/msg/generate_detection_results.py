# python generate_detection_results.py --output_dir ./exp-results/custom-detections --num_videos 5
# --output_dir: 指定输出目录
# --num_videos: 要生成的视频数量
# --video_id_prefix: 视频ID前缀
# --num_frames: 每个视频的帧数
import json
import random
import os
import argparse
from tqdm import tqdm

class DetectionResultGenerator:
    def __init__(self, num_frames=30, start_frame=305.0, frame_interval=0.5, 
                 object_classes=None, output_dir="./generated_detections"):
        """
        初始化检测结果生成器
        
        参数:
            num_frames: 要生成的帧数量
            start_frame: 起始帧号
            frame_interval: 帧间隔
            object_classes: 对象类别字典，格式为 {类别名称: 类别ID}
            output_dir: 输出目录
        """
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.frame_interval = frame_interval
        
        # 如果没有提供对象类别，使用默认类别
        if object_classes is None:
            self.object_classes = {
                "cabinet": 3,
                "chair": 4,
                "counter": 5,
                "curtain": 6,
                "desk": 7,
                "door": 8,
                "lamp": 10,
                "plant": 11,
                "stool": 12,
                "sofa": 13,
                "table": 14,
                "tv_monitor": 16,
                "window": 18
            }
        else:
            self.object_classes = object_classes
            
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_random_bbox(self, img_width=256, img_height=192):
        """生成随机边界框"""
        x1 = random.randint(0, img_width - 40)
        y1 = random.randint(0, img_height - 40)
        x2 = random.randint(x1 + 40, min(x1 + 200, img_width))
        y2 = random.randint(y1 + 40, min(y1 + 200, img_height))
        return [x1, y1, x2, y2]
    
    def generate_frame_detections(self, frame_id):
        """为单个帧生成检测结果"""
        detections = {}
        
        # 随机决定这一帧检测到的对象数量 (0-4)
        num_objects = random.randint(0, 4)
        
        for obj_id in range(num_objects):
            # 随机选择一个对象类别
            obj_class_name = random.choice(list(self.object_classes.keys()))
            obj_class_id = self.object_classes[obj_class_name]
            
            # 生成随机边界框和置信度分数
            bbox = self.generate_random_bbox()
            score = random.uniform(0.35, 0.7)
            
            detections[str(obj_id)] = {
                "bbox": bbox,
                "label": float(obj_class_id),
                "score": score,
                "uniq": f"{obj_class_name}:{obj_class_id}"
            }
        
        return detections
    
    def generate_video_detections(self, video_id):
        """为单个视频生成检测结果"""
        result = {
            "detections": {}
        }
        
        # 生成所有帧的检测结果
        for i in range(self.num_frames):
            frame_id = self.start_frame + i * self.frame_interval
            frame_id_str = f"{frame_id:.3f}"
            
            result["detections"][frame_id_str] = self.generate_frame_detections(frame_id)
        
        return result
    
    def save_video_detections(self, video_id, detections):
        """保存视频检测结果到JSON文件"""
        output_path = os.path.join(self.output_dir, f"{video_id}", "results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        
        return output_path
    
    def generate_all_videos(self, video_ids):
        """为多个视频生成检测结果"""
        results = {}
        
        for video_id in tqdm(video_ids, desc="生成视频检测结果"):
            video_detections = self.generate_video_detections(video_id)
            output_path = self.save_video_detections(video_id, video_detections)
            results[video_id] = output_path
        
        return results

def main():
    parser = argparse.ArgumentParser(description="生成物体检测结果JSON文件")
    parser.add_argument("--output_dir", type=str, default="./generated_detections", 
                        help="输出目录")
    parser.add_argument("--num_videos", type=int, default=5, 
                        help="要生成的视频数量")
    parser.add_argument("--video_id_prefix", type=str, default="video_", 
                        help="视频ID前缀")
    parser.add_argument("--num_frames", type=int, default=30, 
                        help="每个视频的帧数")
    
    args = parser.parse_args()
    
    # 创建视频ID列表
    video_ids = [f"{args.video_id_prefix}{i:03d}" for i in range(args.num_videos)]
    
    # 初始化生成器
    generator = DetectionResultGenerator(
        num_frames=args.num_frames,
        output_dir=args.output_dir
    )
    
    # 生成所有视频的检测结果
    results = generator.generate_all_videos(video_ids)
    
    print(f"已成功生成 {len(results)} 个视频的检测结果")
    for video_id, path in results.items():
        print(f"  - {video_id}: {path}")

if __name__ == "__main__":
    main()