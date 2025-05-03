# python msg_localizer.py  --video_id 001 --query_image_path /home/dell3/lsy/MSG/data/msg/test/001/001_frames/001_000071.png --split test
import os
import json
import torch
import torch.nn.functional as F
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

# 假设以下导入已在环境中可用
from models.msg import MSGer
from util.transforms import get_transform
from util.checkpointing import load_checkpoint
from util.config_utils import get_configs


class SimpleImageDataset(Dataset):
    """
    A simplified dataset that only loads images without relying on external detection files
    """
    def __init__(self, data_split_dir, video_id, image_transforms=None):
        """
        Args:
            data_split_dir: Directory containing the data split
            video_id: ID of the video
            image_transforms: Transforms to apply to the images
        """
        self.data_split_dir = data_split_dir
        self.video_id = video_id
        self.transforms = image_transforms
        
        # Get frame paths
        self.video_path = os.path.join(data_split_dir, video_id)
        self.frame_path = os.path.join(self.video_path, f"{video_id}_frames")
        
        # Get all frame files
        self.frame_files = [f for f in os.listdir(self.frame_path) 
                           if f.endswith((".png", ".jpg", ".jpeg"))]
        self.frame_files.sort()
        
        # Create frame IDs
        self.frame_ids = [os.path.splitext(f)[0].split("_")[1] for f in self.frame_files]
        
        print(f"Loaded {len(self.frame_files)} frames for video {video_id}")
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frame_path, frame_file)
        
        # Load image
        image = read_image(frame_path)
        
        # Apply transforms if provided
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Return image and its ID
        return {
            'image': image,
            'image_idx': self.frame_ids[idx],
            'frame_path': frame_path
        }


def simple_collate_fn(batch):
    """
    Custom collate function for SimpleImageDataset
    """
    images = torch.stack([item['image'] for item in batch])
    image_idx = [item['image_idx'] for item in batch]
    frame_paths = [item['frame_path'] for item in batch]
    
    return {
        'image': images,
        'image_idx': image_idx,
        'frame_path': frame_paths
    }


class MSGLocalizer:
    """
    Leverage MSG for localization
    """
    def __init__(self, video_id, model, data, data_path, device, image_transforms=None):
        self.model = model
        self.data_path = data_path
        self.device = device
        self.video_id = video_id
        
        # Get frame IDs directly from the dataset
        self.frame_ids = data.frame_ids
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        self.data = data
        self.image_transforms = image_transforms
        
        # obtain image embeddings for all images in the scene
        self.place_feature_bank = {}
        dataloader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=8, collate_fn=simple_collate_fn)
        
        print("Extracting embeddings for all frames...")
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                additional_info = {}  # No additional info needed
                
                results = self.model(images, additional_info)
                results['place_embeddings'] = results['place_embeddings'].detach().cpu()
                
                for i, image_id in enumerate(batch['image_idx']):
                    self.place_feature_bank[image_id] = results['place_embeddings'][i]
        
        place_embeddings = [self.place_feature_bank[image_id] for image_id in sorted(self.place_feature_bank.keys())]
        self.place_embeddings = torch.stack(place_embeddings, dim=0)
        print(f"Created embeddings bank with {len(self.place_feature_bank)} frames")
            
    def localize(self, image_path: str):
        query_image = read_image(image_path)
        if self.image_transforms is not None:
            query_image = self.image_transforms(query_image)
        query_image = query_image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            ret = self.model(query_image, {})
        
        query_embedding = ret["place_embeddings"].detach().cpu()
        cos_sim = F.cosine_similarity(query_embedding, self.place_embeddings, dim=1)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        query_frame_id = filename.split('_')[1]

        if query_frame_id in self.frame2idx:
            self_idx = self.frame2idx[query_frame_id]
            cos_sim[self_idx] = -1.0

        closest_frame_idx = torch.argmax(cos_sim).item()
        closest_frame = self.frame_ids[closest_frame_idx]
        return closest_frame, cos_sim


def build_msg_localizer(video_id, model_path=None, experiment_mode="localize", device=0, split="mini-val"):
    base_config_dir = './configs/defaults'
    parser = argparse.ArgumentParser(description="Experiment configurations")
    parser.add_argument("--foo", default="construct args for compatibility")
    args = parser.parse_args([])
    args.__dict__.update({
        "experiment": experiment_mode,
        "device": device,
        "eval_split": split,
    })
    config = get_configs(base_config_dir, args, creat_subdir=False)
    
    device_no = config['device']
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    
    if model_path is None:
        if 'eval_output_dir' not in config or config['eval_output_dir'] is None:
            raise AttributeError("eval_output_dir not specified in config")
        else:
            model_path = config['eval_output_dir']
    assert config["eval_chkpt"] is not None, "eval_chkpt not specified in config"
    chkpt_path = os.path.join(model_path, "checkpoints", config["eval_chkpt"])
    logger = logging.getLogger(__name__)
    load_checkpoint(
        model=model, 
        checkpoint_path=chkpt_path,
        logger=logger,
    )
    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    model = model.to(device)
    model.eval()

    data_split_dir = os.path.join(config["dataset_path"], config["eval_split"])
    if "model_image_size" in config:
        transform = get_transform(config['model_image_size'])
    else:
        transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224), antialias=True)
        ])
    dataset = SimpleImageDataset(data_split_dir, video_id, transform)
    localizer = MSGLocalizer(
        video_id=video_id,
        model=model,
        data=dataset,
        data_path=os.path.join(data_split_dir, video_id),
        device=device,
        image_transforms=transform,
    )
    return localizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MSG Localizer CLI")
    parser.add_argument("--video_id", type=str, required=True, help="视频ID")
    parser.add_argument("--query_image_path", type=str, required=True, help="查询图像路径")
    parser.add_argument("--experiment_mode", type=str, default="localize", help="实验模式，默认为localize")
    parser.add_argument("--device", type=int, default=0, help="设备编号，默认为0")
    parser.add_argument("--split", type=str, default="test", help="数据集拆分，默认为test")
    parser.add_argument("--model_path", type=str, default=None, help="模型检查点路径，可选")
    args = parser.parse_args()

    # 构建本地化器
    localizer = build_msg_localizer(
        video_id=args.video_id,
        model_path=args.model_path,
        experiment_mode=args.experiment_mode,
        device=args.device,
        split=args.split,
    )

    # 执行本地化
    loc, sim = localizer.localize(args.query_image_path)
    print(f"Closest frame ID: {loc}")
    print(f"Similarity tensor size: {sim.size()}, Frames in database: {len(localizer.frame2idx)}, Total frames: {len(localizer.frame_ids)}")
    print(f"Closest frame index: {localizer.frame2idx[loc]}, Max similarity index: {sim.argmax()}, Max similarity value: {sim[sim.argmax()]}" )

    # 可选：打印自相似度
    query_frame_id = os.path.splitext(os.path.basename(args.query_image_path))[0].split('_')[1]
    if query_frame_id in localizer.frame2idx:
        true_id = localizer.frame2idx[query_frame_id]
        print(f"Query frame index in database: {true_id}, Self-similarity: {sim[true_id]}")
