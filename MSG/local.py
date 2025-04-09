import os
import json
import torch
import torch.nn.functional as F
import argparse

from models.msg import MSGer
from torch.utils.data import DataLoader
from arkit_dataset import SimpleDataset, simple_collate_fn
from util.transforms import get_transform
from torchvision.io import read_image
from util.checkpointing import load_checkpoint
from util.config_utils import get_configs
import logging

# 假设以下导入已在环境中可用
# from models.msg import MSGer
# from arkit_dataset import SimpleDataset, simple_collate_fn
# from util.transforms import get_transform
# from util.checkpointing import load_checkpoint
# from util.config_utils import get_configs

class MSGLocalizer:
    """
    Leverage MSG for localization
    """
    def __init__(self, video_id, model, data, data_path, device, image_transforms=None):
        """
        input:
            - video_id: id of the video
            - model: the model (checkpoint, loaded from outside)
            - data: dataset
            - data_path: path to the data directory
            - device: device to run the model on
            - image_transforms: image transformations
        """
        self.model = model
        self.data_path = data_path
        self.device = device
        self.video_id = video_id
        
        # Directly build the scene information without using results.json
        self.scene = self.build_scene_info(data_path, video_id)
        self.frame_ids = self.scene['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        self.data = data
        self.image_transforms = image_transforms
        
        # obtain image embeddings for all images in the scene
        self.place_feature_bank = {}
        dataloader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=8, collate_fn=simple_collate_fn)
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                additional_info = {}
                
                results = self.model(images, additional_info)
                # move the results to cpu
                results['place_embeddings'] = results['place_embeddings'].detach().cpu()
                for i in range(batch['image_idx'].size(0)):
                    image_id = batch['image_idx'][i]
                    self.place_feature_bank[image_id] = results['place_embeddings'][i]
        
        place_embeddings = [self.place_feature_bank[image_id] for image_id in sorted(self.place_feature_bank.keys())]
        self.place_embeddings = torch.stack(place_embeddings, dim=0)
    
    def build_scene_info(self, data_path, video_id):
        """
        Build scene information directly from frame files
        """
        frame_path = os.path.join(data_path, f"{video_id}_frames", "lowres_wide")
        frames = [os.path.splitext(f)[0] for f in os.listdir(frame_path) if f.endswith((".png", ".jpg", ".jpeg"))]
        frame_ids = sorted([f.split("_")[1] for f in frames])
        
        # Create a minimal scene structure
        scene_info = {
            "video_id": video_id,
            "sampled_frames": frame_ids,
            "uidmap": {},  # Empty since we don't have annotations
            "annotations": {}  # Empty since we don't have annotations
        }
        
        # Try to load ground truth if available (optional)
        gt_path = os.path.join(data_path, "refine_topo_gt.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r') as gf:
                    gt = json.load(gf)
                    if "sampled_frames" in gt:
                        scene_info["sampled_frames"] = gt["sampled_frames"]
                    if "uidmap" in gt:
                        scene_info["uidmap"] = gt["uidmap"]
                    if "annotations" in gt:
                        scene_info["annotations"] = gt["annotations"]
            except Exception as e:
                print(f"Warning: Could not load ground truth from {gt_path}: {e}")
        
        return scene_info
            
    def localize(self, image_path: str):
        """
        1. encode the image with model's place encoder
        2. find the nearest neighbor image on the MSG, excluding the query itself if present
        3. return the frame id
        """
        query_image = read_image(image_path)
        if self.image_transforms is not None:
            query_image = self.image_transforms(query_image)
        query_image = query_image.unsqueeze(0).to(self.device)
        ret = self.model(query_image, {})
        query_embedding = ret["place_embeddings"].detach().cpu()  # 1 x Hp

        # Calculate cosine similarity with all stored embeddings
        cos_sim = F.cosine_similarity(query_embedding, self.place_embeddings, dim=1)

        # Extract query frame ID from image path
        filename = os.path.splitext(os.path.basename(image_path))[0]
        query_frame_id = filename.split('_')[1]

        # If query image exists in the database, exclude it from matching
        if query_frame_id in self.frame2idx:
            self_idx = self.frame2idx[query_frame_id]
            # Set similarity to -1 to exclude self-matching
            cos_sim[self_idx] = -1.0

        # Find the closest frame index
        closest_frame_idx = torch.argmax(cos_sim).item()
        closest_frame = self.frame_ids[closest_frame_idx]
        return closest_frame, cos_sim


def build_msg_localizer(video_id, model_path=None, experiment_mode="localize", device=0, split="mini-val"):
    """
    Build MSGLocalizer instance without requiring results.json
    1. load model
    2. initialize dataset
    3. initialize and return MSGLocalizer
    """
    # get configs
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
    
    # get model
    device_no = config['device']
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = MSGer(config, device)
    
    # load from checkpoint
    if model_path is None:
        if 'eval_output_dir' not in config or config['eval_output_dir'] is None:
            raise AttributeError("eval_output_dir not specified in config")
        else:
            model_path = config['eval_output_dir']
    assert config["eval_chkpt"] is not None, "eval_chkpt not specified in config"
    chkpt_path = os.path.join(model_path, "checkpoints", config["eval_chkpt"])
    logger = logging.getLogger(__name__)
    load_checkpoint(
        model = model, 
        checkpoint_path=chkpt_path,
        logger = logger,
    )
    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    model = model.to(device)
    model.eval()

    # prepare data
    data_split_dir = os.path.join(config["dataset_path"], config["eval_split"])
    video_path = os.path.join(data_split_dir, video_id)
    transforms = get_transform(config['model_image_size'])
    dataset = SimpleDataset(data_split_dir, video_id, config, transforms, split=config['eval_split'])
    
    # initialize without msg_path
    localizer = MSGLocalizer(
        video_id = video_id,
        model = model,
        data = dataset,
        data_path = video_path,
        device = device,
        image_transforms = transforms,
    )
    
    return localizer


if __name__ == '__main__':
    # show use case of MSGLocalizer
    video_id = "41069042"
    
    # Initialize localizer without results.json
    localizer = build_msg_localizer(
        video_id = video_id,
        experiment_mode="localize",
        device = 0,
        split = "mini-val",
    )
    
    # single check
    query_image_path = "/Users/shiyu/mycode/data/mini-val/41069042/41069042_frames/lowres_wide/41069042_3048.737.png"
    loc, sim = localizer.localize(query_image_path)
    print(f"Closest frame ID: {loc}")
    print(f"Similarity tensor size: {sim.size()}, Frames in database: {len(localizer.frame2idx)}, Total frames: {len(localizer.frame_ids)}")
    print(f"Closest frame index: {localizer.frame2idx[loc]}, Max similarity index: {sim.argmax()}, Max similarity value: {sim[sim.argmax()]}")
    
    query_frame_id = "3048.737"
    if query_frame_id in localizer.frame2idx:
        true_id = localizer.frame2idx[query_frame_id]
        print(f"Query frame index in database: {true_id}, Self-similarity: {sim[true_id]}")