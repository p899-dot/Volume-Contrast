import torch
import argparse
from voco_train import init_log, VoCoHead
import logging
import numpy as np
import os
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, SpatialPadd, CropForegroundd, Resized, 
    ToTensord
)
from monai.data import Dataset, DataLoader, load_decathlon_datalist

def load_and_preprocess_image(image_path, args):
    # Load the image
    img = nib.load(image_path).get_fdata()
    print(f"Original image shape: {img.shape}")
    
    # Add channel dimension if needed
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    print(f"After adding channel: {img.shape}")
    
    # Convert to tensor
    img = torch.from_numpy(img).float()
    
    # Normalize
    img = (img - args.a_min) / (args.a_max - args.a_min)
    img = img * (args.b_max - args.b_min) + args.b_min
    
    # Resize to target size
    target_size = (args.roi_x * 2, args.roi_y * 2, args.roi_z * 2)
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=target_size,
        mode='trilinear',
        align_corners=True
    ).squeeze(0)
    print(f"After resize: {img.shape}")
    
    return img

def evaluate(args, model, val_files):
    model.eval()
    pos_losses = []
    neg_losses = []
    base_losses = []
    num_processed = 0
    
    with torch.no_grad():
        for val_file in val_files:
            try:
                # Load and preprocess image
                img_path = val_file["image"]
                print(f"Processing image: {img_path}")
                img = load_and_preprocess_image(img_path, args)
                
                # Create a simple crop
                roi_small = args.roi_x
                center_x = roi_small
                center_y = roi_small
                center_z = roi_small
                
                # Extract crop
                crop = img[:, 
                          center_x-roi_small//2:center_x+roi_small//2,
                          center_y-roi_small//2:center_y+roi_small//2,
                          center_z-roi_small//2:center_z+roi_small//2]
                
                # Add batch dimension and convert to list
                img = img.unsqueeze(0).cuda()
                crop = crop.unsqueeze(0).cuda()
                
                # Create positive and negative labels
                pos_label = torch.ones(1, 16).cuda()  # Positive label
                neg_label = torch.zeros(1, 16).cuda()  # Negative label
                
                # Stack multiple copies to match expected batch size
                img = img.repeat(args.sw_batch_size, 1, 1, 1, 1)
                crop = crop.repeat(args.sw_batch_size, 1, 1, 1, 1)
                pos_label = pos_label.repeat(args.sw_batch_size, 1)
                neg_label = neg_label.repeat(args.sw_batch_size, 1)
                
                pos, neg, b_loss = model(img, crop, pos_label=pos_label, neg_label=neg_label)
                
                pos_losses.append(pos.item())
                neg_losses.append(neg.item())
                base_losses.append(b_loss.item())
                num_processed += 1
                print(f"Processed image {num_processed}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                print(f"Image shapes - img: {img.shape if 'img' in locals() else 'N/A'}, "
                      f"crop: {crop.shape if 'crop' in locals() else 'N/A'}, "
                      f"label: {label.shape if 'label' in locals() else 'N/A'}")
                continue
    
    if not pos_losses:
        return 0, 0, 0
    
    return np.mean(pos_losses), np.mean(neg_losses), np.mean(base_losses)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VoCo Model")
    parser.add_argument("--logdir", default="logs", type=str, help="directory containing the model")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--roi_x", default=48, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=48, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=48, type=int, help="roi size in z direction")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=24, type=int, help="embedding size")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="directory to save cache")
    
    args = parser.parse_args()
    logger = init_log('global', logging.INFO)
    
    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load model
    model = VoCoHead(args)
    model.cuda()
    
    # Load the final model weights
    model_path = "logsfinal_model.pth"  # Updated path to match the actual file
    print(f"Loading model from {model_path}")
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    print("Model loaded successfully")
    
    # Load validation files
    jsonlist = "./jsons/btcv.json"
    datadir = "../dataset for VoCo/BTCV"
    val_files = load_decathlon_datalist(jsonlist, False, "validation", base_dir=datadir)
    print(f"Found {len(val_files)} validation files")
    
    # Evaluate
    print("Starting evaluation...")
    pos_loss, neg_loss, base_loss = evaluate(args, model, val_files)
    total_loss = pos_loss + neg_loss + base_loss
    
    logger.info(f"Validation Results:")
    logger.info(f"Positive Loss: {pos_loss:.4f}")
    logger.info(f"Negative Loss: {neg_loss:.4f}")
    logger.info(f"Base Loss: {base_loss:.4f}")
    logger.info(f"Total Loss: {total_loss:.4f}")

if __name__ == "__main__":
    main() 