import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, SpatialPadd, CropForegroundd, Resized, 
    ToTensord
)
from monai.data import load_decathlon_datalist
import nibabel as nib
from voco_train import init_log, VoCoHead
import logging
import os

def load_and_preprocess_image(image_path, args):
    # Load the image
    img = nib.load(image_path).get_fdata()
    
    # Add channel dimension if needed
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
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
    
    return img

def prepare_model_for_visualization(model):
    """Prepare the model for visualization by handling batch normalization"""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

def visualize_features(model, img, crop, save_dir):
    """Visualize the feature representations of the model"""
    prepare_model_for_visualization(model)
    
    with torch.no_grad():
        # Repeat the image to create a larger batch
        img_batch = img.repeat(4, 1, 1, 1, 1)  # Create a batch of 4
        crop_batch = crop.repeat(4, 1, 1, 1, 1)
        
        # Get student features
        student_features = model.student(img_batch)
        teacher_features = model.teacher(img_batch)
        
        # Get crop features
        crop_features = model.student(crop_batch)
        
        # Take mean across batch dimension
        student_features = student_features.mean(0, keepdim=True)
        teacher_features = teacher_features.mean(0, keepdim=True)
        crop_features = crop_features.mean(0, keepdim=True)
        
        # Calculate similarity maps
        similarity_map = torch.nn.functional.cosine_similarity(
            student_features.unsqueeze(1),
            crop_features.unsqueeze(0),
            dim=2
        )
        
        # Save visualizations
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot similarity map
        plt.figure(figsize=(10, 5))
        plt.imshow(similarity_map.cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title('Feature Similarity Map')
        plt.savefig(os.path.join(save_dir, 'similarity_map.png'))
        plt.close()
        
        # Plot feature distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.hist(student_features.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Student')
        plt.hist(teacher_features.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Teacher')
        plt.title('Feature Distribution')
        plt.legend()
        
        plt.subplot(132)
        plt.hist(crop_features.cpu().numpy().flatten(), bins=50, alpha=0.5)
        plt.title('Crop Feature Distribution')
        
        plt.subplot(133)
        plt.hist(similarity_map.cpu().numpy().flatten(), bins=50, alpha=0.5)
        plt.title('Similarity Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
        plt.close()

def visualize_attention(model, img, save_dir):
    """Visualize the intermediate feature maps from the Swin Transformer"""
    prepare_model_for_visualization(model)
    
    with torch.no_grad():
        # Register hooks to get intermediate feature maps
        feature_maps = []
        
        def hook_fn(module, input, output):
            # Take mean across batch dimension if needed
            if len(output.shape) > 3:
                output = output.mean(0, keepdim=True)
            feature_maps.append(output.detach())
        
        # Register hooks for each layer
        hooks = []
        for name, module in model.student.named_modules():
            if isinstance(module, torch.nn.Conv3d):  # Look for convolutional layers
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass with repeated batch
        img_batch = img.repeat(4, 1, 1, 1, 1)
        _ = model.student(img_batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Visualize feature maps
        for i, feature_map in enumerate(feature_maps):
            # Take the middle slice of the 3D feature map
            if len(feature_map.shape) == 5:  # (B, C, D, H, W)
                feature_map = feature_map[0]  # Take first batch
                feature_map = feature_map.mean(0)  # Average across channels
                middle_slice = feature_map.shape[0] // 2
                feature_map = feature_map[middle_slice].cpu().numpy()
            else:
                continue  # Skip non-3D feature maps
            
            plt.figure(figsize=(10, 10))
            plt.imshow(feature_map, cmap='hot')
            plt.colorbar()
            plt.title(f'Feature Map Layer {i}')
            plt.savefig(os.path.join(save_dir, f'feature_map_layer_{i}.png'))
            plt.close()

def visualize_model_architecture(model, save_dir):
    """Visualize the model architecture and parameter distribution"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot parameter distribution for each layer
    plt.figure(figsize=(15, 10))
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.hist(param.data.cpu().numpy().flatten(), bins=50, alpha=0.5, label=name)
    plt.title('Parameter Distribution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_distribution.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize VoCo Model")
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
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
    parser.add_argument("--save_dir", default="visualizations", type=str, help="directory to save visualizations")
    
    args = parser.parse_args()
    logger = init_log('global', logging.INFO)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    model = VoCoHead(args)
    model.cuda()
    
    # Load the model weights
    model_path = "logsfinal_model.pth"
    print(f"Loading model from {model_path}")
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    print("Model loaded successfully")
    
    # Visualize model architecture
    print("Generating model architecture visualization...")
    visualize_model_architecture(model, args.save_dir)
    
    # Load validation files
    jsonlist = "./jsons/btcv.json"
    datadir = "../dataset for VoCo/BTCV"
    val_files = load_decathlon_datalist(jsonlist, False, "validation", base_dir=datadir)
    
    # Process first validation image
    if val_files:
        val_file = val_files[0]
        img_path = val_file["image"]
        print(f"Processing image: {img_path}")
        
        # Load and preprocess image
        img = load_and_preprocess_image(img_path, args)
        
        # Create a crop
        roi_small = args.roi_x
        center_x = roi_small
        center_y = roi_small
        center_z = roi_small
        
        crop = img[:, 
                  center_x-roi_small//2:center_x+roi_small//2,
                  center_y-roi_small//2:center_y+roi_small//2,
                  center_z-roi_small//2:center_z+roi_small//2]
        
        # Add batch dimension and move to GPU
        img = img.unsqueeze(0).cuda()
        crop = crop.unsqueeze(0).cuda()
        
        # Create visualizations
        print("Generating feature visualizations...")
        visualize_features(model, img, crop, args.save_dir)
        
        print("Generating attention visualizations...")
        visualize_attention(model, img, args.save_dir)
        
        print(f"Visualizations saved to {args.save_dir}")
    else:
        print("No validation files found")

if __name__ == "__main__":
    main() 