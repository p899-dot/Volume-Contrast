#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import logging
import torch
import torch.distributed as dist
import torch.optim as optim
from models.voco_head import VoCoHead
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    SpatialPadd, CenterSpatialCropd, RandSpatialCropd,
    RandFlipd, RandRotate90d, RandAffined, RandScaleIntensityd,
    ToTensord, Compose
)
from monai.data import CacheDataset, DataLoader
from monai.data.decathlon_datalist import load_decathlon_datalist
from utils.utils import init_log
from utils.data_utils import get_loader_1k
import platform
import config as cfg
import sys

# Only import resource module on Unix-like systems
if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint to disk"""
    try:
        filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def train(args):
    """Main training function"""
    # Initialize logging
    logger = init_log("train", args.logdir)
    logger.info(f"Starting training with arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args)
    
    # Create model
    model = VoCoHead(
        in_channels=1,
        feature_size=48,
        dropout_path_rate=0.0,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)
    
    # Set up optimizer
    if args.opt.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.decay,
        )
    else:  # default to AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.decay,
        )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs,
        T_mult=1,
        eta_min=1e-6,
    )
    
    # Initialize AMP scaler
    scaler = GradScaler(enabled=not args.noamp)
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(os.path.join(args.logdir, "checkpoint.pth.tar")):
        try:
            checkpoint = torch.load(os.path.join(args.logdir, "checkpoint.pth.tar"))
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    try:
        # Training loop
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(train_loader):
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                
                optimizer.zero_grad()
                
                with autocast(enabled=not args.noamp):
                    outputs = model(images)
                    loss = model.loss(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % args.eval_num == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_data in val_loader:
                        val_images = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device)
                        
                        with autocast(enabled=not args.noamp):
                            val_outputs = model(val_images)
                            loss = model.loss(val_outputs, val_labels)
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Validation loss: {avg_val_loss:.4f}")
                
                # Save checkpoint
                is_best = avg_val_loss < best_loss
                best_loss = min(avg_val_loss, best_loss)
                
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    is_best,
                    args.logdir,
                )
            
            scheduler.step()
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def setup_paths():
    """Set up paths for data and logs"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "dataset for VoCo", "BTCV"))
    default_logs_dir = os.path.join(script_dir, "logs")
    
    # Create logs directory if it doesn't exist
    os.makedirs(default_logs_dir, exist_ok=True)
    
    # Print current working directory and paths for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Default data directory: {default_data_dir}")
    print(f"Default logs directory: {default_logs_dir}")
    
    return default_data_dir, default_logs_dir

def parse_args():
    """Parse command line arguments with better defaults"""
    default_data_dir, default_logs_dir = setup_paths()
    
    parser = argparse.ArgumentParser(description="VoCo Training Script")
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                      help="Path to data directory")
    parser.add_argument("--logdir", type=str, default=default_logs_dir,
                      help="Path to log directory")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--eval_num", type=int, default=1,
                      help="Evaluation frequency")
    # Add intensity scaling parameters
    parser.add_argument("--a_min", type=float, default=-175.0,
                      help="Minimum intensity for scaling")
    parser.add_argument("--a_max", type=float, default=250.0,
                      help="Maximum intensity for scaling")
    parser.add_argument("--b_min", type=float, default=0.0,
                      help="Target minimum intensity")
    parser.add_argument("--b_max", type=float, default=1.0,
                      help="Target maximum intensity")
    # Add ROI size parameters
    parser.add_argument("--roi_x", type=int, default=96,
                      help="ROI size in x dimension")
    parser.add_argument("--roi_y", type=int, default=96,
                      help="ROI size in y dimension")
    parser.add_argument("--roi_z", type=int, default=96,
                      help="ROI size in z dimension")
    # Add optimizer parameters
    parser.add_argument("--opt", type=str, default="adamw",
                      help="Optimizer type (adamw or sgd)")
    parser.add_argument("--momentum", type=float, default=0.9,
                      help="Momentum for SGD optimizer")
    parser.add_argument("--decay", type=float, default=1e-5,
                      help="Weight decay")
    # Add AMP control
    parser.add_argument("--noamp", action="store_true",
                      help="Disable automatic mixed precision")
    
    args = parser.parse_args()
    
    # Print paths for verification
    print(f"Data directory: {args.data_dir}")
    print(f"Logs directory: {args.logdir}")
    
    return args

def get_data_loaders(args):
    """Set up data loaders for training and validation"""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsons", "btcv.json")
    
    # Load training and validation files
    train_files = load_decathlon_datalist(json_path, True, "training", base_dir=args.data_dir)
    val_files = load_decathlon_datalist(json_path, True, "validation", base_dir=args.data_dir)
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
    
    # Common transforms
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode=("constant", "constant"),
        ),
    ]
    
    # Training transforms with augmentation
    train_transforms = common_transforms + [
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.26, 0.26, 0.26),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
    
    # Validation transforms
    val_transforms = common_transforms + [
        CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
        ),
        ToTensord(keys=["image", "label"]),
    ]
    
    # Create datasets
    train_ds = CacheDataset(
        data=train_files,
        transform=Compose(train_transforms),
        cache_rate=1.0,
        num_workers=4,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=Compose(val_transforms),
        cache_rate=1.0,
        num_workers=4,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # validation is done one sample at a time
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def main():
    """Main entry point"""
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main() 