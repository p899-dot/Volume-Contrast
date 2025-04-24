#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
import torch.optim as optim
from models.voco_head import VoCoHead
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    SpatialPadd, CenterSpatialCropd, RandSpatialCropd,
    RandFlipd, RandRotate90d, RandAffined, RandScaleIntensityd,
    ToTensord, Compose, RandCropByPosNegLabeld, EnsureTyped
)
from monai.data import CacheDataset, DataLoader, Dataset
from monai.data.decathlon_datalist import load_decathlon_datalist
from utils.utils import init_log
import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser(description='Train VoCo model')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the data directory')
    parser.add_argument('--logdir', type=str, default=config.LOG_DIR,
                      help='Path to save logs and checkpoints')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--eval_num', type=int, default=config.EVAL_NUM,
                      help='Evaluation frequency')
    parser.add_argument('--warmup_steps', type=int, default=config.WARMUP_STEPS,
                      help='Number of warmup steps')
    
    # Model settings
    parser.add_argument('--in_channels', type=int, default=config.IN_CHANNELS,
                      help='Number of input channels')
    parser.add_argument('--feature_size', type=int, default=config.FEATURE_SIZE,
                      help='Feature size')
    parser.add_argument('--dropout_path_rate', type=float, default=config.DROPOUT_PATH_RATE,
                      help='Dropout path rate')
    parser.add_argument('--use_checkpoint', type=bool, default=config.USE_CHECKPOINT,
                      help='Use checkpoint')
    parser.add_argument('--spatial_dims', type=int, default=config.SPATIAL_DIMS,
                      help='Spatial dimensions')
    
    # Data preprocessing
    parser.add_argument('--a_min', type=float, default=config.A_MIN,
                      help='Minimum value for intensity scaling')
    parser.add_argument('--a_max', type=float, default=config.A_MAX,
                      help='Maximum value for intensity scaling')
    parser.add_argument('--b_min', type=float, default=config.B_MIN,
                      help='Minimum value for output scaling')
    parser.add_argument('--b_max', type=float, default=config.B_MAX,
                      help='Maximum value for output scaling')
    parser.add_argument('--roi_x', type=int, default=config.ROI_X,
                      help='ROI size in x dimension')
    parser.add_argument('--roi_y', type=int, default=config.ROI_Y,
                      help='ROI size in y dimension')
    parser.add_argument('--roi_z', type=int, default=config.ROI_Z,
                      help='ROI size in z dimension')
    
    # Optimizer settings
    parser.add_argument('--opt', type=str, default=config.OPTIMIZER,
                      help='Optimizer type (sgd or adamw)')
    parser.add_argument('--decay', type=float, default=config.DECAY,
                      help='Weight decay')
    parser.add_argument('--momentum', type=float, default=config.MOMENTUM,
                      help='Momentum for SGD optimizer')
    
    # Other settings
    parser.add_argument('--resume', type=str, default=config.RESUME,
                      help='Resume training from checkpoint')
    parser.add_argument('--local_rank', type=int, default=config.LOCAL_RANK,
                      help='Local rank for distributed training')
    parser.add_argument('--noamp', action='store_true', default=config.NOAMP,
                      help='Disable automatic mixed precision')
    parser.add_argument('--dist-url', default=config.DIST_URL,
                      help='URL used to set up distributed training')
    
    return parser.parse_args()

def get_data_loaders(args):
    """Get data loaders for training and validation."""
    json_path = os.path.join("jsons", "btcv.json")
    
    print(f"Loading data from directory: {args.data_dir}")
    print(f"Using JSON file: {json_path}")

    train_files = load_decathlon_datalist(json_path, True, "training", base_dir=args.data_dir)
    val_files = load_decathlon_datalist(json_path, True, "validation", base_dir=args.data_dir)
    
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of validation files: {len(val_files)}")

    # Define transformations
    common_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        SpatialPadd(
            keys=["image"],
            spatial_size=[args.roi_x, args.roi_y, args.roi_z],
            mode="constant"
        ),
    ]

    train_transforms = common_transforms + [
        RandCropByPosNegLabeld(
            keys=["image"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(
            keys=["image"],
            prob=0.5,
            spatial_axis=[0, 1, 2]
        ),
        RandRotate90d(
            keys=["image"],
            prob=0.5,
            max_k=3,
            spatial_axes=[0, 1]
        ),
        RandAffined(
            keys=["image"],
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear"),
            padding_mode="zeros"
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=0.3
        ),
        EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"])
    ]

    val_transforms = common_transforms + [
        CenterSpatialCropd(
            keys=["image"],
            roi_size=[args.roi_x, args.roi_y, args.roi_z]
        ),
        EnsureTyped(keys=["image"]),
        ToTensord(keys=["image"])
    ]

    train_ds = CacheDataset(
        data=train_files,
        transform=Compose(train_transforms),
        cache_rate=0.5,
        num_workers=4
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=Compose(val_transforms),
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint to disk"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def train(args, model, train_loader, val_loader, optimizer, scheduler, scaler):
    """Train the model"""
    model.train()
    val_best = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        print(f"Current lr: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Global step: {global_step}")
        
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=not args.noamp):
                    loss = model(batch)
                
                if not args.noamp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step(epoch + i / len(train_loader))
                
                running_loss += loss.item()
                global_step += 1
                
                if i % 10 == 0:
                    avg_loss = running_loss / (i + 1)
                    print(f"Step [{global_step}] Epoch [{epoch+1}/{args.epochs}][{i}/{len(train_loader)}] "
                          f"Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")
            except Exception as e:
                print(f"Error during training step: {e}")
                continue
        
        if epoch % args.eval_num == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    try:
                        with torch.cuda.amp.autocast(enabled=not args.noamp):
                            loss = model(val_batch)
                        val_loss += loss.item()
                    except Exception as e:
                        print(f"Error during validation step: {e}")
                        continue
            
            val_loss /= len(val_loader)
            print(f"Step [{global_step}] Validation Loss: {val_loss:.4f}")
            
            if val_loss < val_best:
                val_best = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_best,
                    'args': args,
                }, True, args.logdir)
            
            model.train()
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_best,
                'args': args,
            }, False, args.logdir)
    
    return val_best, global_step

def main():
    print("Starting main function...")
    args = parse_args()
    print("Arguments parsed successfully")
    
    print("Setting up directories...")
    args.logdir = os.path.abspath(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    print(f"Using log directory: {args.logdir}")

    print("Setting up logging...")
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    print("Initializing model...")
    model = VoCoHead(args)
    model.cuda()
    print("Model initialized and moved to CUDA")

    print("Setting up optimizer...")
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True
    )
    print("Optimizer initialized")

    print("Setting up scheduler...")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    print("Scheduler initialized")

    print("Setting up AMP scaler...")
    scaler = GradScaler(enabled=not args.noamp)
    print("Scaler initialized")

    print("Loading data...")
    train_loader, val_loader = get_data_loaders(args)
    print("Data loaders created")

    print("Starting training...")
    val_best, final_global_step = train(args, model, train_loader, val_loader, optimizer, scheduler, scaler)
    print(f'Training completed. Best validation loss: {val_best:.4f}')
    print(f'Final global step: {final_global_step}')

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize logging
    os.makedirs(args.logdir, exist_ok=True)
    init_log(args.logdir)
    logging.info(f"Starting training with args: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args)
    logging.info("Data loaders created successfully")
    
    # Create model
    model = VoCoHead(
        in_channels=args.in_channels,
        feature_size=args.feature_size,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims
    ).to(device)
    logging.info("Model created successfully")
    
    # Set up optimizer
    if args.opt.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.decay
        )
    logging.info(f"Using optimizer: {args.opt}")
    
    # Set up scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.warmup_steps, T_mult=2
    )
    
    # Set up AMP scaler
    scaler = GradScaler(enabled=not args.noamp)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            logging.info(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            logging.error(f"No checkpoint found at {args.resume}")
    
    # Train model
    try:
        val_best, global_step = train(args, model, train_loader, val_loader, optimizer, scheduler, scaler)
        logging.info(f"Training completed. Best validation loss: {val_best:.4f}, Global steps: {global_step}")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise e 