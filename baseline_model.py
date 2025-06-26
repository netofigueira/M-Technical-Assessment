import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from sklearn.metrics import jaccard_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for weakly supervised segmentation.
    Only computes loss on labeled pixels, ignoring unlabeled regions.
    """
    def __init__(self, ignore_index=255, weight=None):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, C, H, W) - model outputs (logits)
            targets: (N, H, W) - partial labels with ignore_index for unlabeled pixels
        """
        # Create mask for valid (labeled) pixels
        valid_mask = (targets != self.ignore_index)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Flatten predictions and targets for valid pixels only
        predictions_flat = predictions.permute(0, 2, 3, 1).contiguous().view(-1, predictions.size(1))
        targets_flat = targets.view(-1)
        valid_mask_flat = valid_mask.view(-1)
        
        # Select only valid pixels
        valid_predictions = predictions_flat[valid_mask_flat]
        valid_targets = targets_flat[valid_mask_flat]
        
        # Compute standard cross-entropy on valid pixels
        loss = F.cross_entropy(valid_predictions, valid_targets, weight=self.weight)
        
        return loss

class SimpleUNet(nn.Module):
    """
    Simplified U-Net for binary segmentation
    """
    def __init__(self, in_channels=3, out_channels=2, features=64):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.encoder1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = self._conv_block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = self._conv_block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = self._conv_block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = self._conv_block(features * 2, features)
        
        self.final = nn.Conv2d(features, out_channels, 1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final(dec1)

class RemoteSensingDataset(Dataset):
    """
    Dataset class for remote sensing images with partial labels
    """
    def __init__(self, image_dir, mask_dir, transform=None, sampling_percentage=0.05, 
                 sampling_strategy='random', image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.sampling_percentage = sampling_percentage
        self.sampling_strategy = sampling_strategy
        self.image_size = image_size
        
        # Get all image filenames
        self.filenames = [f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        self.filenames.sort()
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Convert to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Convert mask to binary (assuming white=foreground, black=background)
        binary_mask = (mask_array > 127).astype(np.int32)
        
        # Generate partial mask
        partial_mask = self._generate_partial_mask(binary_mask)
        
        # Apply transforms
        if self.transform:
            image_array = self.transform(image_array)
        else:
            image_array = torch.FloatTensor(image_array).permute(2, 0, 1) / 255.0
            
        return image_array, torch.LongTensor(partial_mask), torch.LongTensor(binary_mask)
    
    def _generate_partial_mask(self, mask, ignore_label=255):
        """Generate partial mask with specified sampling strategy"""
        partial_mask = np.full_like(mask, ignore_label, dtype=np.int32)
        
        if self.sampling_strategy == 'random':
            return self._random_sampling(mask, partial_mask, ignore_label)
        elif self.sampling_strategy == 'stratified':
            return self._stratified_sampling(mask, partial_mask, ignore_label)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _random_sampling(self, mask, partial_mask, ignore_label):
        """Random sampling across all pixels"""
        foreground_indices = np.argwhere(mask == 1)
        background_indices = np.argwhere(mask == 0)
        
        # Sample foreground pixels
        if len(foreground_indices) > 0:
            n_fg_samples = max(1, int(len(foreground_indices) * self.sampling_percentage))
            sampled_fg = foreground_indices[np.random.choice(len(foreground_indices), 
                                                           n_fg_samples, replace=False)]
            for y, x in sampled_fg:
                partial_mask[y, x] = 1
        
        # Sample background pixels
        if len(background_indices) > 0:
            n_bg_samples = max(1, int(len(background_indices) * self.sampling_percentage))
            sampled_bg = background_indices[np.random.choice(len(background_indices), 
                                                           n_bg_samples, replace=False)]
            for y, x in sampled_bg:
                partial_mask[y, x] = 0
                
        return partial_mask
    
    def _stratified_sampling(self, mask, partial_mask, ignore_label):
        """Stratified sampling ensuring balanced representation"""
        foreground_indices = np.argwhere(mask == 1)
        background_indices = np.argwhere(mask == 0)
        
        # Calculate balanced samples
        total_pixels = len(foreground_indices) + len(background_indices)
        target_samples = int(total_pixels * self.sampling_percentage)
        
        # Ensure at least some samples from each class
        min_samples_per_class = max(1, target_samples // 4)
        
        # Sample foreground
        if len(foreground_indices) > 0:
            n_fg = min(len(foreground_indices), max(min_samples_per_class, target_samples // 2))
            sampled_fg = foreground_indices[np.random.choice(len(foreground_indices), n_fg, replace=False)]
            for y, x in sampled_fg:
                partial_mask[y, x] = 1
        
        # Sample background
        if len(background_indices) > 0:
            n_bg = min(len(background_indices), max(min_samples_per_class, target_samples // 2))
            sampled_bg = background_indices[np.random.choice(len(background_indices), n_bg, replace=False)]
            for y, x in sampled_bg:
                partial_mask[y, x] = 0
                
        return partial_mask

def calculate_metrics(predictions, targets, ignore_index=255):
    """Calculate segmentation metrics"""
    # Convert predictions to class predictions
    pred_classes = torch.argmax(predictions, dim=1)
    
    # Create valid mask
    valid_mask = (targets != ignore_index)
    
    if not valid_mask.any():
        return {'accuracy': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Flatten and filter valid pixels
    pred_flat = pred_classes[valid_mask].cpu().numpy()
    target_flat = targets[valid_mask].cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(target_flat, pred_flat)
    iou = jaccard_score(target_flat, pred_flat, average='macro', zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda', save_path='best_model.pth'):
    """Training loop with validation"""
    
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, partial_masks, full_masks in train_bar:
            images = images.to(device)
            partial_masks = partial_masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, partial_masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, partial_masks, full_masks in val_bar:
                images = images.to(device)
                partial_masks = partial_masks.to(device)
                full_masks = full_masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, partial_masks)
                val_loss += loss.item()
                
                # Calculate metrics on full masks for validation
                metrics = calculate_metrics(outputs, full_masks)
                all_metrics.append(metrics)
                
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{metrics["iou"]:.4f}'})
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_metrics['iou'])
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if avg_metrics['iou'] > best_val_iou:
            best_val_iou = avg_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_val_iou,
            }, save_path)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val IoU: {avg_metrics["iou"]:.4f} (Best: {best_val_iou:.4f})')
        print(f'  Val Acc: {avg_metrics["accuracy"]:.4f}')
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'best_val_iou': best_val_iou
    }

def run_model_training(image_dir, mask_dir, sampling_percentage, sampling_strategy, 
                   batch_size=8, num_epochs=30, learning_rate=1e-4):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = RemoteSensingDataset(
        image_dir=os.path.join(image_dir, 'train'),
        mask_dir=os.path.join(mask_dir, 'train_labels'),
        sampling_percentage=sampling_percentage,
        sampling_strategy=sampling_strategy
    )
    
    val_dataset = RemoteSensingDataset(
        image_dir=os.path.join(image_dir, 'val'),
        mask_dir=os.path.join(mask_dir, 'val_labels'),
        sampling_percentage=sampling_percentage,
        sampling_strategy=sampling_strategy
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = SimpleUNet(in_channels=3, out_channels=2).to(device)
    criterion = PartialCrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Train model
    save_path = f'model_{sampling_strategy}_{sampling_percentage:.3f}.pth'
    results = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, num_epochs, device, save_path)
    
    return results, model

def visualize_results(image_dir, mask_dir, model, device, sampling_percentage, 
                     sampling_strategy, num_samples=4):
    """Visualize model predictions"""
    
    dataset = RemoteSensingDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        sampling_percentage=sampling_percentage,
        sampling_strategy=sampling_strategy
    )
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, partial_mask, full_mask = dataset[i]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert tensors to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            partial_mask_np = partial_mask.numpy()
            full_mask_np = full_mask.numpy()
            
            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(full_mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Show partial mask (ignore unlabeled regions)
            partial_vis = np.where(partial_mask_np == 255, 0.5, partial_mask_np)
            axes[i, 2].imshow(partial_vis, cmap='gray')
            axes[i, 2].set_title(f'Partial Labels ({sampling_percentage*100:.1f}%)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_mask, cmap='gray')
            axes[i, 3].set_title('Prediction')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results_{sampling_strategy}_{sampling_percentage:.3f}.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":


    from config import IMAGE_DIR


    pct = 0.20
    print(f"\nRunning model training with {pct*100:.1f}% sampling...")
    results, model = run_model_training(
        IMAGE_DIR, IMAGE_DIR, 
        sampling_percentage=pct,
        sampling_strategy='random',
        num_epochs=1
    )
    
    # Visualize results
    visualize_results(
        os.path.join(IMAGE_DIR, 'val'),
        os.path.join(IMAGE_DIR, 'val_labels'),
        model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        pct, 'random'
    )
    

  