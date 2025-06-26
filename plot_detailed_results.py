import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from baseline_model import RemoteSensingDataset, calculate_metrics, SimpleUNet
from torch.utils.data import DataLoader
from config import IMAGE_DIR
import tqdm

def visualize_results(image_dir, mask_dir, model, device, sampling_percentage=1, 
                     sampling_strategy='random', num_samples=3, save_path=None):
    """Visualize model predictions with enhanced analysis"""
    
    dataset = RemoteSensingDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        sampling_percentage=sampling_percentage,
        sampling_strategy=sampling_strategy
    )
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    all_metrics = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            image, partial_mask, full_mask = dataset[idx]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Get prediction probabilities for confidence map
            probs = torch.softmax(output, dim=1)
            confidence = torch.max(probs, dim=1)[0].squeeze().cpu().numpy()
            
            # Convert tensors to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            partial_mask_np = partial_mask.numpy()
            full_mask_np = full_mask.numpy()
            
            # Calculate metrics for this sample
            sample_metrics = calculate_metrics(output, full_mask.unsqueeze(0).to(device))
            all_metrics.append(sample_metrics)
            
            # Plot 1: Original Image
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'Original Image #{idx}')
            axes[i, 0].axis('off')
            
            # Plot 2: Ground Truth
            axes[i, 1].imshow(full_mask_np, cmap='Reds', alpha=0.7)
            axes[i, 1].imshow(image_np, alpha=0.3)
            axes[i, 1].set_title('Ground Truth Overlay')
            axes[i, 1].axis('off')
            
            # Plot 3: Partial Labels (training data)
            partial_vis = np.where(partial_mask_np == 255, 0.7, partial_mask_np)
            axes[i, 2].imshow(partial_vis, cmap='Blues', alpha=0.7)
            axes[i, 2].imshow(image_np, alpha=0.3)
            axes[i, 2].set_title(f'Training Labels ({sampling_percentage*100:.1f}%)')
            axes[i, 2].axis('off')
            
            # Plot 4: Prediction
            axes[i, 3].imshow(pred_mask, cmap='Greens', alpha=0.7)
            axes[i, 3].imshow(image_np, alpha=0.3)
            axes[i, 3].set_title(f'Prediction\nIoU: {sample_metrics["iou"]:.3f}')
            axes[i, 3].axis('off')
            
            # Plot 5: Confidence Map
            im = axes[i, 4].imshow(confidence, cmap='viridis', vmin=0.5, vmax=1.0)
            axes[i, 4].set_title(f'Confidence Map\nAvg: {confidence.mean():.3f}')
            axes[i, 4].axis('off')
            
            # Add colorbar for confidence
            plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04, label='Confidence')
    
    # Calculate average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    # Add overall title with metrics
    fig.suptitle(f'Model Predictions Visualization', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    return avg_metrics


def visualize_training_progress(train_losses, val_ious, save_path=None):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot validation IoU
    ax2.plot(epochs, val_ious, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Validation IoU', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add best IoU annotation
    best_iou = max(val_ious)
    best_epoch = val_ious.index(best_iou) + 1
    ax2.annotate(f'Best: {best_iou:.3f}\n(Epoch {best_epoch})', 
                xy=(best_epoch, best_iou), xytext=(best_epoch+1, best_iou-0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress saved to: {save_path}")
    
    plt.show()


def analyze_model_performance(model, val_loader, device, save_path=None):
    """Detailed performance analysis"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_metrics = []
    
    print("Analyzing model performance...")
    
    with torch.no_grad():
        for images, _, full_masks in tqdm.tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            full_masks = full_masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Flatten for confusion matrix
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(full_masks.cpu().numpy().flatten())
            
            # Calculate batch metrics
            batch_metrics = calculate_metrics(outputs, full_masks)
            all_metrics.append(batch_metrics)
    
    # Calculate overall metrics
    overall_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    # Create detailed analysis plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Background', 'Road'])
    axes[0].set_yticklabels(['Background', 'Road'])
    
    # 2. Metrics Bar Plot
    metrics_names = ['Accuracy', 'IoU', 'Precision', 'Recall', 'F1']
    metrics_values = [overall_metrics['accuracy'], overall_metrics['iou'], 
                     overall_metrics['precision'], overall_metrics['recall'], overall_metrics['f1']]
    
    bars = axes[1].bar(metrics_names, metrics_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])
    axes[1].set_title('Overall Metrics')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    sample_ious = [m['iou'] for m in all_metrics]
    
    # 4. Performance Summary Table
    axes[2].axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f'{overall_metrics["accuracy"]:.4f}'],
        ['IoU (Jaccard)', f'{overall_metrics["iou"]:.4f}'],
        ['Precision', f'{overall_metrics["precision"]:.4f}'],
        ['Recall', f'{overall_metrics["recall"]:.4f}'],
        ['F1 Score', f'{overall_metrics["f1"]:.4f}'],
        ['', ''],
        ['Statistics', ''],
        ['Min IoU', f'{min(sample_ious):.4f}'],
        ['Max IoU', f'{max(sample_ious):.4f}'],
        ['Std IoU', f'{np.std(sample_ious):.4f}'],
    ]
    
    table = axes[2].table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[2].set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance analysis saved to: {save_path}")
    
    plt.show()
    
    return overall_metrics


def post_training_analysis(model_path, image_dir, model_name, val_dir, device='cuda'):
    """Complete post-training analysis"""
    
    
    # Load model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)
    model = SimpleUNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get training history
    train_losses = checkpoint.get('train_losses', [])
    val_ious = checkpoint.get('val_ious', [])
    best_iou = checkpoint.get('best_val_iou', 0)
    
    print(f"Best IoU achieved: {best_iou:.4f}")
    print(f"Training completed with {len(train_losses)} epochs")
    
    # 1. Visualize training progress
    if train_losses and val_ious:
        print("\n Plotting training progress...")
        visualize_training_progress(train_losses, val_ious, f'{model_name}_training_progress.png')
    
    #  Visualize predictions
    metrics = visualize_results(
        image_dir=os.path.join(image_dir, 'val'),
        mask_dir=os.path.join(image_dir, 'val_labels'),
        model=model,
        device=device,
        num_samples=4,
        save_path=f'{model_name}_predictions.png'
    )
    
    #  Detailed performance analysis
    val_dataset = RemoteSensingDataset(
        image_dir=os.path.join(image_dir, 'val'),
        mask_dir=os.path.join(image_dir, 'val_labels'),
        sampling_percentage=1.0,
        sampling_strategy='random'
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    detailed_metrics = analyze_model_performance(model, val_loader, device, f'{model_name}_performance_analysis.png')
    
    print(f"Final Performance Summary:")
    for metric, value in detailed_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return model, detailed_metrics




model, metrics = post_training_analysis(
    model_path="student_model_balanced.pth",
    image_dir=IMAGE_DIR,
    val_dir=None,  # já incluído no image_dir
    device='cuda',
    model_name='student',
)

visualize_results(
    image_dir=os.path.join(IMAGE_DIR, 'val'),
    mask_dir=os.path.join(IMAGE_DIR, 'val_labels'),
    model=model,  
    device='cuda',
    num_samples=6,
    save_path='student'
)