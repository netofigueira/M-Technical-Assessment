import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from baseline_model import SimpleUNet, RemoteSensingDataset, PartialCrossEntropyLoss, calculate_metrics


def train_student(
    image_dir,
    pseudo_mask_dir,
    output_path='student_model.pth',
    class_weights=[1.0, 2.5],
    batch_size=8,
    num_epochs=15,
    learning_rate=1e-4,
    device='cuda'
):
    print(f"Device used: {device}")
    print(f"Training with {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    # Datasets
    train_dataset = RemoteSensingDataset(
        image_dir=os.path.join(image_dir, 'train'),
        mask_dir=pseudo_mask_dir,
        sampling_percentage=1.0,  
        sampling_strategy='random'  
    )
    val_dataset = RemoteSensingDataset(
        image_dir=os.path.join(image_dir, 'val'),
        mask_dir=os.path.join(image_dir, 'val_labels'),
        sampling_percentage=1.0,
        sampling_strategy='random'
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Modelo
    model = SimpleUNet(in_channels=3, out_channels=2).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss com pesos por classe
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = PartialCrossEntropyLoss(ignore_index=255, weight=weight_tensor)
    print(f"Class weights: {class_weights}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Ajustado

    best_val_iou = 0.0
    train_losses = []
    val_ious = []

    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)

    for epoch in range(num_epochs):
        # =========================
        # TRAINING PHASE
        # =========================
        model.train()
        total_loss = 0.0
        num_batches = 0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1:2d}/{num_epochs}] Train")
        for batch_idx, (images, partial_masks, _) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            partial_masks = partial_masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, partial_masks)
            loss.backward()
            
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # =========================
        # VALIDATION PHASE
        # =========================
        model.eval()
        val_metrics = []
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1:2d}/{num_epochs}] Val  ")
            for images, _, full_masks in val_bar:
                images = images.to(device, non_blocking=True)
                full_masks = full_masks.to(device, non_blocking=True)

                outputs = model(images)
                
                # Calcular loss de validação também
                val_batch_loss = criterion(outputs, full_masks)
                val_loss += val_batch_loss.item()
                
                # Calcular métricas
                metrics = calculate_metrics(outputs, full_masks)
                val_metrics.append(metrics)
                
                val_bar.set_postfix({
                    'val_loss': f'{val_batch_loss.item():.4f}',
                    'iou': f'{metrics["iou"]:.4f}'
                })

        # Calcular métricas médias
        avg_val_loss = val_loss / len(val_loader)
        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
        val_ious.append(avg_metrics['iou'])

        # =========================
        # LOGGING
        # =========================
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1:2d}/{num_epochs} RESULTS:")
        print(f"{'='*60}")
        print(f"Train Loss:     {avg_train_loss:.4f}")
        print(f"Val Loss:       {avg_val_loss:.4f}")
        print(f"Val IoU:        {avg_metrics['iou']:.4f}")
        print(f"Val Accuracy:   {avg_metrics['accuracy']:.4f}")
        print(f"Val Precision:  {avg_metrics['precision']:.4f}")
        print(f"Val Recall:     {avg_metrics['recall']:.4f}")
        print(f"Val F1:         {avg_metrics['f1']:.4f}")
        print(f"Learning Rate:  {optimizer.param_groups[0]['lr']:.2e}")
        
        # Salvar melhor modelo
        if avg_metrics['iou'] > best_val_iou:
            best_val_iou = avg_metrics['iou']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'train_losses': train_losses,
                'val_ious': val_ious,
                'class_weights': class_weights,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs
                }
            }, output_path)
            print(f"🎉 NEW BEST MODEL SAVED! IoU: {best_val_iou:.4f} -> {output_path}")
        
        scheduler.step()
        print()

    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Final model saved at: {output_path}")

    return {
        'best_val_iou': best_val_iou,
        'train_losses': train_losses,
        'val_ious': val_ious
    }


if __name__ == "__main__":


    from config import IMAGE_DIR, PSEUDO_LABELS_DIR

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


    # Executar treinamento
    results = train_student(
        image_dir=IMAGE_DIR,
        pseudo_mask_dir=PSEUDO_LABELS_DIR,
        output_path="student_model_balanced.pth",
        class_weights=[1.0, 3.0],  
        batch_size=8,
        num_epochs=12,  
        learning_rate=1e-4,
        device=device
    )
    
    print(f"\nFinal Results:")
    print(f"Best IoU achieved: {results['best_val_iou']:.4f}")