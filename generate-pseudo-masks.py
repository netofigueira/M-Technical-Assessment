import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from baseline_model import RemoteSensingDataset  
from baseline_model import SimpleUNet 
from config import  TRAIN_DIR, TRAIN_LABELS_DIR, PSEUDO_LABELS_DIR


# Configs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = 'model_random_0.200.pth'
OUTPUT_MASK_DIR = PSEUDO_LABELS_DIR
CONFIDENCE_THRESHOLD = 0.7

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# load baseline (teacher) model
model = SimpleUNet(in_channels=3, out_channels=2)
checkpoint = torch.load(SAVED_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()


dataset = RemoteSensingDataset(
    image_dir=TRAIN_DIR,
    mask_dir=TRAIN_LABELS_DIR,
    sampling_percentage=1.0,
    sampling_strategy='random',  
)

for i in tqdm(range(len(dataset)), desc="Generating pseudo-labels"):
    image_tensor, _, _ = dataset[i]
    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_batch)
        probs = torch.softmax(output, dim=1)
        confidences, preds = torch.max(probs, dim=1) 

    pred_np = preds.squeeze().cpu().numpy()
    conf_np = confidences.squeeze().cpu().numpy()

    pseudo_mask = np.where(conf_np >= CONFIDENCE_THRESHOLD, pred_np, 255).astype(np.uint8)

    filename = dataset.filenames[i]
    save_path = os.path.join(OUTPUT_MASK_DIR, filename)
    ToPILImage(mode='L')(torch.tensor(pseudo_mask)).save(save_path)

