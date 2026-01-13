import torch
import os
import warnings
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# --- SETTINGS ---
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_NAME = "frcnn_missile.pth"
NUM_EPOCHS = 10  # How many epochs to run today
BATCH_SIZE = 2   # Change to 1 if your computer runs out of memory

print(f"--> Using device: {DEVICE}")

# --- DATASET DEFINITION ---
class MissileDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    _, x, y, w, h = map(float, line.split())
                    x1 = (x - w/2) * width
                    y1 = (y - h/2) * height
                    x2 = (x + w/2) * width
                    y2 = (y + h/2) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # Class 1 = Missile

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }
        return F.to_tensor(image), target

    def __len__(self):
        return len(self.images)

# --- INITIALIZE DATA LOADER ---
dataset = MissileDataset("dataset/train/images", "dataset/train/labels")
data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x))
)

# --- LOAD MODEL ---
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, 2) # Background + Missile
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- RESUME LOGIC ---
if os.path.exists(CHECKPOINT_NAME):
    print(f"--> Found {CHECKPOINT_NAME}. Resuming training...")
    model.load_state_dict(torch.load(CHECKPOINT_NAME, map_location=DEVICE))
else:
    print("--> No checkpoint found. Starting fresh.")

# --- TRAINING LOOP ---
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    # Progress bar setup
    loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
    
    for images, targets in loop:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    # Save progress after every epoch
    torch.save(model.state_dict(), CHECKPOINT_NAME)
    print(f"--> Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(data_loader):.4f}")

print("Training finished and model saved!")