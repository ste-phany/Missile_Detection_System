import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.transforms import functional as F

# 1. Setup Device
device = torch.device('cpu') # Use CPU since Colab GPU is gone

# 2. Recreate the Model Architecture
model = fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, 2)

# 3. Load your saved weights (The ones from Google Drive)
model.load_state_dict(torch.load(r"Test/frcnn_missile.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully! You can now use it for testing.")

# 4. Simple Test Function
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # This will print the boxes the model found
    print(prediction)

# Run a test on one of your images
predict(r"Dataset\test\images\0a9a13fb89716b15_jpg.rf.a3ee9748e6b4f501f5c128abfd6e04de.jpg")

