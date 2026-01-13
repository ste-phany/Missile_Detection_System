import torch
import torchvision
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
from torchvision.transforms import functional as F

# 1. Setup Device
device = torch.device('cpu')

# 2. Dynamic Model Loader
def load_saved_model(checkpoint_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 1. We fix the Predictor to the shapes the error is demanding:
    # It wants cls_score to be 2
    # But it wants bbox_pred to stay at the default COCO size (91 classes)
    
    # First, set it to 91 to fix the '364' error
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91) 
    
    # Second, manually overwrite ONLY the classification layer to 2
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    
    # 2. Load the weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print("SUCCESS: Hybrid architecture synchronized and weights loaded!")
    return model

# Load it using the function
weights_path = r"Test/frcnn_missile.pth"
model = load_saved_model(weights_path)
model.to(device)
model.eval()

print("Model synced and loaded successfully!")

# ... (rest of your OpenCV code remains the same)

# 2. Setup Video Source and Output
video_path = r"Test\videoplayback (2).mp4" # Ensure this file is in your folder
cap = cv2.VideoCapture(video_path)

# Get video properties for saving
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print("Processing video... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR to RGB PIL Image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # 4. Draw Detections
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    
    for i in range(len(scores)):
        if scores[i] > 0.7:  # Confidence threshold
            box = boxes[i].cpu().numpy().astype(int)
            # Draw rectangle: (x_min, y_min), (x_max, y_max)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Add Label
            label = f"Missile: {scores[i]:.2f}"
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Show and Save
    out.write(frame)
    cv2.imshow('Missile Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Saved as 'detection_output.mp4'")