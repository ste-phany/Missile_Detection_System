# =========================================================================================================================
#                    Make sure to comment every type of testing before running one test output preferrred 
# ========================================================================================================================= 

#=========================
#    Image testing
#=========================

from ultralytics import YOLO

# 1. Load the model you trained (use the 'best.pt' or 'last.pt')
model = YOLO(r"runs\yolov11_missile\weights\best.pt")

# 2. Run inference on an image or video
results = model.predict(source="Test\missile_test_pic.jpg", save=True, conf=0.5)

# 3. Print where the result was saved
print(f"Results saved to: {results[0].save_dir}")


#=====================================
#    Video testing into screenshots
#=====================================

from ultralytics import YOLO
import cv2
import os

# 1. Load your model
model = YOLO(r"runs\yolov11_missile\weights\best.pt")

# 2. Create a folder for the snapshots if it doesn't exist
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# 3. Open the video
video_path = r"Test\Russia's Artillery Capabilities_ On target! BM-30 Smerch 9K58, Tornado-G, TOS1-A, BM-27 Uragan.mp4" 
results = model.predict(source=video_path, stream=True, conf=0.5)

count = 0
for r in results:
    # Check if any missiles were detected in this frame
    if len(r.boxes) > 0:
        for i, box in enumerate(r.boxes):
            # Only take a screenshot if confidence is high (e.g., > 0.75)
            confidence = box.conf[0]
            if confidence > 0.75:
                # Get the coordinates of the box [x1, y1, x2, y2]
                # These are the boundaries of the missile
                b = box.xyxy[0].cpu().numpy().astype(int)
                
                # Crop the original image to just the missile
                # r.orig_img is the frame before boxes were drawn
                crop = r.orig_img[b[1]:b[3], b[0]:b[2]]
                
                # Save the closeup
                count += 1
                file_name = f"snapshots/missile_detected_{count}.jpg"
                cv2.imwrite(file_name, crop)
                print(f"🚀 High-confidence detection! Saved closeup to {file_name}")

print("Video processing complete.")


#=======================================
#    Video testing Video output tests
#=======================================

from ultralytics import YOLO
import os

# 1. Load your trained weights
# Make sure this points to your BEST weights from the 'runs' folder
model = YOLO(r'runs\yolov11_missile\weights\best.pt')

# 2. Run detection on a video file
results = model.predict(
    source= r"Test\Russia's Artillery Capabilities_ On target! BM-30 Smerch 9K58, Tornado-G, TOS1-A, BM-27 Uragan.mp4", 
    conf=0.25,                       # Confidence threshold
    save=True,                       # This creates the output video
    device='cpu'                     # Use 'cuda' if you have a local GPU
)

print("✅ Video processing complete!")
print("📂 Check 'runs/detect/predict' for the output video file.")