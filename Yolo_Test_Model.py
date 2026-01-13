from ultralytics import YOLO

# 1. Load the model you trained (use the 'best.pt' or 'last.pt')
model = YOLO(r"runs\yolov11_missile\weights\best.pt")

# 2. Run inference on an image or video
results = model.predict(source="test_image.jpg", save=True, conf=0.5)

# 3. Print where the result was saved
print(f"Results saved to: {results[0].save_dir}")