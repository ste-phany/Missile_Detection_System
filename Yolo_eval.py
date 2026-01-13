#Yolov11 Missile Detection Evaluation

from ultralytics import YOLO

def evaluate_model():
    model = YOLO(r"runs\yolov11_missile\weights\best.pt") 
    metrics = model.val()

    print("YOLOv11 Evaluation Results")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
