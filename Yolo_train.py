# # YOLOv11 Missile Detection Training
# # Portions of this code were assisted by ChatGPT (OpenAI, 2025)

# from ultralytics import YOLO

# def train_model():
#     # Load YOLOv11 pretrained model
#     model = YOLO("yolo11n.pt")  # nano model = faster for students

#     # Train the model
#     model.train(
#         data="dataset/data.yaml",
#         epochs=50, 
#         imgsz=640,
#         batch=16,
#         project="runs",
#         name="yolov11_missile"
#     )

# if __name__ == "__main__":
#     train_model()

from ultralytics import YOLO

def resume_training():
    # 1. Load the LAST saved checkpoint
    model = YOLO(r"runs\yolov11_missile\weights\last.pt")

    # 2. Resume training to the target epoch
    model.train(resume=True, epochs=50)

if __name__ == "__main__":
    resume_training()