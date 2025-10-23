from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"runs\classify\train11\weights\best.pt")

# Define path to the image file
source = r""

# Run inference on the source
results = model(source)  # list of Results objects

print(results)