import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
import time

# Use built-in webcam
cap = cv2.VideoCapture(0)

# Load a pretrained CNN
model = models.resnet18(pretrained=True)
model.eval()

# Fake 2-class classifier (placeholder logic)
def simple_classify(image_tensor):
    # Just a dummy threshold for pixel brightness
    mean_val = image_tensor.mean().item()
    return "IN-GAME" if mean_val < 0.4 else "NOT-IN-GAME"

# Preprocessing: resize and normalize
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

print("Starting webcam stream...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    img_tensor = transform(frame)

    label = simple_classify(img_tensor)

    # Overlay the result
    cv2.putText(frame, f"Detected: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("In-Game Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()