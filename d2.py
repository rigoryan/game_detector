import clip
import torch
import cv2
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

labels = ["a video game screen", "a desktop", "a browser window"]
text = clip.tokenize(labels).to(device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    pred_label = labels[probs.argmax()]
    print("CLIP Prediction:", pred_label)

    cv2.imshow("CLIP In-Game Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break