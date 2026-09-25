import io
import json
import os
import urllib.request
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Download URLs from your GitHub Release
CLASSES_URL = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/classes.json"
MODEL_URL   = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/plant_hybrid_model.pth"

CLASSES_FILE = "classes.json"
MODEL_FILE   = "plant_hybrid_model.pth"

# Auto-download on startup if not present locally
if not os.path.exists(CLASSES_FILE):
    print("Downloading classes.json from GitHub Releases...")
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)
    print("classes.json downloaded successfully!")

if not os.path.exists(MODEL_FILE):
    print("Downloading plant_hybrid_model.pth from GitHub Releases (~90MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("plant_hybrid_model.pth downloaded successfully!")

# Load class labels
with open(CLASSES_FILE, "r") as f:
    class_labels = json.load(f)
if isinstance(class_labels, dict):
    class_labels = [class_labels[k] for k in sorted(class_labels, key=lambda x: int(x))]
print(f"Loaded {len(class_labels)} disease classes.")

# 2. Exact Hybrid Architecture matching your Colab training
class HybridPlantClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        dense = models.densenet121(weights=None)

        self.eff_feat = eff.features
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        self.dense_feat = dense.features
        self.dense_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = torch.flatten(self.eff_pool(self.eff_feat(x)), 1)
        f2 = torch.flatten(self.dense_pool(self.dense_feat(x)), 1)
        return self.classifier(torch.cat((f1, f2), dim=1))

# 3. Load model onto CPU
device = torch.device("cpu")
model = HybridPlantClassifier(num_classes=len(class_labels)).to(device)

ckpt = torch.load(MODEL_FILE, map_location=device)
state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"Hybrid Model loaded successfully with {len(class_labels)} classes!")

# 4. Standard Preprocessing
predict_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 5. FastAPI Application
app = FastAPI(title="Uzhavan 2.0 Deep Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Uzhavan 2.0 Hybrid Inference API",
        "classes_count": len(class_labels)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("tomato"),
    language: str = Form("en")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = predict_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    raw_label = class_labels[idx.item()]
    confidence = round(float(conf.item()), 4)

    # Format label cleanly (e.g., Tomato___Early_blight -> Early blight)
    clean_label = raw_label.replace("___", " ").replace("_", " ").strip()
    if clean_label.lower().startswith(crop.lower()):
        clean_label = clean_label[len(crop):].strip()

    return {
        "disease": clean_label or raw_label,
        "crop": crop,
        "confidence": confidence,
        "raw_label": raw_label
    }
