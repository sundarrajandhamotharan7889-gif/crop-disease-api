import os
import urllib.request
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------------------------
# 1. Download URLs from GitHub Releases
# ---------------------------------------------------------------------------
CLASSES_URL = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/classes.json"
MODEL_URL = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/plant_hybrid_model.pth"

CLASSES_FILE = "classes.json"
MODEL_FILE = "plant_hybrid_model.pth"

# Auto-download on startup if not present locally
if not os.path.exists(CLASSES_FILE):
    print("Downloading classes.json from GitHub Releases...")
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)
    print("classes.json downloaded successfully!")

if not os.path.exists(MODEL_FILE):
    print("Downloading plant_hybrid_model.pth from GitHub Releases (~90MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("plant_hybrid_model.pth downloaded successfully!")

with open(CLASSES_FILE, "r") as f:
    CLASS_NAMES = json.load(f)
print(f"Loaded {len(CLASS_NAMES)} disease classes.")

# ---------------------------------------------------------------------------
# 2. Hybrid Model Architecture (EfficientNet-B0 + DenseNet-121)
# ---------------------------------------------------------------------------
class HybridPlantClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HybridPlantClassifier, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None)
        self.densenet = models.densenet121(weights=None)

        eff_in = self.effnet.classifier[1].in_features      # 1280
        dense_in = self.densenet.classifier.in_features     # 1024

        self.effnet.classifier = nn.Identity()
        self.densenet.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(eff_in + dense_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.effnet(x)
        f2 = self.densenet(x)
        features = torch.cat((f1, f2), dim=1)
        return self.classifier(features)

# Load model onto CPU (lightweight for Render Free Tier)
device = torch.device("cpu")
model = HybridPlantClassifier(num_classes=len(CLASS_NAMES))
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("PyTorch Hybrid Model loaded and ready on CPU!")

# Standard PlantVillage preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# 3. FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Uzhavan 2.0 Deep Learning Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "app": "Uzhavan 2.0 Inference Engine",
        "architecture": "Hybrid EfficientNet-B0 + DenseNet-121",
        "classes_count": len(CLASS_NAMES)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("tomato"),
    language: str = Form("en")
):
    image = Image.open(file.file).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    raw_label = CLASS_NAMES[top_idx.item()]
    
    # Format label cleanly (e.g., Tomato___Early_blight -> Early blight)
    clean_label = raw_label.replace("___", " ").replace("_", " ").strip()
    if clean_label.lower().startswith(crop.lower()):
        clean_label = clean_label[len(crop):].strip()

    return {
        "disease": clean_label or raw_label,
        "crop": crop,
        "confidence": round(float(top_prob.item()), 4),
        "raw_label": raw_label
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
