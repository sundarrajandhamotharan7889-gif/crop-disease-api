import io
import json
import os
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="Leaf Doctor Hybrid Model API")

# Enable CORS for Leaf Doctor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

# 1. Load classes.json
with open("classes.json", "r") as f:
    class_labels = json.load(f)

# 2. Define the exact Hybrid Model Architecture
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

# 3. Load Model Weights
model = HybridPlantClassifier(len(class_labels)).to(device)
model.load_state_dict(torch.load("plant_hybrid_model.pth", map_location=device))
model.eval()
print(f"Loaded Hybrid Model with {len(class_labels)} disease classes!")

# Image Transform
predict_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Leaf Doctor Hybrid Inference API",
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
    confidence = round(conf.item(), 4)

    # Format nicely (Tomato___Early_blight -> Tomato Early Blight)
    clean_label = raw_label.replace("___", " ").replace("_", " ")

    return {
        "disease": clean_label,
        "crop": crop,
        "confidence": confidence
    }
