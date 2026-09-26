import os
import io
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import json

# Optimize CPU threads for Render free tier (prevents CPU throttling)
torch.set_num_threads(2)

app = FastAPI(title="Uzhavan Pro Hybrid Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES_URL = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/classes.json"
MODEL_URL = "https://github.com/sundarrajandhamotharan7889-gif/crop-disease-api/releases/download/v1.0/plant_hybrid_model.pth"

MODEL_PATH = "plant_hybrid_model.pth"
CLASSES_PATH = "classes.json"

# Download model & classes if not present
if not os.path.exists(CLASSES_PATH):
    print("Downloading classes.json...")
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_PATH)

if not os.path.exists(MODEL_PATH):
    print("Downloading plant_hybrid_model.pth...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_data = json.load(f)
    if isinstance(class_data, dict):
        class_names = [class_data[str(i)] for i in range(len(class_data))]
    else:
        class_names = class_data

print(f"Loaded {len(class_names)} classes.")

# Define Hybrid Model Architecture
class HybridModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        self.eff_feat = eff.features
        self.eff_pool = nn.AdaptiveAvgPool2d((1, 1))

        dense = models.densenet121(weights=None)
        self.dense_feat = dense.features
        self.dense_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f1 = torch.flatten(self.eff_pool(self.eff_feat(x)), 1)
        f2 = torch.flatten(self.dense_pool(self.dense_feat(x)), 1)
        out = torch.cat((f1, f2), dim=1)
        return self.classifier(out)

device = torch.device("cpu")
model = HybridModel(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Optimized transformation for 224x224 input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Crop mapping prefixes in PlantVillage dataset
CROP_PREFIXES = {
    "tomato": ["tomato"],
    "potato": ["potato"],
    "apple": ["apple"],
    "corn": ["corn"],
    "grapes": ["grape"],
    "rice": ["rice"],
    "cotton": ["cotton"],
}

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Uzhavan Pro Hybrid Inference API",
        "classes_count": len(class_names)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("tomato"),
    language: str = Form("en")
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Pre-resize if image is large to speed up transform
    if img.size[0] > 512 or img.size[1] > 512:
        img.thumbnail((512, 512))

    tensor = transform(img).unsqueeze(0).to(device)

    # 3-5x faster inference without autograd overhead
    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Crop-Aware Filtering: Only evaluate classes that belong to the user's selected crop!
    selected_crop = crop.lower().strip()
    valid_prefixes = CROP_PREFIXES.get(selected_crop, [selected_crop])
    
    matching_indices = [
        i for i, name in enumerate(class_names)
        if any(prefix in name.lower() for prefix in valid_prefixes)
    ]

    # If crop matches known classes, pick best among them
    if matching_indices:
        sub_probs = probabilities[matching_indices]
        best_sub_idx = torch.argmax(sub_probs).item()
        final_idx = matching_indices[best_sub_idx]
        confidence = float(probabilities[final_idx].item())
    else:
        final_idx = torch.argmax(probabilities).item()
        confidence = float(probabilities[final_idx].item())

    raw_label = class_names[final_idx]
    clean_label = raw_label.replace("___", " ").replace("_", " ")

    return {
        "disease": clean_label,
        "crop": crop,
        "confidence": round(confidence, 4),
        "raw_label": raw_label
    }
