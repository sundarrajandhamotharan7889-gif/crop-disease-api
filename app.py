import os, io, json, urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

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

torch.set_num_threads(2)

def download_if_missing():
    if not os.path.exists(CLASSES_PATH):
        urllib.request.urlretrieve(CLASSES_URL, CLASSES_PATH)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_if_missing()

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

class HybridModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        eff = models.efficientnet_b0(weights=None)
        dense = models.densenet121(weights=None)
        self.eff_feat = eff.features
        self.dense_feat = dense.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.eff_feat(x)
        f2 = self.dense_feat(x)
        p1 = nn.functional.adaptive_avg_pool2d(f1, (1, 1)).flatten(1)
        p2 = nn.functional.adaptive_avg_pool2d(f2, (1, 1)).flatten(1)
        return self.classifier(torch.cat([p1, p2], dim=1))

device = torch.device("cpu")
model = HybridModel(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CROP_PREFIXES = {
    "tomato": "Tomato",
    "potato": "Potato",
    "apple": "Apple",
    "corn": "Corn",
    "grapes": "Grape",
}

@app.get("/")
def health():
    return {"status": "online", "service": "Uzhavan Pro Hybrid Inference API", "classes_count": len(class_names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form(None), language: str = Form("en")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        
        # Crop-aware masking: prevent cross-crop errors
        if crop and crop.lower() in CROP_PREFIXES:
            target_prefix = CROP_PREFIXES[crop.lower()]
            mask = torch.full_like(logits, float("-inf"))
            matched_indices = [i for i, c in enumerate(class_names) if c.startswith(target_prefix)]
            if matched_indices:
                for idx in matched_indices:
                    mask[0, idx] = logits[0, idx]
                logits = mask

        probs = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        predicted_class = class_names[top_idx.item()]
        confidence = float(top_prob.item())

    return {
        "prediction": predicted_class,
        "disease": predicted_class,
        "label": predicted_class,
        "confidence": confidence,
    }
