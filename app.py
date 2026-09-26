import io
import json
import os
import time
import urllib.request

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES_URL = (
    "https://github.com/"
    "sundarrajandhamotharan7889-gif/"
    "crop-disease-api/releases/download/v1.0/classes.json"
)

MODEL_URL = (
    "https://github.com/"
    "sundarrajandhamotharan7889-gif/"
    "crop-disease-api/releases/download/v1.0/plant_hybrid_model.pth"
)

CLASSES_FILE = "classes.json"
MODEL_FILE = "plant_hybrid_model.pth"

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Model input size
IMAGE_SIZE = 224

# CPU
device = torch.device("cpu")


# ============================================================
# PYTORCH CPU SETTINGS
# ============================================================

try:
    cpu_count = os.cpu_count() or 1

    torch.set_num_threads(
        max(1, min(4, cpu_count))
    )

    torch.set_num_interop_threads(1)

except RuntimeError:
    # PyTorch thread pools may already be initialized.
    pass


# ============================================================
# DOWNLOAD REQUIRED MODEL FILES
# ============================================================

def download_if_missing(
    url: str,
    filename: str,
    description: str
):
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return

    print(f"Downloading {description}...")

    urllib.request.urlretrieve(
        url,
        filename
    )

    print(
        f"{filename} downloaded successfully."
    )


download_if_missing(
    CLASSES_URL,
    CLASSES_FILE,
    "classes.json from GitHub Releases"
)

download_if_missing(
    MODEL_URL,
    MODEL_FILE,
    "plant_hybrid_model.pth (~90 MB)"
)


# ============================================================
# LOAD CLASS LABELS
# ============================================================

with open(
    CLASSES_FILE,
    "r",
    encoding="utf-8"
) as f:

    class_labels = json.load(f)


if isinstance(class_labels, dict):

    class_labels = [
        class_labels[k]
        for k in sorted(
            class_labels,
            key=lambda x: int(x)
        )
    ]


print(
    f"Loaded {len(class_labels)} disease classes."
)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridPlantClassifier(nn.Module):

    def __init__(
        self,
        num_classes
    ):
        super().__init__()

        # EfficientNet-B0
        eff = models.efficientnet_b0(
            weights=None
        )

        # DenseNet-121
        dense = models.densenet121(
            weights=None
        )

        self.eff_feat = eff.features

        self.eff_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.dense_feat = dense.features

        self.dense_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                2304,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        f1 = torch.flatten(
            self.eff_pool(
                self.eff_feat(x)
            ),
            1
        )

        f2 = torch.flatten(
            self.dense_pool(
                self.dense_feat(x)
            ),
            1
        )

        combined = torch.cat(
            (f1, f2),
            dim=1
        )

        return self.classifier(
            combined
        )


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Creating hybrid model...")

model = HybridPlantClassifier(
    num_classes=len(class_labels)
).to(device)


print("Loading model weights...")


ckpt = torch.load(
    MODEL_FILE,
    map_location=device
)


if hasattr(
    ckpt,
    "state_dict"
):

    state_dict = ckpt.state_dict()

else:

    state_dict = ckpt


missing_keys, unexpected_keys = (
    model.load_state_dict(
        state_dict,
        strict=False
    )
)


if missing_keys:

    print(
        "WARNING: Missing model keys:"
    )

    print(
        missing_keys
    )


if unexpected_keys:

    print(
        "WARNING: Unexpected model keys:"
    )

    print(
        unexpected_keys
    )


model.eval()


print(
    f"Hybrid model loaded successfully "
    f"with {len(class_labels)} classes."
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

predict_tf = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# MODEL WARM-UP
# ============================================================

def warmup_model():

    print(
        "Warming up model..."
    )

    dummy_image = Image.new(
        "RGB",
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        (
            128,
            128,
            128
        )
    )

    tensor = (
        predict_tf(
            dummy_image
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        model(tensor)


    print(
        "Model warm-up complete."
    )


try:

    warmup_model()

except Exception as e:

    print(
        f"Model warm-up failed: {e}"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Uzhavan 2.0 Deep Learning API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "status": "online",

        "service":
            "Uzhavan 2.0 Hybrid Inference API",

        "classes_count":
            len(class_labels),

        "device":
            str(device),

        "image_size":
            IMAGE_SIZE
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded":
            model is not None,

        "classes_count":
            len(class_labels)
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...),

    crop: str = Form(
        "tomato"
    ),

    language: str = Form(
        "en"
    )
):

    request_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file:

        raise HTTPException(
            status_code=400,
            detail="No image file was provided."
        )


    if (
        file.content_type
        and
        not file.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type: "
                f"{file.content_type}"
            )
        )


    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:

        contents = await file.read()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read uploaded image: "
                f"{str(e)}"
            )
        )


    upload_size = len(
        contents
    )


    if upload_size == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )


    if upload_size > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 10 MB."
            )
        )


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    decode_start = (
        time.perf_counter()
    )


    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file "
                "is not a valid image."
            )
        )


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode image: "
                f"{str(e)}"
            )
        )


    decode_time = (
        time.perf_counter()
        - decode_start
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    preprocess_start = (
        time.perf_counter()
    )


    tensor = (
        predict_tf(image)
        .unsqueeze(0)
        .to(device)
    )


    preprocess_time = (
        time.perf_counter()
        - preprocess_start
    )


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )


    try:

        with torch.inference_mode():

            outputs = model(
                tensor
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            conf, idx = torch.max(
                probs,
                dim=1
            )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                f"{str(e)}"
            )
        )


    inference_time = (
        time.perf_counter()
        - inference_start
    )


    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    class_index = idx.item()


    if (
        class_index < 0
        or
        class_index >= len(class_labels)
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid model class index: "
                f"{class_index}"
            )
        )


    raw_label = class_labels[
        class_index
    ]


    confidence = round(
        float(
            conf.item()
        ),
        4
    )


    # --------------------------------------------------------
    # Clean disease label
    # --------------------------------------------------------

    clean_label = (
        str(raw_label)
        .replace(
            "___",
            " "
        )
        .replace(
            "_",
            " "
        )
        .strip()
    )


    crop_clean = (
        str(crop)
        .strip()
    )


    if crop_clean:

        prefix = (
            crop_clean.lower()
        )

        if clean_label.lower().startswith(
            prefix
        ):

            clean_label = clean_label[
                len(crop_clean):
            ].strip()


    if not clean_label:

        clean_label = str(
            raw_label
        )


    # --------------------------------------------------------
    # Total processing time
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - request_start
    )


    # --------------------------------------------------------
    # Server log
    # --------------------------------------------------------

    print(

        f"[PREDICT] "
        f"crop={crop_clean} "
        f"label={clean_label} "
        f"confidence={confidence} "
        f"upload={upload_size / 1024:.1f}KB "
        f"decode={decode_time:.3f}s "
        f"preprocess={preprocess_time:.3f}s "
        f"inference={inference_time:.3f}s "
        f"total={total_time:.3f}s"
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "disease":
            clean_label,

        "crop":
            crop,

        "confidence":
            confidence,

        "raw_label":
            raw_label,

        "timing": {

            "decode_seconds":
                round(
                    decode_time,
                    3
                ),

            "preprocess_seconds":
                round(
                    preprocess_time,
                    3
                ),

            "inference_seconds":
                round(
                    inference_time,
                    3
                ),

            "total_seconds":
                round(
                    total_time,
                    3
                )
        }
    }
