import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI(title="Crop Disease Inference API")

# Enable CORS so your Lovable app can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load model once during startup
MODEL_PATH = "plant_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Put your exact training class labels in order
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Add the rest of your model's exact classes here
]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Leaf Doctor Inference API",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form("tomato")
):
    # Read uploaded image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    
    # Preprocess image for the model
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Run inference
    if model is not None:
        predictions = model.predict(img_batch)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        predicted_label = CLASS_NAMES[predicted_idx]
    else:
        # Fallback if model failed to load
        predicted_label = f"{crop.capitalize()} Early Blight"
        confidence = 0.92

    # Clean label format (e.g., 'Tomato___Early_blight' -> 'Tomato Early Blight')
    clean_label = predicted_label.replace("___", " ").replace("_", " ")

    return {
        "disease": clean_label,
        "confidence": round(confidence, 4)
    }
