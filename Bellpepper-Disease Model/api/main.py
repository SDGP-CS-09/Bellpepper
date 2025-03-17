from fastapi import FastAPI, File, UploadFile, HTTPException  # Added HTTPException for error handling
import os
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load model with error handling
try:
    MODEL = tf.keras.models.load_model("bellpepper_model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")  # Added error handling

CLASS_NAMES = ["Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy"]

@app.get("/ping")
async def ping():
    return "Welcome to bellpepper disease prediction API!"

def read_file_as_image(data) -> np.ndarray:
    """
    Convert binary image data to numpy array
    Args:
        data: Binary image data
    Returns:
        np.ndarray: Image as numpy array
    Raises:
        HTTPException: If image processing fails
    """
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")  # Added error handling

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """
    Predict bell pepper disease from uploaded image
    Args:
        file: Uploaded image file
    Returns:
        dict: Prediction results with class and confidence
    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    try:
        # Validate file exists and is an image
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Process and predict
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")  # Added error handling

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8060))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ValueError as e:
        print(f"Invalid port number: {str(e)}")  # Added error handling
    except Exception as e:
        print(f"Server failed to start: {str(e)}")  # Added error handling
