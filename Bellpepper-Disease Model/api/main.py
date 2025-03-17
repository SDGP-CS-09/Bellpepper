from fastapi import FastAPI, File, UploadFile, HTTPException  # Added HTTPException for error handling
import os
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Initialize FastAPI application
app = FastAPI()

# Load the pre-trained model with error handling
try:
    # Attempt to load the TensorFlow model for bell pepper disease prediction
    MODEL = tf.keras.models.load_model("bellpepper_model.h5")
except Exception as e:
    # Raise a runtime error if model loading fails
    raise RuntimeError(f"Failed to load model: {str(e)}")  # Added error handling

# Define class names for prediction outcomes
CLASS_NAMES = ["Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy"]

# Define a simple GET endpoint for health checking
@app.get("/ping")
async def ping():
    # Return a welcome message when accessing the /ping endpoint
    return "Welcome to bellpepper disease prediction API!"

# Function to convert uploaded file data to numpy array
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
        # Open the binary data as an image and convert to numpy array
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        # Raise an HTTP exception if image processing fails
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")  # Added error handling

# Define the prediction endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # Expect an uploaded file as input
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
        # Validate the uploaded file
        if not file:
            # Raise error if no file is provided
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.content_type.startswith('image/'):
            # Raise error if uploaded file is not an image
            raise HTTPException(status_code=400, detail="File must be an image")

        # Process the image and make prediction
        image = read_file_as_image(await file.read())  # Convert file to numpy array
        img_batch = np.expand_dims(image, 0)  # Add batch dimension for model input
        
        # Get predictions from the model
        predictions = MODEL.predict(img_batch)

        # Extract the predicted class and confidence score
        predicted_class = CLASS_NAMES
