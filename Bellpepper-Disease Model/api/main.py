from fastapi import FastAPI, File, UploadFile, HTTPException  # Import FastAPI and modules for handling files and HTTP errors
import os  # For accessing environment variables like PORT
import uvicorn  # For running the FastAPI server
import numpy as np  # For numerical operations and image array manipulation
from io import BytesIO  # For handling binary data streams
from PIL import Image  # For opening and processing image files
import tensorflow as tf  # For loading and using the TensorFlow model

# Initialize the FastAPI application instance
app = FastAPI()

# Load the pre-trained model with error handling
try:
    MODEL = tf.keras.models.load_model("bellpepper_model.h5")  # Load the saved TensorFlow model from file
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")  # Raise an error if model loading fails

# List of class names representing possible bell pepper conditions
CLASS_NAMES = ["Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy"]

# Define a simple health check endpoint
@app.get("/ping")
async def ping():
    """Returns a welcome message to verify API is running"""
    return "Welcome to bellpepper disease prediction API!"  # Return a static string to confirm API is active

# Function to convert uploaded image data to a numpy array
def read_file_as_image(data) -> np.ndarray:
    """
    Convert binary image data to numpy array
    Args:
        data: Binary image data from the uploaded file
    Returns:
        np.ndarray: Image converted to a numpy array
    Raises:
        HTTPException: If image processing fails (e.g., corrupt or unreadable file)
    """
    try:
        image = np.array(Image.open(BytesIO(data)))  # Open image from binary data and convert to numpy array
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")  # Handle image processing errors

# Define the prediction endpoint for disease classification
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # Expect an uploaded file as input (required)
):
    """
    Predict bell pepper disease from uploaded image
    Args:
        file: Uploaded image file (multipart/form-data)
    Returns:
        dict: Prediction results with class and confidence
    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    try:
        # Validate that a file was provided
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")  # Return 400 if no file is sent
            
        # Validate that the file is an image based on content type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")  # Ensure it’s an image file
            
        # Convert the uploaded file to a numpy array
        image = read_file_as_image(await file.read())  # Read file content and process it
        
        # Prepare image for model prediction
        img_batch = np.expand_dims(image, 0)  # Add batch dimension for model input
        predictions = MODEL.predict(img_batch)  # Run the model to get prediction probabilities

        # Extract the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # Get class with highest probability
        confidence = np.max(predictions[0])  # Get the highest confidence score
        
        # Return the prediction result as a dictionary
        return {
            'class': predicted_class,  # Predicted disease or healthy state
            'confidence': float(confidence)  # Confidence score as a float
        }
    except HTTPException as e:
        raise e  # Re-raise client errors (e.g., 400)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")  # Handle server-side errors

# Main execution block to start the server
if __name__ == "__main__":
    """
    Start the FastAPI server
    Uses port from environment variable PORT or defaults to 8000
    """
    try:
        # Get the port number from environment variable or use default
        port = int(os.getenv("PORT", 8000))  # Convert PORT to integer or use 8000
        # Run the FastAPI app with uvicorn on all network interfaces
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ValueError as e:
        print(f"Invalid port number: {str(e)}")  # Handle invalid port configuration
    except Exception as e:
        print(f"Server failed to start: {str(e)}")  # Handle general startup errors
