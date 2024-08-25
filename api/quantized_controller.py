from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import keras
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

quantized_tflite_path = '../tflite_quant_model.tflite'

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_path=quantized_tflite_path)
interpreter.allocate_tensors()

# If needed, you can get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


MODEL = keras.models.load_model("../models/3")
CLASS_NAMES =["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Welcome!  Himanshu Sharma"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Normalize the image to match the expected input data type (FLOAT32)
    img_batch = (np.expand_dims(image, 0) / 255.0).astype(np.float32)

    # Set input tensor for the TFLite model
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


def read_file_as_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)