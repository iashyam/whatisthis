import onnxruntime as ort 
import numpy as np
from utils.preprocessing import preprocess_image
from utils.labels import labels

session = ort.InferenceSession("BurrahMobileNet.onnx")

def make_prediction(image):
    image = preprocess_image(image).astype("float32")
    image = np.expand_dims(image, axis=0)
    y = session.run(None, {"input": image})[0]
    label = labels[y.argmax()]

    return label
