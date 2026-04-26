from fastapi import FastAPI
from inference import make_prediction
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(image: bytes):
    image = Image.open(io.BytesIO(image))
    return make_prediction(image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5911)