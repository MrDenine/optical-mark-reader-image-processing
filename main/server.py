from fastapi import FastAPI
from fastapi import UploadFile,File
import uvicorn

from main.prediction import preprocess, read_image

app = FastAPI()

@app.get('/index')
async def hello_world():
    return 'Hello World'

@app.post('/predict')
async def predict_image(file:UploadFile = File(...)):
    pass

if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=3000)