from fastapi import FastAPI
from fastapi import UploadFile,File
from image_processing_core import cropPerspective ,find_squares
from PIL import Image
import uvicorn
import uuid
import numpy as np
import cv2


    

app = FastAPI()
IMAGEDIR = "images/"
IMAGECROPDIR = "crop/"

@app.get('/')
async def hello_world():
    return 'services is online.'

@app.post('/predict')
async def predict_image(file:UploadFile = File(...)):
    
    # Get image and save file to images folder
    file.filename = f"{uuid.uuid4()}.jpg"
    contents= await file.read()
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
        
    numpy_array = cropPerspective(f"{IMAGEDIR}{file.filename}")
    im = Image.fromarray(numpy_array)
    im_crop = f"{IMAGECROPDIR}{uuid.uuid4()}.jpg"
    im.save(im_crop)
    find_squares(im_crop)
    
    return {"headers":file.headers ,"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=3000)