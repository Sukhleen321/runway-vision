import pathlib
import sys
import torch
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

"""
TO RUN
python app.py
python -m http.server 8080 

at http://localhost:8080/runway_frontend.html
"""
# Ensure workspace is on Python path for importing inference
sys.path.append(str(pathlib.Path(__file__).parent))
# pad_to_train_aspect is imported here too so the API uses the EXACT same
# preprocessing as inference.py — otherwise the two can silently drift apart,
# which is what caused the narrow/undersized lines you saw.
from inference import RunwayNet, IMG_SIZE, CHECKPOINT_PATH, pad_to_train_aspect

app = FastAPI(title='Runway Detection API')

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load model once at startup
model = RunwayNet()
state = torch.load(CHECKPOINT_PATH, map_location='cpu')
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state)
model.eval()

def decode_and_preprocess(image_bytes: bytes):
    """
    Decodes the uploaded image and prepares it for the model, using the
    SAME steps as inference.py: pad to the training aspect ratio (16:9)
    first, THEN stretch to 256x256. This keeps geometry consistent with
    what the model was trained on, no matter what shape the upload is.

    Returns the tensor for the model, the original (BGR) image for getting
    width/height, and the padding info needed to map predictions back to
    the original image's coordinates.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    orig_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise ValueError('Unable to decode image')

    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    padded, pad_left, pad_top = pad_to_train_aspect(rgb)
    ph, pw = padded.shape[:2]

    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    return tensor.unsqueeze(0), orig_img, pad_left, pad_top, pw, ph

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {'image/jpeg','image/png','image/bmp','image/webp'}:
        raise HTTPException(status_code=400, detail='Unsupported image type')
    content = await image.read()

    try:
        inp, orig_img, pad_left, pad_top, pw, ph = decode_and_preprocess(content)
    except ValueError:
        raise HTTPException(status_code=400, detail='Invalid image data')

    h, w = orig_img.shape[:2]

    with torch.no_grad():
        _, anchor_out = model(inp)

    anchors = anchor_out.squeeze(0).cpu().numpy()
    # Step 1: scale from [0,1] fractions to pixel coords on the PADDED canvas
    anchors[0::2] *= pw
    anchors[1::2] *= ph
    # Step 2: shift back into the ORIGINAL image's coordinate frame by
    # removing the padding offset (this is the part the old version skipped)
    anchors[0::2] -= pad_left
    anchors[1::2] -= pad_top

    return {'anchors': anchors.tolist(), 'width': w, 'height': h}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)