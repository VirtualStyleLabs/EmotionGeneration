from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Generator
from mtcnn import MTCNN
import base64
from io import BytesIO


app = Flask(__name__)

# Initialize model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_number = 211000  # Default model number

# Reuse the utility functions from the original app.py
def crop_face_mtcnn(pil_image, output_size=(256, 256), margin=50):
    image = np.array(pil_image)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None
    x, y, width, height = faces[0]['box']
    x -= margin // 2
    y -= margin // 2
    width += margin
    height += margin
    x = max(0, x)
    y = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)
    face_crop = image[y:y2, x:x2]
    face_crop_pil = Image.fromarray(face_crop)
    face_crop_pil = face_crop_pil.resize(output_size, Image.BICUBIC)
    return face_crop_pil

def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(device:torch.device = "cpu"):
    c_trg_list = []
    c_dim = 7
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones((1,))*i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def load_model(model_number:int, device:torch.device = "cpu") -> None:
    global model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "stargan_rafd", "models", f"{model_number}-G.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = Generator(64,7,6)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

transform = []
transform.append(T.Resize(256))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

def convertImageToTensor(image : Image.Image, device:torch.device = "cpu") -> torch.Tensor:
    img = transform(image)
    img = img.unsqueeze(0).to(device)
    return img

def inference(img : torch.Tensor, emotion_number:int, device:torch.device = "cpu") -> Image.Image:
    start, end = 0, 8
    if(emotion_number != 7):
        start = emotion_number
        end = start + 1
    global model
    if model is None:
        load_model(model_number, device)
    with torch.inference_mode():    
        c_trg_list = create_labels(device)
        x_fake_list = []
        for c_trg in c_trg_list[start:end]:
            x_fake_list.append(model(img, c_trg))
        x_concat = torch.cat(x_fake_list,dim=3)
        image = denorm(x_concat.data.cpu())
        return T.ToPILImage()(image.squeeze(0))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        emotion_number = int(request.json['emotion'])
        
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_tensor = crop_face_mtcnn(image)
        
        # Convert to tensor once
        img_tensor = convertImageToTensor(img_tensor, device)
        # Generate the emotion transformations
        result_image = inference(img_tensor, emotion_number, device)
        
        # Convert result directly to base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': f'data:image/png;base64,{img_str}'})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Add debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
