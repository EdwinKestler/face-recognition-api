import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from pymongo import MongoClient
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()

# Get database choice from environment variable
db_choice = os.getenv("DB_CHOICE")

# Use CPU
device = torch.device('cpu')

# Initialize FaceNet model and MTCNN detector
mtcnn = MTCNN(device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize FastAPI
app = FastAPI()

mongo_connection_string = "mongodb://faceapiAdmin:faceapi!admin@localhost:27017"

print(db_choice, "mongo")

if db_choice == 'mongo':
    client = MongoClient(mongo_connection_string)
    db = client['fastapi']
    users = db['users']
else:
    raise ValueError("Invalid database choice")

def preprocess_image(image: Image.Image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Detect faces
    face, _ = mtcnn(img_array, return_prob=True)
    if face is not None:
        return face
    else:
        raise ValueError("No face detected in the image")

def generate_embedding(face: torch.Tensor):
    # Generate face embeddings
    embedding = facenet(face.unsqueeze(0)).detach().numpy()[0]
    return embedding

@app.post("/identify_face/")
async def identify_face(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    try:
        face = preprocess_image(image)
    except ValueError:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    embedding = generate_embedding(face)

    # Compare the embedding with embeddings in the database
    for user in users.find():
        for saved_embedding in user['embeddings']:
            score = cosine(np.array(saved_embedding), embedding)
            if score <= 0.4:  # consider it a match if score is <= 0.4
                return {"match": True, "user_id": user['user_id']}
    return {"match": False}

@app.post("/add_face/{user_id}")
async def add_face(user_id: str, file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Check if user_id exists, if not create, without this check the app will crash
    user = users.find_one({'user_id': user_id})
    if not user:
        user = {"user_id": user_id, "embeddings": []}
        users.insert_one(user)

    # Preprocess the image
    try:
        face = preprocess_image(image)
    except ValueError:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Generate embedding for the face
    embedding = generate_embedding(face)

    # Store the embedding in the MongoDB collection
    users.update_one(
        {'user_id': user_id},
        {'$push': {'embeddings': embedding.tolist()}},
    )

    return {"success": True}
