from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import requests
import google.generativeai as genai
import pinecone
import re

app = Flask(__name__)

# Load the processors and models for different types of images
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)

# Gemini API client
gemini_api_key = "AIzaSyCC4RjhaS6Gzv4IihjpwGrulqFx5xyjb-A"  # Replace with your Gemini API key
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

# Pinecone initialization
pinecone_api_key = "pcsk_5ntaZX_PLFbigks34pksxck2sL4fm7imcKrL31E1Tr4HCos4WphG8jz5ZpHV4GL4QnjcZ1"  # Replace with your Pinecone API key
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to the index
index_name = "alt-text-context"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # Adjust dimension as needed
        metric="cosine",  # Use cosine similarity
        spec=ServerlessSpec(cloud="aws", region="gcp-starter")  # Use AWS as the cloud provider
    )

# Connect to the index
index = pc.Index(index_name)

# Validate image file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the homepage with the image upload form."""
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    """Generate a caption for the uploaded image and process it with Gemini API."""
    try:
        # Check if an image file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400

        # Open and convert the image to RGB format
        image = Image.open(file.stream).convert("RGB")

        # Determine the type of image based on a parameter or heuristic
        image_type = request.form.get('image_type', 'general').lower()

        # Use different models for different image types (placeholder for future implementation)
        if image_type == 'medical':
            processor = blip_processor
            model = blip_model
        elif image_type == 'sports':
            processor = blip_processor
            model = blip_model
        else:
            # Default to general image captioning model
            processor = blip_processor
            model = blip_model

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Run the model to generate the caption
        outputs = model.generate(**inputs)

        # Decode the caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Call Gemini API to process the caption
        response = model.generate_content(caption)
        gemini_response = response.text

        # Store the caption and its embedding in Pinecone
        embedding = model.embed_content(caption)['embedding'] # Generate embedding
        index.upsert([(file.filename, embedding)])  # Store in Pinecone

        # Return the results
        return jsonify({
            'caption': caption,
            'gemini_response': gemini_response,
            'message': 'Caption generated and stored successfully'
        })

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)