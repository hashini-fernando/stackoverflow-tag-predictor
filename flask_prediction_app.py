"""
Flask Application for Stack Overflow Tag Prediction using Fine-tuned T5 LoRA Model

Usage:
1. Make sure your model files are in the correct directory
2. Install requirements: pip install flask transformers torch peft
3. Run: python app.py
4. Send POST requests to http://localhost:5000/predict

Expected model directory structure:
/models/
  ├── t5-lora-adapter/
  │   ├── adapter_config.json
  │   ├── adapter_model.bin
  │   └── ...
"""

import os
import re
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CONFIG = {
    'BASE_MODEL': 't5-small',
    'ADAPTER_DIR': 'C:\\Semester8\\AI\\stack-tags-api\\t5-lora-adapter',  # Adjust path as needed
    'MAX_INPUT_LENGTH': 512,
    'MAX_OUTPUT_LENGTH': 64,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'PORT': 5000,
    'HOST': '0.0.0.0',
    'DEBUG': False
}

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the fine-tuned LoRA model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(CONFIG['BASE_MODEL'])
        
        logger.info("Loading base model...")
        base_model = T5ForConditionalGeneration.from_pretrained(CONFIG['BASE_MODEL'])
        
        # Check if adapter exists
        if not os.path.exists(CONFIG['ADAPTER_DIR']):
            raise FileNotFoundError(f"LoRA adapter not found at {CONFIG['ADAPTER_DIR']}")
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, CONFIG['ADAPTER_DIR'])
        model.to(CONFIG['DEVICE'])
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {CONFIG['DEVICE']}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_question(title, body):
    """Preprocess question title and body for model input"""
    # Combine title and body
    combined_text = f"Title: {title.strip()} Body: {body.strip()}"
    
    # Clean and truncate
    combined_text = re.sub(r'\s+', ' ', combined_text)  # Remove extra whitespace
    combined_text = combined_text.strip()
    
    return combined_text

def postprocess_tags(raw_output):
    """Extract and clean predicted tags from model output"""
    if not raw_output:
        return []
    
    # Remove special tokens and clean
    cleaned = raw_output.replace('<pad>', '').replace('</s>', '').strip()
    
    # Extract tags using different patterns
    tags = []
    
    # Pattern 1: <tag1><tag2><tag3> format
    tag_matches = re.findall(r'<([^>]+)>', cleaned)
    if tag_matches:
        tags = [tag.strip() for tag in tag_matches if tag.strip()]
    
    # Pattern 2: comma-separated format
    elif ',' in cleaned:
        tags = [tag.strip() for tag in cleaned.split(',') if tag.strip()]
    
    # Pattern 3: space-separated format
    elif ' ' in cleaned:
        tags = [tag.strip() for tag in cleaned.split() if tag.strip()]
    
    # Pattern 4: single tag
    else:
        if cleaned:
            tags = [cleaned]
    
    # Clean tags (remove common artifacts)
    cleaned_tags = []
    for tag in tags:
        # Remove common artifacts
        tag = re.sub(r'>+$', '', tag)  # Remove trailing >
        tag = re.sub(r'^<+', '', tag)  # Remove leading <
        tag = tag.lower().strip()
        
        # Filter out empty or invalid tags
        if tag and len(tag) > 0 and not tag.isspace():
            cleaned_tags.append(tag)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in cleaned_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags[:5]  # Return top 5 tags max

def predict_tags(question_text, max_length=None):
    """Predict tags for a given question"""
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    if max_length is None:
        max_length = CONFIG['MAX_OUTPUT_LENGTH']
    
    try:
        # Tokenize input
        inputs = tokenizer(
            question_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=CONFIG['MAX_INPUT_LENGTH']
        ).to(CONFIG['DEVICE'])
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Postprocess tags
        predicted_tags = postprocess_tags(raw_prediction)
            
        return predicted_tags, raw_prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'device': CONFIG['DEVICE'],
        'base_model': CONFIG['BASE_MODEL']
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model is properly loaded before making predictions'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Please provide JSON data with title and body fields'
            }), 400
        
        # Extract required fields
        title = data.get('title', '').strip()
        body = data.get('body', '').strip()
        
        if not title and not body:
            return jsonify({
                'error': 'Missing required fields',
                'message': 'Please provide at least a title or body'
            }), 400
        
        # Preprocess input
        question_text = preprocess_question(title, body)
        
        # Make prediction
        predicted_tags, raw_output = predict_tags(question_text)
        
        # Prepare response
        response = {
            'predicted_tags': predicted_tags,
            'raw_output': raw_output,
            'input': {
                'title': title,
                'body': body,
                'processed_text': question_text[:200] + '...' if len(question_text) > 200 else question_text
            },
            'metadata': {
                'model': CONFIG['BASE_MODEL'],
                'device': CONFIG['DEVICE'],
                'num_tags': len(predicted_tags)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        questions = data.get('questions', [])
        
        if not questions or not isinstance(questions, list):
            return jsonify({
                'error': 'Invalid input',
                'message': 'Please provide a list of questions'
            }), 400
        
        results = []
        for i, question in enumerate(questions):
            try:
                title = question.get('title', '').strip()
                body = question.get('body', '').strip()
                
                if not title and not body:
                    results.append({
                        'index': i,
                        'error': 'Missing title and body'
                    })
                    continue
                
                question_text = preprocess_question(title, body)
                predicted_tags, raw_output = predict_tags(question_text)
                
                results.append({
                    'index': i,
                    'predicted_tags': predicted_tags,
                    'raw_output': raw_output,
                    'title': title,
                    'body': body[:100] + '...' if len(body) > 100 else body
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    instructions = {
        'message': 'Stack Overflow Tag Prediction API',
        'endpoints': {
            '/health': 'GET - Check API health status',
            '/predict': 'POST - Predict tags for a single question',
            '/predict_batch': 'POST - Predict tags for multiple questions'
        },
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': {
                'title': 'How to reverse a list in Python?',
                'body': 'I have a list [1, 2, 3, 4] and I want to reverse it to [4, 3, 2, 1]. What is the best way to do this in Python?'
            }
        },
        'model_info': {
            'base_model': CONFIG['BASE_MODEL'],
            'adapter_path': CONFIG['ADAPTER_DIR'],
            'device': CONFIG['DEVICE']
        }
    }
    return jsonify(instructions)

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Flask application...")
    logger.info(f"Configuration: {CONFIG}")
    
    if load_model():
        logger.info("Model loaded successfully. Starting server...")
        app.run(
            host=CONFIG['HOST'],
            port=CONFIG['PORT'],
            debug=CONFIG['DEBUG']
        )
    else:
        logger.error("Failed to load model. Please check your model files and configuration.")
        print("""
        SETUP INSTRUCTIONS:
        1. Make sure your LoRA adapter is saved at: ./models/t5-lora-adapter/
        2. Install required packages: pip install flask transformers torch peft flask-cors
        3. Update the ADAPTER_DIR path in the CONFIG section if needed
        4. Run: python app.py
        
        Expected directory structure:
        ./
        ├── app.py
        ├── models/
        │   └── t5-lora-adapter/
        │       ├── adapter_config.json
        │       ├── adapter_model.bin
        │       └── ...
        """)
