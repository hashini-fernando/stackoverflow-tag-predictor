Automatic Stack Overflow Tag Generator

AI-powered system that automatically predicts relevant tags for programming questions using transformer-based Natural Language Processing.

Overview

Tagging questions correctly on Stack Overflow is essential for organizing knowledge and helping developers find relevant solutions quickly. However, manual tagging can be inconsistent and time-consuming.

This project introduces an AI-powered tag prediction system that automatically generates relevant tags from a question's title and description. The system uses a fine-tuned T5-small model and serves predictions through a Flask API.

The application takes a programming question as input and generates multiple relevant tags.

Features

Automatic tag generation for developer questions

Transformer-based NLP model

Multi-label tag prediction

Web interface for user input

Fast inference using Flask backend

Extensible training pipeline

Tech Stack
Machine Learning

PyTorch

Hugging Face Transformers

T5-small

Backend

Flask

Database

MySQL

Development Tools

Google Colab

Python 3.10+

System Architecture
User Input (Title + Body)
        │
        ▼
Text Preprocessing
(Tokenization / Cleaning)
        │
        ▼
Fine-tuned T5 Model
(Tag Generation)
        │
        ▼
Postprocessing
(Tag Formatting)
        │
        ▼
Flask API Response
        │
        ▼
Frontend Display
Installation

Clone the repository:

git clone https://github.com/yourusername/stack-tags-api.git
cd stack-tags-api

Create a virtual environment:

python -m venv .venv

Activate the environment:

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Application

Start the Flask server:

python app_new.py

Server will run at:

http://localhost:5000

Model Training

The model was fine-tuned using the Hugging Face Trainer API.

Training involved:

Tokenizing Stack Overflow question text

Fine-tuning the T5 model

Evaluating using Precision, Recall, and F1-score

Challenges

Transformer models require GPU resources

Long questions require truncation

First inference can be slower due to model loading

Understanding code snippets remains challenging

Future Improvements

Deploy using Docker + Cloud hosting

Upgrade to T5-Base / T5-Large

Improve code snippet understanding

Add browser plugin for Stack Overflow

Continuous model retraining with new data

Resources

Project Video
https://drive.google.com/file/d/1uv_H4OCa7mjcY9Q6i9eZF73HGFGZ5SMG

Training Notebook
https://colab.research.google.com/drive/1bIdKWmgoMoZdZDIF2pu6JIQi4QeOuVkY

EDA Notebook
https://colab.research.google.com/drive/1_B7lHWnwMaC7Nji75GPJbofQjI5lFofb

Authors

Tharsi S.
Madhunicka M.
Sandarenu D.T
Fernando R.H.S

Department of Computer Engineering
Faculty of Engineering
University of Ruhuna
