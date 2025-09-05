import os
import json
import torch
import mysql.connector
from mysql.connector import pooling
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --------------------
# Config
# --------------------
CHECKPOINT_DIR = r"C:\Semester8\AI\stack-tags-api\checkpoint-9000-20250903T180242Z-1-001\checkpoint-9000"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",       # change for your MySQL setup
    "password": "Hashini@123",   # change for your MySQL setup
    "database": "stackoverflow",
    "pool_name": "mypool",
    "pool_size": 5
}

TRAIN_TAGS_JSON = r"C:\Semester8\AI\stack-tags-api\stackoverflow_tags.json"

# --------------------
# Flask setup
# --------------------
app = Flask(__name__)
CORS(app)

# --------------------
# DB Connection Pool
# --------------------
try:
    db_pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
    print("✅ MySQL connection pool created")
except Exception as e:
    print(f"❌ Error creating MySQL pool: {e}")
    exit(1)

# Ensure posts & new_tags tables exist
def init_db():
    conn = db_pool.get_connection()
    cursor = conn.cursor()

    # Main posts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            title TEXT NOT NULL,
            body MEDIUMTEXT NOT NULL,
            tags TEXT NULL,
            raw_output TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # New tags training data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS new_tags (
            Id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            Title TEXT NOT NULL,
            Body MEDIUMTEXT NOT NULL,
            Tags TEXT NOT NULL,
            CreationDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (Id),
            INDEX idx_creationdate (CreationDate)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    conn.commit()
    cursor.close()
    conn.close()

init_db()

# --------------------
# Load Model & Tokenizer
# --------------------
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(
    CHECKPOINT_DIR,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_safetensors=True
)
model.eval()

# --------------------
# Load training tags JSON
# --------------------
if os.path.exists(TRAIN_TAGS_JSON):
    with open(TRAIN_TAGS_JSON, "r", encoding="utf-8") as f:
        TRAIN_TAGS = set(json.load(f))
else:
    TRAIN_TAGS = set()
    print("⚠️ Warning: Training tags JSON not found, treating all manual tags as new.")

# --------------------
# Helper Functions
# --------------------
def format_tags(tags: list) -> str:
    """Convert list ['php','mysql'] -> '<php><mysql>' format"""
    return "".join([f"<{t}>" for t in tags])

def save_post(title: str, body: str, tags: list, raw_output: str):
    conn = db_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO posts (title, body, tags, raw_output)
        VALUES (%s, %s, %s, %s)
    """, (title, body, format_tags(tags), raw_output))
    conn.commit()
    post_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return post_id

def tag_exists_in_new_tags(tag: str) -> bool:
    """Check if a tag already exists in new_tags table"""
    conn = db_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM new_tags WHERE Tags LIKE %s", (f"%<{tag}>%",))
    (count,) = cursor.fetchone()
    cursor.close()
    conn.close()
    return count > 0

def save_new_tags(title: str, body: str, tags: list):
    """Save unseen tags into new_tags table for retraining."""
    # filter out tags already in TRAIN_TAGS or new_tags DB
    filtered = [
        t.strip().lower()
        for t in tags
        if t.strip().lower() not in TRAIN_TAGS and not tag_exists_in_new_tags(t.strip().lower())
    ]
    if not filtered:
        return []  # nothing new to save

    tags_str = format_tags(filtered)

    conn = db_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO new_tags (Title, Body, Tags)
        VALUES (%s, %s, %s)
    """, (title, body, tags_str))
    conn.commit()
    cursor.close()
    conn.close()
    return filtered

# --------------------
# API Routes
# --------------------
@app.route("/")
def home():
    return jsonify({"message": "T5 Flask API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    """Step 1: Predict tags"""
    try:
        data = request.get_json()
        if not data or "title" not in data or "body" not in data:
            return jsonify({"error": "Missing 'title' or 'body' field"}), 400

        title = data["title"]
        body = data["body"]

        input_text = f"{title} {body}"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Split raw output by comma/space
        predicted_tags = [t.strip().lower() for t in raw_output.replace(",", " ").split() if t.strip()]

        return jsonify({
            "title": title,
            "body": body,
            "predicted_tags": format_tags(predicted_tags),
            "raw_output": raw_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/finalize", methods=["POST"])
def finalize():
    """Step 2: Save post and insert into new_tags if unseen tags exist"""
    try:
        data = request.get_json()
        if not data or "title" not in data or "body" not in data or "tags" not in data:
            return jsonify({"error": "Missing 'title', 'body', or 'tags'"}), 400

        title = data["title"]
        body = data["body"]
        final_tags = [t.strip().lower() for t in data["tags"]]  # normalize tags
        raw_output = data.get("raw_output", "")

        # Save post always
        post_id = save_post(title, body, final_tags, raw_output)

        # Save only unseen tags into new_tags
        new_tags_saved = save_new_tags(title, body, final_tags)

        return jsonify({
            "message": "Post saved successfully",
            "post_id": post_id,
            "final_tags": format_tags(final_tags),
            "new_tags_saved": format_tags(new_tags_saved) if new_tags_saved else ""
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------
# Run Server
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
