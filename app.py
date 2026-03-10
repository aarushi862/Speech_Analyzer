from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline
import os

app = Flask(__name__)

# model load
model = whisper.load_model("small")
sentiment_pipeline = pipeline("sentiment-analysis")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    if "audio" not in request.files:
        return "No file uploaded"

    file = request.files["audio"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Provide an initial prompt to guide Whisper towards correct spelling of tricky nouns
    prompt = "BTech, Ben Hur, B.Tech."

    # speech to text
    result = model.transcribe(filepath, initial_prompt=prompt)
    text = result["text"]

    # sentiment
    sentiment = sentiment_pipeline(text)

    return render_template(
        "index.html",
        transcription=text,
        sentiment=sentiment[0]["label"],
        score=sentiment[0]["score"]
    )

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    if not text.strip():
        return jsonify({"sentiment": "N/A", "score": 0.0})

    sentiment = sentiment_pipeline(text)
    return jsonify({
        "sentiment": sentiment[0]["label"],
        "score": sentiment[0]["score"]
    })


if __name__ == "__main__":
    app.run(debug=True)
    