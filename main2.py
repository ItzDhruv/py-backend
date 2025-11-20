import os, tempfile, re, traceback
from datetime import timedelta
from flask import Flask, request, jsonify, render_template_string
from pydub import AudioSegment
import whisper
from google import genai
from google.genai import types
from pyngrok import ngrok, conf
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ---------------- API KEYS ----------------
GEN_API_KEY = "AIzaSyCRSWxAMNGLEi5n4KHwtgb06nbDwSwBqt4"
client = genai.Client(api_key=GEN_API_KEY)
# ---------------- Load Whisper ----------------
print("Loading Whisper model...")
model = whisper.load_model("small")
print("Whisper model loaded successfully!")
# ---------------- Supported Languages ----------------
LANGUAGES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
    "kn": "Kannada", "bn": "Bengali", "ur": "Urdu"
}
app = Flask(__name__)
# Store uploads inside your current project folder
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# ----------- Helper Functions -----------
def preprocess_audio(file_path):
    try:
        print("Preprocessing audio...")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        processed_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio.export(processed_path, format="wav")
        return processed_path
    except Exception as e:
        print(f"Error in preprocess_audio: {e}")
        raise
def transcribe_audio(audio_path, chunk_length_sec=60, language=None):
    try:
        print("Transcribing audio...")
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000
        print(f"Audio duration: {duration} seconds")
        chunks = [audio[i:i+chunk_length_sec*1000] for i in range(0, len(audio), chunk_length_sec*1000)]
        full_text = ""
        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                result = model.transcribe(temp_audio.name, language=language)
                start_time = str(timedelta(seconds=i * chunk_length_sec))
                full_text += f"[{start_time}]\n{result['text'].strip()}\n\n"
                os.remove(temp_audio.name)
        return full_text.strip()
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        raise
def correct_text_with_gemini(text, audio_path=None):
    try:
        prompt = (
            "Here is an audio transcription along with the original audio file.\n"
            "Please analyze both and:\n"
            "1) Correct grammar and spelling\n"
            "2) Improve formatting and readability\n"
            "3) Ensure accuracy by comparing with the audio\n"
            "4) Only give clean corrected Transcription and nothing else (header or summary)\n\n"
            f"Transcription:\n{text}"
        )
        parts = [types.Part.from_text(text=prompt)]
        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
        )
        return re.sub(r"\*+", "", (resp.text or "")).strip()
    except Exception as e:
        print(f"Error in correct_text_with_gemini: {e}")
        return text
# ----------- HTML Template -----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Speech-to-Text</title>
</head>
<body>
  <h1>Upload Audio</h1>
  <form method="POST" action="/process" enctype="multipart/form-data">
    <label>Audio File:</label>
    <input type="file" name="file" accept="audio/*" required><br><br>
    <label>Source Language (e.g., en, hi):</label>
    <input type="text" name="language" value="en"><br><br>
    <button type="submit">Transcribe</button>
  </form>
  {% if results %}
  <h2>Results</h2>
  <h3>Corrected Transcription:</h3>
  <pre>{{ results.corrected_text }}</pre>
  {% endif %}
  {% if error %}
  <h3 style="color:red;">Error: {{ error }}</h3>
  {% endif %}
</body>
</html>
"""
# ----------- Routes -----------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, results=None, error=None)
@app.route("/process", methods=["POST"])
def process_audio():
    try:
        lang = request.form.get("language") or (request.json.get("language") if request.is_json else "en")
        audio_url = request.form.get("audioUrl") or (request.json.get("audioUrl") if request.is_json else None)
        # CASE 1: audio uploaded as file
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": False, "error": "No file selected"}), 400
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)
        # CASE 2: audio provided via S3 URL (Node.js backend)
        elif audio_url:
            save_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            os.system(f"curl -s '{audio_url}' -o {save_path}")
        else:
            return jsonify({"status": False, "error": "No file or audioUrl provided"}), 400
        # ---- Preprocess and transcribe ----
        processed_path = preprocess_audio(save_path)
        transcribed_text = transcribe_audio(processed_path, language=lang)
        corrected_text = correct_text_with_gemini(transcribed_text, processed_path)
        # ---- Cleanup ----
        if os.path.exists(processed_path): os.remove(processed_path)
        if os.path.exists(save_path): os.remove(save_path)
        return jsonify({
            "status": True,
            "results": {
                "raw_transcription": transcribed_text,
                "corrected_text": corrected_text,
            }
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": False, "error": str(e)}), 500
# ---------------- ngrok setup ----------------
conf.get_default().auth_token = "34xO2n7Gg07celkDfXAM6uZu1mR_6YcmXkNvL5azb5gMwwGgs"
port = 5000
# public_url = ngrok.connect(port).public_url
# print("Ngrok URL:", public_url)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)