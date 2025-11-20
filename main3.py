import os
import tempfile
import re
import json
import traceback
from datetime import timedelta

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from pydub import AudioSegment
import whisper
import requests

from google import genai
from google.genai import types
from pyngrok import ngrok, conf


# ---------------- ENV / API CONFIG ----------------
load_dotenv()
GEN_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEN_API_KEY)

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------- FLASK APP ----------------
app = Flask(__name__)

# Uploads folder
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ---------------- Load Whisper ----------------
print("Loading Whisper model...")
model = whisper.load_model("small")
print("Whisper model loaded successfully!")


# =================================================================
#  AUDIO FUNCTIONS
# =================================================================

def preprocess_audio(file_path):
    try:
        print("Preprocessing audio...")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        processed_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio.export(processed_path, format="wav")

        return processed_path
    except Exception as e:
        print("Error in preprocess_audio:", e)
        raise


def transcribe_audio(audio_path, chunk_length_sec=60, language=None):
    try:
        print("Transcribing...")
        audio = AudioSegment.from_file(audio_path)
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
        print("Error in transcribe_audio:", e)
        raise


def correct_text_with_gemini(text, audio_path=None):
    try:
        prompt = (
            "Here is an audio transcription along with the original audio file.\n"
            "Fix grammar, spelling, and improve clarity. Compare with audio.\n"
            "Return ONLY the corrected text.\n\n"
            f"Transcription:\n{text}"
        )

        parts = [types.Part.from_text(text=prompt)]

        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts
        )
        return re.sub(r"\*+", "", (resp.text or "")).strip()

    except Exception as e:
        print("Error in Gemini correction:", e)
        return text


# =================================================================
#   MEDICAL TEXT → JSON FUNCTIONS
# =================================================================

def convert_text_to_patient_json(text):
    prompt = f"""
Extract patient medical data from the text below:

{text}

Return ONLY valid JSON with this structure:
[... your structure ...]
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    cleaned = resp.text.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)

    meds = data.get("medicationList", []) or []
    data["medicationList"] = meds

    if meds and len(data.get("medicationName", [])) == 0:
        data["medicationName"] = [m.get("medicationName") for m in meds]
        data["dosage"] = [m.get("dosage") for m in meds]
        data["frequency"] = [m.get("frequency") for m in meds]
        data["remarks"] = [m.get("remarks") for m in meds]

    return data


# =================================================================
#   ROUTES
# =================================================================

# -------------------- AUDIO --------------------
@app.route("/process", methods=["POST"])
def process_audio_route():
    try:
        data = request.get_json(silent=True) or {}

        lang = request.form.get("language") or data.get("language", "en")
        audio_url = request.form.get("audioUrl") or data.get("audioUrl")

        if "file" in request.files:
            file = request.files["file"]
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

        elif audio_url:
            save_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            os.system(f"curl -s '{audio_url}' -o {save_path}")

        else:
            return jsonify({"status": False, "error": "No file or audioUrl provided"}), 400

        processed = preprocess_audio(save_path)
        raw = transcribe_audio(processed, language=lang)
        clean = correct_text_with_gemini(raw, processed)

        return jsonify({
            "status": True,
            "results": {
                "raw_transcription": raw,
                "corrected_text": clean
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": False, "error": str(e)}), 500


# -------------------- MEDICAL TEXT --------------------
@app.route("/convert-text", methods=["POST"])
def convert_text():
    try:
        file_url = request.args.get("url")
        if not file_url:
            return jsonify({"error": "url parameter missing"}), 400

        response = requests.get(file_url, timeout=10)
        if response.status_code != 200:
            return jsonify({"error": "Cannot download file"}), 400

        text = response.text

        result = convert_text_to_patient_json(text)
        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
