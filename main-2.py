from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os, json, re

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

def convert_textfile_to_patient_json(txt_path):

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------- UPDATED PROMPT -------------------
    prompt = f"""
Extract patient medical data from the text below:
{text}

Return ONLY valid JSON with this structure:

{{
  "symptoms": "<short summary>",
  "diagnosis": ["<item1>", "<item2>"],
  "treatment": "<string|null>",
  "exercise": "<string|null>",
  "diet": "<string|null>",
  "mindSet": "<string|null>",
  "followUps": [],
  "books": [],
  "sleepFrom": "<string|null>",
  "sleepTo": "<string|null>",
  "appointment": "<string|null>",

  "supplementList": [
    {{
      "supplementName": "<string|null>"
    }}
  ],
    "supplementName" [],
  "bloodPressure": "<string|null>",
  "bloodPressureUnit": "<string|null>",
  "bodyTemperature": "<string|null>",
  "bodyTemperatureUnit": "<string|null>",
  "bodyHeartRate": "<string|null>",
  "bodyHeartRateUnit": "<string|null>",
  "respiratoryRate": "<string|null>",
  "weightKg": "<string|null>",
  "bmi": "<string|null>",

  "medicationList": [
    {{
      "medicationName": "<string|null>",
      "dosage": "<string|null>",
      "frequency": "<string|null>",
      "remarks": "<string|null>"
    }}
  ],

  "medicationName": [],
  "dosage": [],
  "frequency": [],
  "remarks": []
}}

Rules:
- Extract ALL real medicines.
- Return full medicationList objects.
- ALSO return separate arrays:
  medicationName[], dosage[], frequency[], remarks[]
- Arrays must align with medicationList index.
- If nothing found → empty arrays.
- Use camelCase keys.
"""
    # --------------------------------------------------

    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    res = model_ai.generate_content(prompt)

    raw = res.text.strip()
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    data = json.loads(cleaned)

    # Ensure medicationList exists
    meds = data.get("medicationList", [])
    if meds is None:
        meds = []
    data["medicationList"] = meds

    # Ensure separate arrays exist
    arr_keys = ["medicationName", "dosage", "frequency", "remarks"]
    for key in arr_keys:
        if key not in data or data[key] is None:
            data[key] = []

    # Auto-fill arrays from medicationList if arrays are empty
    if meds and len(data["medicationName"]) == 0:
        data["medicationName"] = [m.get("medicationName") for m in meds]
        data["dosage"] = [m.get("dosage") for m in meds]
        data["frequency"] = [m.get("frequency") for m in meds]
        data["remarks"] = [m.get("remarks") for m in meds]

    # Ensure supplementList exists
    sups = data.get("supplementList", [])
    if sups is None:
        sups = []
    data["supplementList"] = sups

    return data


@app.route("/convert-text", methods=["POST"])
def convert_text():
    try:
        txt_path = request.args.get("file")

        if not txt_path:
            return jsonify({"error": "file parameter missing"}), 400

        if txt_path.startswith("uploads/"):
            txt_path = os.path.join(os.getcwd(), txt_path)

        if not os.path.exists(txt_path):
            return jsonify({"error": f"File not found: {txt_path}"}), 404

        result = convert_textfile_to_patient_json(txt_path)

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
