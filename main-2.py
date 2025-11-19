from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os, json, re, requests

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)


def convert_text_to_patient_json(text):

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
  "supplementName": [],

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

    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    res = model_ai.generate_content(prompt)

    raw = res.text.strip()
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    data = json.loads(cleaned)

    # Ensure medicationList exists
    meds = data.get("medicationList", []) or []
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
    sups = data.get("supplementList", []) or []
    data["supplementList"] = sups

    return data


@app.route("/convert-text", methods=["POST"])
def convert_text():
    try:
        file_url = request.args.get("url")

        if not file_url:
            return jsonify({"error": "url parameter missing"}), 400

        # Fetch remote text file
        response = requests.get(file_url, timeout=10)
        if response.status_code != 200:
            return jsonify({"error": "Cannot download file", "status": response.status_code}), 400

        text = response.text

        # Parse using Gemini
        result = convert_text_to_patient_json(text)

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
