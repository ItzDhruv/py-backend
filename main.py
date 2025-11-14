

from fastapi import FastAPI, UploadFile, File, HTTPException
import google.generativeai as genai
from dotenv import load_dotenv
import os, json, re

# Load ENV
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="TXT → Patient JSON API")


def extract_value_and_unit(text_value, default_unit=None):
    if not text_value or not isinstance(text_value, str):
        return None, default_unit

    text_value = text_value.strip()
    match = re.match(r"([0-9./]+)\s*([A-Za-z%°/ ]+)?", text_value)

    if not match:
        return text_value, default_unit

    number = match.group(1)
    unit = match.group(2).strip() if match.group(2) else None

    if not unit and default_unit:
        unit = default_unit

    return number, unit



def convert_textfile_to_patient_json(txt_path):

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    prompt = f"""
Extract patient medical data from this conversation text:

{text}

Return ONLY valid JSON matching this structure.

{{
  "name": "<name|null>",
  "age": <number|null>,
  "gender": "<male|female|other|null>",
  "address": "<address|null>",
  "phone": "<phone|null>",
  "symptoms": "<short summary>",

  "diagnosis": ["list"],

  "medications": [
    {{
      "name": "<medicine name|null>",
      "dosage": "<dosage|null>",
      "frequency": "<frequency|null>"
    }}
  ],

  "treatment": "<string|null>",
  "exercise": "<string|null>",
  "diet": "<string|null>",

  "mindSet": [],
  "followUps": [],
  "books": [],

  "sleepFrom": "<string|null>",
  "sleepTo": "<string|null>",
  "appointment": "<string|null>",

  "supplementList": [],

  "blood_pressure": "<value|null>",
  "blood_pressure_unit": "<unit|null>",

  "body_heartRate": "<value|null>",
  "body_heartRate_unit": "<unit|null>",

  "weight_kg": "<value|null>",
  "weight_kg_unit": "<unit|null>",

  "hba1c_percent": "<value|null>",
  "bsl_fasting": "<value|null>",
  "bsl_postprandial": "<value|null>",
  "bsl_random": "<value|null>",
  "insulin_fasting": "<value|null>",
  "insulin_postprandial": "<value|null>",
  "tsh_level": "<value|null>",
  "c_peptide_fasting": "<value|null>",
  "c_peptide_postprandial": "<value|null>",
  "creatinine_level": "<value|null>",

  "prescription": {{
    "medicationList": [
      {{
        "medicationName": "<medicine name only>"
      }}
    ],
    "fileKey": null
  }}
}}
"""

    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    res = model_ai.generate_content(prompt)

    raw = res.text.strip()
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    data = json.loads(cleaned)

    # Fix units
    data["blood_pressure"], data["blood_pressure_unit"] = extract_value_and_unit(
        data.get("blood_pressure"), default_unit="mmHg"
    )

    data["body_heartRate"], data["body_heartRate_unit"] = extract_value_and_unit(
        data.get("body_heartRate"), default_unit="beats/min"
    )

    data["weight_kg"], data["weight_kg_unit"] = extract_value_and_unit(
        data.get("weight_kg"), default_unit="kg"
    )

    # Final prescription format
    final_meds = []
    for m in data.get("medications", []):
        name = m.get("name")
        if name:
            final_meds.append({"medicationName": name})

    data["prescription"] = {
        "medicationList": final_meds,
        "fileKey": None
    }

    return data


@app.post("/convert-text")
async def convert_text(txt_file: UploadFile = File(...)):
    try:
        # os.makedirs("transcriptions", exist_ok=True)

        # temp_txt = f"transcriptions/{txt_file.filename}"
        # with open(temp_txt, "wb") as f:
        #     f.write(await txt_file.read())

        result = convert_textfile_to_patient_json(temp_txt)

        return {"status": "success", "patient_data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
