from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Ensure temp directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


@app.post("/transcribe/")
async def transcribe(
    file: UploadFile,
    language: str = Form(...),
    translate_to_english: bool = Form(False),
):
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save uploaded MP3 file
    file_path = f"{temp_dir}/{file.filename}"
    with open(file_path, "wb") as audio_file:
        content = await file.read()
        audio_file.write(content)
    # Send file to OpenAI for transcription
    with open(file_path, "rb") as audio:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        print("Hrvatski text: ", response.text)

    if translate_to_english:
        # Translate the transcription to English
        translated_text = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following {language} text to grammatically correct English:\n{response.text}",
                }
            ],
            max_tokens=1000,
            response_format="verbose_json",
        )
        print(translated_text.choices[0].message.content)
        return {"transcription": translated_text.choices[0].message.content}

    return {"transcription": response["text"]}
