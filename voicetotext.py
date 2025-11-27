import os
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv("environment.env")



client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY"),
)

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{"role": "user", "content": "Convert this message into a JSON object with fields: type, location, address, additional_details, time, severity. If any field is missing, set it to null but do include all of these fields in your json response. Return ONLY valid JSON." + req.message}],
        )
        # return {"reply": completion.choices[0].message["content"]}
        return {completion.choices[0].message.content}

    except Exception as e:
        print("ERROR:", e)

    