import os
import re
import json
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv("environment.env")
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

# Request model
class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Call the model
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{
                "role": "user",
                "content": (
                    "Convert this message into a JSON object with fields: type, location, address, additional_details, severity. "
                    "Values of type can be: Fire, flood, earthquake, landslide, storm, other. "
                    "Value of severity can be: Low, Moderate, High, Critical. "
                    "If any field is missing, set it to null but do include all of these fields in your json response. Return ONLY valid JSON. "
                    + req.message
                )
            }],
        )

        # Extract raw response
        response_content = completion.choices[0].message.content.strip()

        # Extract JSON using regex
        match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Remove newlines, tabs, and extra spaces
            json_str = json_str.replace('\n', '').replace('\t', '').strip()
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON", "raw_response": json_str}
        else:
            return {"error": "No JSON found in model response", "raw_response": response_content}

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}

# [
#     "```json\n{\n  \"type\": \"fire\",\n  \"location\": \"Bray Road\",\n  \"address\": \"Bray Road\",\n  \"additional_details\": \"help\",\n  \"time\": null,\n  \"severity\": null\n}\n```\n\nThis JSON object includes all the specified fields, setting any missing fields to `null`."
# ]

    
