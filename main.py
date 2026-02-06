from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google GenAI Client
# Note: This uses the new 'google-genai' library
try:
    client = genai.Client(api_key=API_KEY)
    
    chat_session = client.chats.create(
        model="gemini-flash-latest",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful visa processing helper assistant. Your role is to assist users with visa-related queries such as documents required, procedure to apply, etc. for different countries in a clear and concise manner. If any question is out of visa processing, politely inform them that you can only assist with visa-related queries."
        )
    )
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")

# Define the data format coming from the frontend
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(user_msg: UserMessage):
    if not user_msg.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Send message to Gemini
        response = chat_session.send_message(user_msg.message)
        return {"response": response.text}
    except Exception as e:
        print(f"Error during chat: {str(e)}")
        # Check if API Key is missing or invalid
        if "403" in str(e) or "API_KEY" in str(e):
             return {"response": "Error: Invalid or missing API Key."}
        return {"response": "I encountered an error processing your request."}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 allows external access; use 127.0.0.1 for local only
    uvicorn.run(app, host="127.0.0.1", port=8000)    