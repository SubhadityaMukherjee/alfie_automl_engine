from datetime import datetime
from typing import Optional

import ollama
from pydantic import BaseModel


class ChatHandler:
    @staticmethod
    def chat(message: str, model: str = "gemma3:4b") -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            error_message = str(e).lower()
            if "not found" in error_message:
                return f"Model '{model}' not found. Please refer to Documentation at https://ollama.com/library."
            else:
                return f"An unexpected error occurred with model '{model}': {str(e)}"


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None
