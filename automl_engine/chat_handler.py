import ollama


class ChatHandler:
    @staticmethod
    def chat(message,context:str, model: str = "gemma3:4b") -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    },
                    {
                        "role": "user-hidden",
                        "context": context,
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