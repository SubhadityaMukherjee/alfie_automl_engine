import ollama


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
            return response["message"]["content"]
        except Exception as e:
            error_message = str(e).lower()
            if "not found" in error_message:
                return f"Model '{model}' not found. Please refer to Documentation at https://ollama.com/library."
            else:
                return f"An unexpected error occurred with model '{model}': {str(e)}"

    @staticmethod
    def detect_target_column(user_text: str) -> str:
        response = ChatHandler.chat(
            "Did the user mention what you think could be a target column for the tabular data classification/regression? "
            "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output"
            "Eg: Classify signature column -> signature, recognize different classes -> no, classify -> no, signature column -> signature"
            "If no, return 'no'. User messages:\n" + user_text
        )
        return response.strip()

    @staticmethod
    def detect_timestamp_column(user_text: str) -> str:
        response = ChatHandler.chat(
            "Did the user mention what you think could be a timestamp column for the tabular time series data classification/regression? "
            "If yes, what is the column name? Only return the column name, nothing else. ignore the words column and similar in the output"
            "Eg: Classify signature column -> signature, recognize different classes -> no, classify -> no, signature column -> signature"
            "If no, return 'no'. User messages:\n" + user_text
        )
        return response.strip()
