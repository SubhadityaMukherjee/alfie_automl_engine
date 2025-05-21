# agent.py (with file processing support)
import asyncio
import re
from pathlib import Path
from typing import Optional, Union

import nest_asyncio
from ollama import AsyncClient
from docx import Document

nest_asyncio.apply()

class InteractiveProjectAgent:
    def __init__(self):
        self.client = AsyncClient()
        self.session_context = {}
        self.conversation_history = []

        self.instruction = """
Analyze the input and give guidance on what to do next. Based on your understanding, answer the following in free text format (no JSON required):

- Do you have enough information to act?
- What other info (if any) do you need?
- Does this involve file/data processing?
- Is the user trying to train a model?
- If so, do they want to use an existing model?
- If they want to process files, process them and return the results.
- Provide a next step or action plan.
"""

    def read_word_document(self, path: Path) -> str:
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            return f"⚠️ Error reading document: {e}"

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    async def ask_agent(self, user_input: str, context: Optional[str] = "") -> str:
        # Add user input to conversation history
        if context:
            full_prompt = f"{user_input.strip()}\n\n---\nAdditional context:\n{context.strip()}"
        else:
            full_prompt = user_input.strip()

        self.add_to_history("user", full_prompt)

        # Create message list including full history
        messages = [{"role": "system", "content": self.instruction}] + self.conversation_history

        # Get model response
        response = await self.client.chat(model="gemma3:4b", messages=messages)
        reply = response.message.content
        self.add_to_history("assistant", reply)
        return reply

    def parse_intent_from_text(self, text: str) -> dict:
        lower_text = text.lower()
        return {
            "enough_information": "enough information" in lower_text and "yes" in lower_text,
            "needs_files": "upload" in lower_text or "file" in lower_text,
            "wants_to_train": "train" in lower_text,
            "use_existing_model": "existing model" in lower_text,
            "extra_question": self.extract_question(text),
            "final_answer": text.strip(),
            "proccessed_files": [],
        }

    def extract_question(self, text: str) -> str:
        match = re.search(r"(?:(?:need|please provide|missing).+?\?)", text, re.IGNORECASE)
        return match.group(0) if match else ""

    async def process_user_input(
        self, user_input: str, file_path: Optional[Union[str, Path]] = None
    ) -> dict:
        # Step 1: Base LLM analysis
        initial_response = await self.ask_agent(user_input)
        parsed_intent = self.parse_intent_from_text(initial_response)

        # Step 2: Handle file if needed
        if parsed_intent["needs_files"] and file_path:
            file_text = self.read_word_document(Path(file_path))
            if not file_text.startswith("⚠️"):
                file_summary = await self.ask_agent(f"The user uploaded this file:\n\n{file_text[:2000]}")
                parsed_intent["file_summary"] = file_summary
            else:
                parsed_intent["file_error"] = file_text

        parsed_intent["initial_response"] = initial_response
        return parsed_intent

    def get_chat_history(self) -> str:
        return "\n".join(f"**{m['role'].capitalize()}**: {m['content']}" for m in self.conversation_history)
