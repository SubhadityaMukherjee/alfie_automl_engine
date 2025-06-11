import json
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from src.tasks import (LLMProcessingTask, TabularSupervisedClassificationTask,
                       TabularSupervisedRegressionTask,
                       TabularSupervisedTimeSeriesTask)


class ChatbotTaskSchema(BaseModel):
    task_type: Optional[str] = Field(
        default=None,
        description="What type of task is this? (e.g. classification, regression, etc.)",
    )
    need_to_train: Optional[str] = Field(
        default=None, description="Does the user need to train the model? (yes/no)"
    )
    train_file: Optional[str] = Field(default=None, description="Path to training file")
    test_file: Optional[str] = Field(default=None, description="Path to test file")
    target_column: Optional[str] = Field(
        default=None, description="Target column to predict"
    )
    external_handler: Optional[str] = Field(
        default=None,
        description="External handler like openml, huggingface or '' if not mentioned.",
    )

    def missing_fields(self) -> list[str]:
        return [
            name
            for name, field in type(self).model_fields.items()
            if getattr(self, name) in (None, "")
        ]

    def field_prompts(self) -> dict[str, str]:
        return {
            name: field.description or f"Please provide a value for '{name}'"
            for name, field in type(self).model_fields.items()
        }


class LLMClient:
    def __init__(self, backend: str, model: str, client: object):
        self.backend = backend
        self.model = model
        self.client = client

    async def chat(self, messages: List[dict]):
        if self.backend == "openai":
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content
        elif self.backend == "ollama":
            response = await self.client.chat(model=self.model, messages=messages)
            return response.message.content
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def check_model_exists(self):
        try:
            await self.chat(messages=[{"role": "user", "content": "Hello"}])
            return True
        except Exception:
            return False


class InteractiveAgent:
    def __init__(
        self, llm_client: LLMClient
    ):
        self.client = llm_client

        possible_tasks = [
            LLMProcessingTask,
            TabularSupervisedClassificationTask,
            TabularSupervisedRegressionTask,
            TabularSupervisedTimeSeriesTask,
        ]

        external_handlers = ["openml"]

        self.task_types = ", ".join([t.__name__ for t in possible_tasks])
        self.external_handlers = ", ".join(external_handlers)

        self.instruction = f"""
        You are a chatbot that needs to understand what the user wants to do. From their query, pick one of the following types of tasks.
        Return the JSON output:
        {{
        "task_type": ..,
        "need_to_train": ..,
        "train_file": ..,
        "test_file": ..,
        "target_column": ..,
        "external_handler": ..
        }}

        Schema: {ChatbotTaskSchema.__pydantic_fields__}

        Task type can be one of the following: {self.task_types}.
        All results should be of string type. If you do not know the answer, return an empty string.
        """

    async def get_initial_response(
        self, user_query: str, history: List[dict]
    ) -> Tuple[ChatbotTaskSchema, List[dict]]:
        history += [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_query},
        ]

        reply = await self.client.chat(history)
        reply_cleaned = reply.strip().replace("```", "").replace("json", "")
        data = json.loads(reply_cleaned)
        history.append({"role": "assistant", "content": reply})
        return ChatbotTaskSchema(**data), history

    async def complete_missing_fields(
        self, task: ChatbotTaskSchema, history: List[dict]
    ) -> ChatbotTaskSchema:
        missing = task.missing_fields()
        if not missing:
            return task

        prompts = task.field_prompts()

        for field in missing:
            question = prompts[field]
            user_answer = input(
                f"I still need the following information: {question}\n> "
            )
            history.append({"role": "user", "content": user_answer})

            followup = history + [
                {
                    "role": "system",
                    "content": f"You're helping complete the missing field '{field}'. Please respond with ONLY the value, not a full JSON.",
                }
            ]
            response = await self.client.chat(followup)
            answer = response.strip().replace("```", "").replace("json", "")

            try:
                parsed = json.loads(answer)
                if isinstance(parsed, dict) and field in parsed:
                    answer = parsed[field]
            except json.JSONDecodeError:
                pass

            setattr(task, field, answer)
            history.append({"role": "assistant", "content": answer})

        return task
