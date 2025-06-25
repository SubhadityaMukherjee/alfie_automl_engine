from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class LLMProcessingTask(BaseModel):
    """
    LLM processing format
    """

    input_text: str = ""
    query: str = ""

    class Config:
        arbitrary_types_allowed = True
