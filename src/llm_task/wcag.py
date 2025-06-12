import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import nest_asyncio
import pandas as pd
import streamlit as st

from src.chat_module.handler import ChatHandler
from src.file_processing.reader import FileHandler
from src.llm_task.tasks import LLMProcessingTask
from src.tabular_task.pipeline import AutoGluonTabularPipeline
from src.tabular_task.tasks import (
    TabularSupervisedClassificationTask,
    TabularSupervisedRegressionTask,
    TabularSupervisedTimeSeriesTask,
)
from src.pipeline_manager import PipelineManager

nest_asyncio.apply()


class WCAGPipeline:
    @staticmethod
    def _split_into_chunks(content, chunk_size):
        lines = content.splitlines()
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)  # +1 for newline

        chunks, line_ranges = [], []
        i = 0
        while i < len(content):
            end = i + chunk_size
            chunks.append(content[i:end])

            start_line = (
                next(j for j, offset in enumerate(line_offsets) if offset > i) - 1
            )
            end_line = (
                next(
                    (j for j, offset in enumerate(line_offsets) if offset > end),
                    len(lines),
                )
                - 1
            )
            line_ranges.append((start_line + 1, end_line + 1))  # 1-based

            i = end
        return chunks, line_ranges

    @staticmethod
    def _build_chunk_prompt(filename, chunk, idx, total, start_line, end_line):
        return f"""
You're a WCAG (Web Content Accessibility Guidelines) checker. Do not explain the code. Your job is only to evaluate the code against WCAG 2.1 guidelines.

Evaluate the following file named `{filename}`:
1. Score from 0–10 on how well it follows WCAG (0 = not at all, 10 = perfect). Use the format: Score: X (we will extract it using regex).
2. If score < 10, list specific improvements needed, with code suggestions in markdown.
3. Only evaluate the code below. Do not make assumptions beyond the content.

### Begin File Content (chunk {idx+1} of {total}, lines {start_line}–{end_line})
{chunk}
### End File Content
"""
    @staticmethod
    def _extract_wcag_score(response):
        match = re.search(r"\bScore[:\s]*([0-9](?:\.\d+)?)", response, re.IGNORECASE)
        return float(match.group(1)) if match else None

