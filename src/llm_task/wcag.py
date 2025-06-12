import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
import pandas as pd
import streamlit as st

from src.chat_module.handler import ChatHandler
from src.file_processing.reader import FileHandler

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

    @staticmethod
    def process_uploaded_files(uploaded_files):
        try:
            _, file_paths = FileHandler.aggregate_file_content(uploaded_files)
            return {"status": "success", "file_paths": file_paths}
        except Exception as e:
            return {"status": "error", "message": f"Error processing files: {e}"}

    @staticmethod
    def process_file(filename, content, chunk_size=3000):
        chunks, line_ranges = WCAGPipeline._split_into_chunks(content, chunk_size)
        outputs, scores = [], []

        for i, (chunk, (start_line, end_line)) in enumerate(zip(chunks, line_ranges)):
            prompt = WCAGPipeline._build_chunk_prompt(
                filename, chunk, i, len(chunks), start_line, end_line
            )
            response = ChatHandler.chat(prompt)
            score = WCAGPipeline._extract_wcag_score(response)
            if score is not None:
                scores.append(score)

            feedback = {
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "response": response,
            }
            outputs.append(feedback)

        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "average_score": avg_score,
            "feedback": outputs,
        }

    @staticmethod
    def handle_user_input(user_input: str, uploaded_files) -> Dict[str, Any]:
        if not uploaded_files:
            return {"status": "error", "message": "No files uploaded"}

        try:
            files = FileHandler.read_each_file(uploaded_files)
        except Exception as e:
            return {"status": "error", "message": f"Error reading files: {e}"}

        results: List[Dict[str, Any]] = []
        for filename, content in files.items():
            result = WCAGPipeline.process_file(filename, content)
            results.append(result)

        return {"status": "success", "results": results}
