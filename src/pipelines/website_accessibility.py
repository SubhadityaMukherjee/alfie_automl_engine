import re
from typing import Any, Dict, List

import nest_asyncio
import streamlit as st

nest_asyncio.apply()


from src.chat_handler import ChatHandler
from src.file_handler import FileHandler
from src.pipelines.base import BasePipeline, PipelineRegistry


@PipelineRegistry.register("Website Accessibility")
class WebsiteAccesibilityPipeline(BasePipeline):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__(session_state, output_placeholder_ui_element)
        self.chunk_outputs: List[str] = []
        self.output_placeholder_ui_element = output_placeholder_ui_element
        self.initial_display_message = (
            "Hello, I will help you verify how accessible your website is"
        )

    @staticmethod
    def return_basic_prompt() -> str:
        return "You're a WCAG (Web Content Accessibility Guidelines) checker. Do not explain the code. Your job is only to evaluate the code against the most recent WCAG guidelines."

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

    def _build_chunk_prompt(self, filename, chunk, idx, total, start_line, end_line):
        return (
            self.return_basic_prompt()
            + f"""
Evaluate the following file named `{filename}`:
1. Score from 0–10 on how well it follows WCAG (0 = not at all, 10 = perfect). Use the format: Score: X (we will extract it using regex).
2. If score < 10, list specific improvements needed, with code suggestions in markdown.
3. Only evaluate the code below. Do not make assumptions beyond the content.

### Begin File Content (chunk {idx+1} of {total}, lines {start_line}–{end_line})
{chunk}
### End File Content
"""
        )

    @staticmethod
    def _extract_wcag_score(response):
        match = re.search(r"\bScore[:\s]*([0-9](?:\.\d+)?)", response, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def process_file(self, filename, content, chunk_size=3000):
        chunks, line_ranges = WebsiteAccesibilityPipeline._split_into_chunks(content, chunk_size)
        scores = []

        for i, (chunk, (start_line, end_line)) in enumerate(zip(chunks, line_ranges)):
            prompt = self._build_chunk_prompt(
                filename, chunk, i, len(chunks), start_line, end_line
            )
            response = ChatHandler.chat(prompt)
            score = WebsiteAccesibilityPipeline._extract_wcag_score(response)
            if score is not None:
                scores.append(score)

            yield {
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "response": response,
                "score": score,
                "num_chunks": len(chunks),
                "filename": filename,
            }

        # Final summary after all chunks
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        yield {
            "summary": True,
            "filename": filename,
            "num_chunks": len(chunks),
            "average_score": avg_score,
        }

    def main_flow(self, user_input: str, uploaded_files) -> Dict[str, Any] | None:
        self.session_state.add_message(
            role="assistant", content="Processing uploaded files"
        )
        if self.session_state.stop_requested == True:
            self.session_state.add_message(
                role="assistant", content="Processing stopped"
            )
            return

        try:
            files = FileHandler.read_each_file(uploaded_files)
        except Exception as e:
            self.session_state.add_message(
                role="assistant", content=f"Error reading files: {e}"
            )
            return

        for filename, content in files.items():
            self.session_state.add_message(
                role="assistant", content=f"Processing: {filename}"
            )
            for chunk_result in self.process_file(filename, content):
                if st.session_state.get("stop_requested", False):
                    self.session_state.add_message(
                        role="assistant", content="Processing stopped by user."
                    )
                    return

                if "summary" in chunk_result:
                    summary = f"✅ Finished analyzing `{chunk_result['filename']}`.\n\n**Average WCAG Score:** {chunk_result['average_score']}/10"
                    self.session_state.add_message(role="assistant", content=summary)
                    self.chunk_outputs.append(summary)
                else:
                    feedback = f"""
📄 **Chunk {chunk_result['chunk_index']+1}/{chunk_result['num_chunks']} of `{filename}` (lines {chunk_result['start_line']}-{chunk_result['end_line']})**  
{chunk_result['response']}
"""

                    self.session_state.add_message(role="assistant", content=feedback)
                    self.chunk_outputs.append(feedback)
                    self.output_placeholder_ui_element.markdown(
                        "\n\n---\n\n".join(self.chunk_outputs)
                    )
