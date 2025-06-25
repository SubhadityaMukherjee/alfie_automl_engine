import re
from typing import Any, Dict, List

import nest_asyncio
import streamlit as st

nest_asyncio.apply()


import base64
from io import BytesIO

import requests
from ollama import Client
from PIL import Image
from textblob import TextBlob
import textstat

from src.chat_handler import ChatHandler
from src.file_handler import FileHandler
from src.pipelines.base import BasePipeline, PipelineRegistry
from bs4 import BeautifulSoup

client = Client()


def image_to_base64(image_path_or_url):
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def check_alt_text_with_ollama(image_url_or_path, alt_text, model="qwen2.5vl"):
    image_b64 = image_to_base64(image_url_or_path)
    messages = [
        {
            "role": "system",
            "content": "You are a WCAG accessibility checker. Your job is to determine if the alt text meaningfully and accurately represents the image.",
        },
        {"role": "user", "content": f"Alt text: {alt_text}"},
        {
            "role": "user",
            "content": "Does this alt text correctly describe the image? Respond with 'Yes' or 'No' and give a short justification.",
            "images": [image_b64],
        },
    ]
    response = client.chat(model=model, messages=messages)
    return response["message"]["content"]


@PipelineRegistry.register("Website Accessibility")
class WebsiteAccesibilityPipeline(BasePipeline):
    def __init__(self, session_state, output_placeholder_ui_element) -> None:
        super().__init__(session_state, output_placeholder_ui_element)
        self.chunk_outputs: List[str] = []
        self.output_placeholder_ui_element = output_placeholder_ui_element
        self.initial_display_message = (
            "Hello, I will help you verify how accessible your website is"
        )
        self.enable_image_alt_text_checker = False
        self.dict_readability_metrics = {
            "Flesh Reading Ease": textstat.flesch_reading_ease,
            "Flesch Kincaid Grade": textstat.flesch_kincaid_grade,
            "Smog Index": textstat.smog_index,
            "Automated Readability Index": textstat.automated_readability_index,
            "Dale Chall Readability Score": textstat.dale_chall_readability_score,
            "Difficult Words": textstat.difficult_words,
            "Lexicon Count": textstat.lexicon_count,
            "Avg Sentence Length": textstat.avg_sentence_length,
        }

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
        chunks, line_ranges = WebsiteAccesibilityPipeline._split_into_chunks(
            content, chunk_size
        )
        scores = []

        for i, (chunk, (start_line, end_line)) in enumerate(zip(chunks, line_ranges)):
            prompt = self._build_chunk_prompt(
                filename, chunk, i, len(chunks), start_line, end_line
            )
            response = ChatHandler.chat(prompt)
            score = WebsiteAccesibilityPipeline._extract_wcag_score(response)
            if score is not None:
                scores.append(score)

            image_feedback = ""
            if self.enable_image_alt_text_checker:
                image_eval = self._evaluate_images_in_chunk_with_ollama(chunk)
                if image_eval:
                    image_feedback = "\n".join(
                        f"- **Image**: {r['src']}\n  - **ALT**: `{r['alt_text']}`\n  - **Result**: {r.get('ollama_evaluation', r.get('error'))}"
                        for r in image_eval
                    )

            yield {
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "response": response,
                "score": score,
                "num_chunks": len(chunks),
                "filename": filename,
                "image_feedback": image_feedback,
            }

        # Final summary after all chunks
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        yield {
            "summary": True,
            "filename": filename,
            "num_chunks": len(chunks),
            "average_score": avg_score,
        }

    def _evaluate_images_in_chunk_with_ollama(self, chunk: str) -> list[dict]:
        matches = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
        results = []
        for src, alt_text in matches:
            try:
                evaluation = check_alt_text_with_ollama(src, alt_text)
                results.append(
                    {
                        "src": src,
                        "alt_text": alt_text,
                        "ollama_evaluation": evaluation,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "src": src,
                        "alt_text": alt_text,
                        "error": str(e),
                    }
                )
        return results

    def apply_metric_to_text(self, metric, text):
        try:
            return metric(text)
        except Exception as e:
            return "N/A"

    def process_text_and_add_readability_scores(self, content):
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        all_text: str = self.get_all_text_in_html(soup)
        readability_scores_string: str = (
            "### Readability scores for all the text in your website:\n\n"
            + "\n".join(
                f"- **{name}**: {self.apply_metric_to_text(metric=metric, text=all_text)}"
                for name, metric in self.dict_readability_metrics.items()
            )
        )
        self.chunk_outputs.append(readability_scores_string)
        self.session_state.add_message(role="assistant", content=readability_scores_string)

    def get_all_text_in_html(self, soup):
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
        return text

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

            self.process_text_and_add_readability_scores(content)

            self.output_placeholder_ui_element.markdown(
                "\n\n---\n\n".join(self.chunk_outputs)
            )
            for chunk_result in self.process_file(filename, content):
                if self.session_state.stop_requested:
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
{chunk_result['response']} \n\n
"""

                    if chunk_result["image_feedback"]:
                        feedback += f"🖼**Image ALT Text Review**:\n{chunk_result['image_feedback']}"

                    self.session_state.add_message(role="assistant", content=feedback)
                    self.chunk_outputs.append(feedback)
                    self.output_placeholder_ui_element.markdown(
                        "\n\n---\n\n".join(self.chunk_outputs)
                    )

