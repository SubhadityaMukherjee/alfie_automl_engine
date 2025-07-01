import re
from typing import Any, Dict, List

import nest_asyncio
import streamlit as st

nest_asyncio.apply()


import base64
from io import BytesIO

import requests
import textstat
from bs4 import BeautifulSoup
from ollama import Client
from PIL import Image
from textblob import TextBlob

from automl_engine.utils import render_template
from automl_engine.chat_handler import ChatHandler
from automl_engine.file_handler import FileHandler
from automl_engine.pipelines.models import BasePipeline

client = Client()


class ImageConverter:
    @staticmethod
    def to_base64(image_path_or_url):
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


class AltTextChecker:
    @staticmethod
    def check(image_url_or_path, alt_text, model="qwen2.5vl"):
        image_b64 = ImageConverter.to_base64(image_url_or_path)
        messages = [
            {
                "role": "system",
                "content": render_template("wcag_checker_default_prompt.txt"),
            },
            {"role": "user", "content": f"Alt text: {alt_text}"},
            {
                "role": "user",
                "content": render_template("image_alt_checker_prompt.txt"),
                "images": [image_b64],
            },
        ]
        response = client.chat(model=model, messages=messages)
        return response["message"]["content"]


class ReadabilityAnalyzer:
    METRICS = {
        "Flesh Reading Ease": textstat.flesch_reading_ease,
        # "Flesch Kincaid Grade": textstat.flesch_kincaid_grade,
        # "Smog Index": textstat.smog_index,
        # "Automated Readability Index": textstat.automated_readability_index,
        # "Dale Chall Readability Score": textstat.dale_chall_readability_score,
        "Difficult Words": textstat.difficult_words,
        "Lexicon Count": textstat.lexicon_count,
        "Avg Sentence Length": textstat.avg_sentence_length,
    }

    @staticmethod
    def apply_metric(metric, text):
        try:
            return metric(text)
        except Exception:
            return "N/A"

    @classmethod
    def analyze(cls, text):
        return {
            name: cls.apply_metric(metric, text) for name, metric in cls.METRICS.items()
        }


class ChunkProcessor:
    @staticmethod
    def split(content, chunk_size):
        lines = content.splitlines()
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)

        chunks, line_ranges, i = [], [], 0
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
            line_ranges.append((start_line + 1, end_line + 1))
            i = end
        return chunks, line_ranges

    @staticmethod
    def extract_score(response):
        match = re.search(
            r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response, re.IGNORECASE
        )
        return float(match.group(1)) if match else None


class ImageChunkEvaluator:
    @staticmethod
    def evaluate(chunk):
        matches = re.findall(r'<img[^>]+src="([^"]+)"[^>]*alt="([^"]+)"', chunk)
        results = []
        for src, alt in matches:
            try:
                result = AltTextChecker.check(src, alt)
                results.append(
                    {"src": src, "alt_text": alt, "ollama_evaluation": result}
                )
            except Exception as e:
                results.append({"src": src, "alt_text": alt, "error": str(e)})
        return results


class WebsiteAccesibilityPipeline(BasePipeline):
    """This pipeline is used for when the user wants to check the accessibility of a website. Eg queries: Check WCAG, Check website guidelines, Check website accessibility, Check alt text, check website, website accessibility"""
    def __init__(self, session_state, output_placeholder_ui_element):
        super().__init__(session_state, output_placeholder_ui_element)
        self.chunk_outputs = []
        self.enable_image_alt_text_checker = True

    def get_all_text(self, soup):
        lines = (line.strip() for line in soup.get_text().splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)
    
    @staticmethod
    def get_required_files():
        return {"sites": "file_upload_multi"}

    def analyze_readability(self, content):
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = self.get_all_text(soup)
        scores = ReadabilityAnalyzer.analyze(text)
        summary = (
            "### Readability scores for all the text in your website:\n\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in scores.items())
        )
        self.chunk_outputs.append(summary)
        self.session_state.add_message(role="assistant", content=summary)

    def process_file(self, filename, content, chunk_size=3000):
        chunks, ranges = ChunkProcessor.split(content, chunk_size)
        scores = []

        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges)):
            prompt = render_template(
                jinja_environment=self.session_state.jinja_environment,
                template_name="build_chunk_prompt.txt",
                filename=filename,
                chunk=chunk,
                idx=i,
                total=len(chunks),
                start_line=start,
                end_line=end,
            )
            response = ChatHandler.chat(prompt)
            score = ChunkProcessor.extract_score(response)
            if score:
                scores.append(score)

            image_feedback = ""
            if self.enable_image_alt_text_checker:
                evaluations = ImageChunkEvaluator.evaluate(chunk)
                if evaluations:
                    image_feedback = "\n".join(
                        f"- **Image**: {r['src']}\n  - **ALT**: `{r['alt_text']}`\n  - **Result**: {r.get('ollama_evaluation', r.get('error'))}"
                        for r in evaluations
                    )

            yield {
                "chunk_index": i,
                "start_line": start,
                "end_line": end,
                "response": response,
                "score": score,
                "num_chunks": len(chunks),
                "filename": filename,
                "image_feedback": image_feedback,
            }

        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        yield {
            "summary": True,
            "filename": filename,
            "num_chunks": len(chunks),
            "average_score": avg_score,
        }

    def main_flow(self, user_input, uploaded_files):
        self.session_state.add_message(
            role="assistant", content="Processing uploaded files"
        )
        if self.session_state.stop_requested:
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
            self.analyze_readability(content)
            self.output_placeholder_ui_element.markdown(
                "\n\n---\n\n".join(self.chunk_outputs)
            )

            for chunk_result in self.process_file(filename, content):
                if self.session_state.stop_requested:
                    self.session_state.add_message(
                        role="assistant", content="Processing stopped by user."
                    )
                    return

                message = render_template(
                    jinja_environment=self.session_state.jinja_environment,
                    template_name="chunk_result.txt",
                    **chunk_result,
                    # summary="summary" in chunk_result,
                )
                self.session_state.add_message(
                    role="assistant", content=message.strip()
                )
                self.chunk_outputs.append(message.strip())
                self.output_placeholder_ui_element.markdown(
                    "\n\n---\n\n".join(self.chunk_outputs)
                )
