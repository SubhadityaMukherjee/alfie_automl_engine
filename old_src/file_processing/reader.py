import mimetypes
import tempfile
from pathlib import Path

import pandas as pd
from docx import Document


class FileHandler:
    @staticmethod
    def read_word_document(path: Path) -> str:
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            return f"⚠️ Error reading document: {e}"

    # @staticmethod
    # def csv_summary(path: Path) -> str:
    #     try:
    #         df = pd.read_csv(path)
    #         return str(df.describe())
    #     except Exception as e:
    #         return f"⚠️ Error reading document: {e}"

    @staticmethod
    def read_file_content(file_path: Path, mime: str) -> str:
        try:
            if mime.startswith("text/") or file_path.suffix.lower() in [
                ".html",
                ".css",
                ".py",
                ".json",
                ".csv",
                ".ts",
                ".txt",
            ]:
                return file_path.read_text(encoding="utf-8", errors="ignore")
            elif file_path.suffix.lower() == ".docx":
                return FileHandler.read_word_document(file_path)
            elif file_path.suffix.lower() == ".csv":
                # return FileHandler.csv_summary(file_path)
                return ""
            else:
                return f"📦 Binary or unsupported file type: {file_path.name} ({mime})"
        except Exception as e:
            return f"⚠️ Failed to read {file_path.name}: {e}"

    @staticmethod
    def save_temp_file(file) -> tuple[str, str, Path]:
        """Save uploaded file to a temp file and return (filename, mime, tmp_path)."""
        file_suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)

        mime_type, _ = mimetypes.guess_type(tmp_path.name)
        return file.name, mime_type or "application/octet-stream", tmp_path

    @staticmethod
    def read_each_file(uploaded_files) -> dict[str, str]:
        """Returns {filename: content} for each uploaded file."""
        result = {}
        for file in uploaded_files:
            filename, mime, tmp_path = FileHandler.save_temp_file(file)
            content = FileHandler.read_file_content(tmp_path, mime)
            result[filename] = content
        return result

    @staticmethod
    def aggregate_file_content(uploaded_files) -> tuple[str, dict[str, str]]:
        """Returns an aggregated string summary and file paths."""
        file_info: str = ""
        aggregated_context: str = ""
        file_paths: dict[str, str] = {}

        for file in uploaded_files:
            filename, mime, tmp_path = FileHandler.save_temp_file(file)
            file_suffix = Path(filename).suffix
            file_paths[filename] = str(tmp_path)

            file_info += (
                f"The user has uploaded a file {filename} of type {file_suffix}.\n"
            )
            content = FileHandler.read_file_content(tmp_path, mime)

            aggregated_context += f"\n---\nFile: {filename} ({mime})"
            if file_suffix not in [".zip"]:
                aggregated_context += f"\n{content[:100]}\n"

        return file_info + aggregated_context, file_paths
