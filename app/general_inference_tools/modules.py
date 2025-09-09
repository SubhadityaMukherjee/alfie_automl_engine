from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore

from lightning import Fabric  # type: ignore

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Reuse upload utilities from tabular module
from app.tabular_automl.services import create_session_directory, save_upload

torch.set_grad_enabled(False)


@dataclass
class TaskConfig:
    model_tag: str
    tokenizer_tag: Optional[str]
    input_types: List[str]


class BaseTask:
    def __init__(self, model: Any, tokenizer: Any, fabric: Fabric, config: TaskConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.fabric = fabric
        self.config = config

    @classmethod
    def requires_hf_model(cls) -> bool:
        return True

    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        raise NotImplementedError


class VideoUnderstandingTask(BaseTask):
    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        with self.fabric.autocast():
            response, new_history = self.model.chat(
                self.tokenizer,
                query,
                inputs,
                history=history,
                do_sample=gen_kwargs.get("do_sample", False),
                num_beams=gen_kwargs.get("num_beams", 3),
                use_meta=gen_kwargs.get("use_meta", True),
            )
        return response, new_history


class ScreenshotToWebpageTask(BaseTask):
    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        seed = gen_kwargs.get("seed", 202)
        repetition_penalty = gen_kwargs.get("repetition_penalty", 3.0)
        with self.fabric.autocast():
            response = self.model.screen_2_webpage(
                query,
                inputs,
                seed=seed,
                repetition_penalty=repetition_penalty,
            )
        return response, None


class InstructionToWebpageTask(BaseTask):
    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        seed = gen_kwargs.get("seed", 202)
        repetition_penalty = gen_kwargs.get("repetition_penalty", 3.0)
        task_name = gen_kwargs.get("task", "Instruction-aware Webpage Generation")
        with self.fabric.autocast():
            response = self.model.write_webpage(
                query,
                seed=seed,
                task=task_name,
                repetition_penalty=repetition_penalty,
            )
        return response, None


class DocumentQATask(BaseTask):
    _pipeline = None  # type: ignore

    @classmethod
    def requires_hf_model(cls) -> bool:
        # Uses HF pipeline directly, no AutoModel loading needed
        return False

    def _get_pipeline(self):
        if self._pipeline is None:
            # Determine device for pipeline
            device_index = 0 if torch.cuda.is_available() else -1
            self._pipeline = hf_pipeline(
                "document-question-answering",
                model=self.config.model_tag,
                device=device_index,
            )
        return self._pipeline

    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        pipe = self._get_pipeline()
        # Expect a single document input; use first path/url
        document_input = inputs[0] if len(inputs) > 0 else ""
        result = pipe(document_input, query)
        # Pipeline may return list of answers; normalize to string
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "answer" in result[0]:
            answer = result[0]["answer"]
        elif isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
        else:
            answer = str(result)
        return answer, None


class GeneralInferenceEngine:
    def __init__(
        self,
        precision: str = "bf16-mixed",
        devices: int = 1,
    ) -> None:
        # Lightning Fabric manages device/precision
        self.fabric = Fabric(accelerator="auto", devices=devices, precision=precision)

        self.registry: Dict[str, TaskConfig] = {}
        self.task_classes: Dict[str, Type[BaseTask]] = {}
        self.models_cache: Dict[str, Any] = {}
        self.tokenizers_cache: Dict[str, Any] = {}

        # Default tasks
        self.register_task(
            "video_understanding",
            TaskConfig(
                model_tag="internlm/internlm-xcomposer2d5-7b",
                tokenizer_tag="internlm/internlm-xcomposer2d5-7b",
                input_types=["*.mp4"],
            ),
            VideoUnderstandingTask,
        )

        # Screenshot to webpage (image -> Tailwind HTML)
        self.register_task(
            "screenshot_to_webpage",
            TaskConfig(
                model_tag="internlm/internlm-xcomposer2d5-7b",
                tokenizer_tag="internlm/internlm-xcomposer2d5-7b",
                input_types=["*.jpg", "*.jpeg", "*.png"],
            ),
            ScreenshotToWebpageTask,
        )

        # Instruction to webpage (text -> Tailwind HTML)
        self.register_task(
            "instruction_to_webpage",
            TaskConfig(
                model_tag="internlm/internlm-xcomposer2d5-7b",
                tokenizer_tag="internlm/internlm-xcomposer2d5-7b",
                input_types=[],
            ),
            InstructionToWebpageTask,
        )

        # Document Q&A (image/pdf/url -> answer)
        self.register_task(
            "document_qa",
            TaskConfig(
                model_tag="impira/layoutlm-document-qa",
                tokenizer_tag=None,
                input_types=["*.jpg", "*.jpeg", "*.png", "*.pdf"],
            ),
            DocumentQATask,
        )

    def register_task(self, name: str, config: TaskConfig, task_cls: Type[BaseTask]) -> None:
        self.registry[name] = config
        self.task_classes[name] = task_cls

    def _resolve_torch_dtype(self) -> torch.dtype:
        precision = str(self.fabric._precision).lower()  # type: ignore[attr-defined]
        if "bf16" in precision:
            return torch.bfloat16
        if "16" in precision:
            return torch.float16
        return torch.float32

    def _get_model_and_tokenizer(self, task_name: str) -> Tuple[Any, Any]:
        if task_name not in self.registry:
            raise ValueError(f"Unknown task '{task_name}'. Registered: {list(self.registry.keys())}")
        cfg = self.registry[task_name]

        if cfg.model_tag in self.models_cache:
            cached_model = self.models_cache[cfg.model_tag]
            cached_tokenizer = None
            if cfg.tokenizer_tag is not None and cfg.tokenizer_tag in self.tokenizers_cache:
                cached_tokenizer = self.tokenizers_cache[cfg.tokenizer_tag]
            return cached_model, cached_tokenizer

        torch_dtype = self._resolve_torch_dtype()
        model = AutoModel.from_pretrained(
            cfg.model_tag,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()
        model = self.fabric.setup_module(model)

        tokenizer = None
        if cfg.tokenizer_tag is not None:
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_tag, trust_remote_code=True)

        # Some repos expect model.tokenizer to be set
        if tokenizer is not None:
            try:
                model.tokenizer = tokenizer  # type: ignore[attr-defined]
            except Exception:
                pass

        self.models_cache[cfg.model_tag] = model
        if cfg.tokenizer_tag is not None and tokenizer is not None:
            self.tokenizers_cache[cfg.tokenizer_tag] = tokenizer
        return model, tokenizer

    def run_task(
        self,
        task_name: str,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        if task_name not in self.task_classes:
            raise ValueError(f"Task '{task_name}' not registered")

        task_cls = self.task_classes[task_name]
        cfg = self.registry[task_name]
        if task_cls.requires_hf_model():
            model, tokenizer = self._get_model_and_tokenizer(task_name)
        else:
            model, tokenizer = None, None
        task = task_cls(model, tokenizer, self.fabric, cfg)
        return task.run(query, inputs, history=history, **gen_kwargs)


# ---------------------- FastAPI Endpoints ----------------------

# Singleton engine to reuse model weights across requests
_engine: Optional[GeneralInferenceEngine] = None

def get_engine() -> GeneralInferenceEngine:
    global _engine
    if _engine is None:
        _engine = GeneralInferenceEngine(precision="bf16-mixed", devices=1)
    return _engine


class RunTaskRequest(BaseModel):
    session_id: str
    task_name: str
    query: str
    gen_kwargs: Optional[Dict[str, Any]] = None


class RequirementsRequest(BaseModel):
    requirements: str
    gen_kwargs: Optional[Dict[str, Any]] = None


class VideoUnderstandingRequest(BaseModel):
    query: str
    inputs: List[str]
    history: Optional[Any] = None
    gen_kwargs: Optional[Dict[str, Any]] = None


class DocumentQARequest(BaseModel):
    document: str  # path/url to the document or image
    question: str
    gen_kwargs: Optional[Dict[str, Any]] = None

