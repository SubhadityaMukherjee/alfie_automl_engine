from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

from lightning import Fabric  # type: ignore

from pydantic import BaseModel  # type: ignore

# Reuse upload utilities from tabular module (none needed in this module)

torch.set_grad_enabled(False)


@dataclass
class TaskConfig:
    model_tag: str
    tokenizer_tag: Optional[str]
    input_types: List[str]


class BaseTask:
    def __init__(self, model: Any, tokenizer: Any, fabric: Fabric, config: TaskConfig) -> None:
        self.model = model
        # Prefer explicit tokenizer; fall back to model.tokenizer when available
        if tokenizer is None and hasattr(model, "tokenizer"):
            try:
                tokenizer = getattr(model, "tokenizer")
            except Exception:
                pass
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
        from PIL import Image  # type: ignore

        if len(inputs) == 0 or inputs[0] is None:
            return "", history

        image_path = inputs[0]
        image = Image.open(image_path).convert("RGB")

        msgs = [{"role": "user", "content": query}]

        with self.fabric.autocast():
            # Use positional args to satisfy backends that require them
            res, ctx, _ = self.model.chat(
                image,
                msgs,
                history,
                self.tokenizer,
                sampling=gen_kwargs.get("sampling", True),
                temperature=gen_kwargs.get("temperature", 0.7),
            )
        return res, ctx


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
        prompt = f"Generate Tailwind CSS HTML based on screenshot(s). Requirements: {query}"
        with self.fabric.autocast():
            # Prefer specialized method if present
            if hasattr(self.model, "screen_2_webpage"):
                response = self.model.screen_2_webpage(  # type: ignore[attr-defined]
                    query,
                    inputs,
                    seed=seed,
                    repetition_penalty=repetition_penalty,
                )
                return response, None
            # Fallback to generic chat if available
            if hasattr(self.model, "chat"):
                response, new_history = self.model.chat(  # type: ignore[attr-defined]
                    prompt,
                    inputs,
                    tokenizer = self.tokenizer,
                    history=history,
                    do_sample=gen_kwargs.get("do_sample", False),
                    num_beams=gen_kwargs.get("num_beams", 3),
                    use_meta=gen_kwargs.get("use_meta", True),
                )
                return response, new_history
        # Final fallback: minimal template
        html = (
            "<html><head><script src=\"https://cdn.tailwindcss.com\"></script></head>"
            "<body class=\"min-h-screen flex items-center justify-center bg-gray-50\">"
            f"<div class=\"p-6 bg-white rounded shadow\"><p class=\"mb-2 text-sm text-gray-600\">{query}</p>"
            "<button class=\"px-4 py-2 bg-blue-600 text-white rounded\">Action</button>"
            "</div></body></html>"
        )
        return html, None


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
        prompt = f"{task_name}: Generate Tailwind CSS HTML. Requirements: {query}"
        with self.fabric.autocast():
            # Prefer specialized method if present
            if hasattr(self.model, "write_webpage"):
                response = self.model.write_webpage(  # type: ignore[attr-defined]
                    query,
                    seed=seed,
                    task=task_name,
                    repetition_penalty=repetition_penalty,
                )
                return response, None
            # Fallback to generic chat if available
            if hasattr(self.model, "chat"):
                response, new_history = self.model.chat(  # type: ignore[attr-defined]
                    prompt,
                    [],
                    tokenizer = self.tokenizer,
                    history=history,
                    do_sample=gen_kwargs.get("do_sample", False),
                    num_beams=gen_kwargs.get("num_beams", 3),
                    use_meta=gen_kwargs.get("use_meta", True),
                )
                return response, new_history
        # Final fallback: minimal template
        html = (
            "<html><head><script src=\"https://cdn.tailwindcss.com\"></script></head>"
            "<body class=\"min-h-screen flex items-center justify-center bg-gray-50\">"
            f"<div class=\"p-6 bg-white rounded shadow\"><h1 class=\"text-xl font-semibold mb-4\">{query}</h1>"
            "<form class=\"space-y-3\"><input class=\"border px-3 py-2 rounded w-64\" placeholder=\"Email\"/>"
            "<input class=\"border px-3 py-2 rounded w-64\" placeholder=\"Password\" type=\"password\"/>"
            "<button class=\"px-4 py-2 bg-blue-600 text-white rounded w-64\">Submit</button></form>"
            "</div></body></html>"
        )
        return html, None


class DocumentQATask(BaseTask):
    _model = None  # type: ignore
    _processor = None  # type: ignore

    @classmethod
    def requires_hf_model(cls) -> bool:
        # This task manages its own model loading (Donut DocVQA)
        return False

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        model_tag = self.config.model_tag

        # Resolve device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Donut DocVQA components
        self._processor = AutoProcessor.from_pretrained(model_tag)
        self._model = AutoModelForVision2Seq.from_pretrained(model_tag)
        self._model.eval()
        try:
            self._model.to(device)
        except Exception:
            # Some backends may not support .to(mps); ignore to stay on CPU
            pass

    def run(
        self,
        query: str,
        inputs: List[str],
        history: Optional[Any] = None,
        **gen_kwargs: Any,
    ) -> Tuple[str, Any]:
        from PIL import Image  # type: ignore  # local import to avoid hard dependency elsewhere

        self._lazy_load()

        # Help type-checkers
        assert self._model is not None
        assert self._processor is not None

        if len(inputs) == 0 or inputs[0] is None:
            return "", None

        document_path = inputs[0]
        image = Image.open(document_path).convert("RGB")

        # Donut requires special task prompt format
        task_prompt = f"<s_docvqa><s_question>{query}</s_question><s_answer>"

        # Prepare inputs for Donut
        enc = self._processor(image, task_prompt, return_tensors="pt")

        # Move to model device if possible
        device = next(self._model.parameters()).device  # type: ignore[attr-defined]
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

        max_length = int(gen_kwargs.get("max_length", 512))
        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids=enc["input_ids"],
                pixel_values=enc["pixel_values"],
                max_length=max_length,
            )

        # Decode and extract answer text using the processor
        answer = self._processor.decode(outputs[0], skip_special_tokens=True).strip()
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
                model_tag="openbmb/MiniCPM-V-2",
                tokenizer_tag="openbmb/MiniCPM-V-2",
                input_types=["*.mp4"],
            ),
            VideoUnderstandingTask,
        )

        # Screenshot to webpage (image -> Tailwind HTML)
        self.register_task(
            "screenshot_to_webpage",
            TaskConfig(
                model_tag="openbmb/MiniCPM-V-2",
                tokenizer_tag="openbmb/MiniCPM-V-2",
                input_types=["*.jpg", "*.jpeg", "*.png"],
            ),
            ScreenshotToWebpageTask,
        )

        # Instruction to webpage (text -> Tailwind HTML)
        self.register_task(
            "instruction_to_webpage",
            TaskConfig(
                model_tag="openbmb/MiniCPM-V-2",
                tokenizer_tag="openbmb/MiniCPM-V-2",
                input_types=[],
            ),
            InstructionToWebpageTask,
        )

        # Document Q&A (image/pdf/url -> answer)
        self.register_task(
            "document_qa",
            TaskConfig(
                model_tag="naver-clova-ix/donut-base-finetuned-docvqa",
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

        # Resolve device and ensure dtype/device placement aligns with backend guidance
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Adjust dtype for MPS where bfloat16 is not supported
        effective_dtype = torch_dtype
        if device.type == "mps" and torch_dtype == torch.bfloat16:
            effective_dtype = torch.float16

        try:
            model = model.to(device=device, dtype=effective_dtype)  # type: ignore[call-arg]
        except Exception:
            # Some wrapped modules may not support .to with dtype; ignore
            try:
                model = model.to(device=device)  # type: ignore[call-arg]
            except Exception:
                pass

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

