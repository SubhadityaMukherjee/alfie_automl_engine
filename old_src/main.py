import json
from collections import OrderedDict
from typing import List

import ollama
import requests
from automl_engine.chat_handler import ChatHandler
from automl_engine.models import SessionState
from automl_engine.pipelines import PIPELINES
from automl_engine.utils import render_template
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, SessionState] = {}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]


@app.post("/message/{session_id}")
async def post_message(session_id: str, request: Request):
    content = (await request.json())["content"]
    session = sessions.setdefault(session_id, SessionState())
    session.add_message("user", content)
    return {"status": "ok"}


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    if session_id in sessions:
        sessions[session_id].reset()
    return {"status": "reset"}


def classify_intent(session_id: str, query: str, context: str):
    session = sessions.setdefault(session_id, SessionState())
    generate_reply = render_template(
        jinja_environment=session.jinja_environment,
        template_name="classify_intent.txt",
        prompt=query,
        context=context,
    )
    return {
        "response": ChatHandler.chat(message=generate_reply, context=context).strip()
    }


@app.post("/chat/")
async def handle_chat(
    session_id: str = Form(...),
    prompt: str = Form(...),
    files: List[UploadFile] = File(default_factory=list),
    context: str = Form(...),
):
    session = sessions.setdefault(session_id, SessionState())
    session.add_message("user", prompt)

    reply = "I am a reply"
    session.add_message("assistant", reply)
    return {"reply": reply}


@app.post("/chat/intent_recog/")
async def handle_intent(
    session_id: str = Form(...),
    prompt: str = Form(...),
    files: List[UploadFile] = File(default_factory=list),
    context: str = Form(...),
):
    session = sessions.setdefault(session_id, SessionState())
    session.add_message("user", prompt)

    intent = classify_intent(session_id=session_id, query=prompt, context=context)
    verified = False
    if PIPELINES.get(intent["response"].strip()):
        verified = True

    session.add_message(
        "assistant", f"Recognized Intent {intent.get('response', 'GeneralLLMPipeline')}"
    )
    return {"reply": intent.get("response", "GeneralLLMPipeline"), "verified": verified}


@app.post("/get_pipeline_requirements/{recognized_intent}")
async def get_pipeline_requirements(recognized_intent: str):
    recognized_intent = recognized_intent.strip()
    if PIPELINES.get(recognized_intent, None) is not None:
        return {
            "reply": PIPELINES.get(
                recognized_intent, "GeneralLLMPipeline"
            ).get_required_files()
        }
    else:
        return {"reply": None}
