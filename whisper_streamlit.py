import streamlit as st
import whisper
import tempfile
import os
from utils.langchain_llms import LangchainLLama
from pydantic import BaseModel, Field
from typing import Any

st.title("Whisper Audio Transcription App")

# Load the model only once using session state


class STState(BaseModel):
    default_state: dict[str, Any] = Field(default={})

    def init_state(self):
        if not st.session_state.get("initialized", None):
            for (key, value) in self.default_state.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state["initialized"] = True
            st.session_state["st_state"] = self

    @staticmethod
    def get_state(key: str) -> any:
        return st.session_state[key]

    @staticmethod
    def delete_state(key: str):
        return st.session_state.pop(key, None)

    @staticmethod
    def get_initialized():
        return st.session_state.get("initialized", None)

    @staticmethod
    def get_st_state():
        return st.session_state["st_state"]

    @staticmethod
    def set_state(key: str, value: any):
        st.session_state[key] = value


def load_model():
    return whisper.load_model("base.en")


def load_chat_model():
    return LangchainLLama(model="llama3.2")


st_state = None
if not STState.get_initialized():
    st_state = STState(default_state={
        "whisper_model": load_model(),
        "chat_model": load_chat_model(),
        "transcription": None,
        "summary": None,
        "upload_file": None,
    })
    st_state.init_state()
else:
    st_state = STState.get_st_state()

uploaded_file = st.file_uploader("Upload an audio file", type=[
                                 "mp3", "wav", "m4a", "webm", "ogg", "flac"])


def summarize():
    with st.spinner("Summarizing..."):
        transcription = STState.get_state("transcription")
        summary = st_state.get_state("chat_model").get_response(
            f"Summarize the following text: {transcription}")
        STState.set_state("summary", summary)
        summarize_placeholder.markdown(">" + summary)


def transcribe():
    tmp_filepath = STState.get_state("upload_file")
    with st.spinner("Transcribing audio..."):
        result = st_state.get_state(
            "whisper_model").transcribe(tmp_filepath)
        STState.set_state("transcription", result["text"])
        transcript_placeholder.subheader("Transcription:")
        transcript_placeholder.markdown(">" + '"' + result["text"] + '"')
    os.remove(tmp_filepath)


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    STState.delete_state("upload_file")
    STState.delete_state("transcription")
    STState.delete_state("summary")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filepath = tmp_file.name
        STState.set_state("upload_file", tmp_filepath)

    if st.button("Transcribe", on_click=transcribe, key="transcribe"):
        st.button("Summarize", on_click=summarize, key="summarize")


summarize_placeholder = st.empty()
transcript_placeholder = st.empty()
