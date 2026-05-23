import os

from google import genai
from google.genai import types
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()


def _gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API in .env.")
    return api_key


client = genai.Client(api_key=_gemini_api_key())

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    # Retrieves text when a loaded document exposes a text attribute.
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_text(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return [item.values for item in response.embeddings]
