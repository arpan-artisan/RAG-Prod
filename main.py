import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from google import genai
from google.genai import types
from data_loader import load_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGSearchResults, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()


def _gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API in .env.")
    return api_key


gemini_client = genai.Client(api_key=_gemini_api_key())

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,  # in production we need more security
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vectors = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        # Storing in Qdrant
        QdrantStorage().upsert(ids, vectors, payloads)

        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()  # takes pydantic model and converts it into json or object


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k=5) -> RAGSearchResults:
        query_vector = embed_text([question], task_type="RETRIEVAL_QUERY")[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResults(contexts=found["contexts"], sources=found["sources"])

    def _generate_answer(prompt: str) -> str:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.2,
                system_instruction="You answer questions using only the context provided.",
            ),
        )
        return (response.text or "").strip()

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResults)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    ans = await ctx.step.run("llm-answer", lambda: _generate_answer(user_content))
    return {"answer": ans, "sources": found.sources, "num_context": len(found.contexts)}


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
