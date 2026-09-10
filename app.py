# import os, re, networkx as nx
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# from uuid import uuid4

# # Langchain Imports
# from langchain.chains import LLMChain, create_retrieval_chain
# from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
# from langchain_community.vectorstores import Chroma
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_groq import ChatGroq
# from groq import Groq 
# load_dotenv()

# from transformers import pipeline

# print("✅ Loading moderation model...")
# classifier = pipeline("text-classification", model="arun86/hate-offensive-suicidal-bert1")


# # === Flask Setup ===
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust as needed

# import os

# print("✅ Current working directory:", os.getcwd())
# print("✅ Files in /app/pdfs:", os.listdir("pdfs"))

# # === Document Setup ===
# pdf_filename = os.path.join("pdfs", "GriefBot.pdf")
# loader = PyPDFLoader(pdf_filename)
# documents = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
# chunks = splitter.split_documents(documents)

# # === Vector Store ===
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(chunks, embedding=embedding)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# # === LLM Setup ===
# api_key = os.getenv("GROQ")
# if not api_key:
#     raise ValueError("Missing GROQ API key")
# # llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
# # llm = ChatGroq(groq_api_key=api_key, model_name="qwen/qwen3-32b")
# llm = ChatGroq(
#     groq_api_key=api_key,
#     model_name="qwen/qwen3-32b",
#     temperature=0.6,
#     top_p=0.95
# )

# # === Knowledge Graph Setup ===
# KG = nx.DiGraph()

# triple_extraction_prompt = PromptTemplate.from_template(
#     """Extract all subject-relationship-object triples from the text below.

# Text: "{text}"

# Format each triple like: (subject, relation, object)

# Only return the list of triples. Do not explain.

# Triples:"""
# )
# triple_extraction_chain = LLMChain(llm=llm, prompt=triple_extraction_prompt)

# def extract_triples(text):
#     try:
#         raw_output = triple_extraction_chain.invoke({"text": text})["text"]
#         pattern = r"\(\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*\)"
#         matches = re.findall(pattern, raw_output)
#         return [(s.strip(), r.strip(), o.strip()) for s, r, o in matches]
#     except Exception as e:
#         print("⚠️ Triple extraction failed:", e)
#         return []

# def store_kg(triples):
#     for s, r, o in triples:
#         KG.add_edge(s, o, label=r)

# def get_kg_facts(entity=None):
#     facts = []
#     for u, v, d in KG.edges(data=True):
#         if not entity or entity in (u, v):
#             facts.append(f"{u} {d['label']} {v}")
#     return "\n".join(facts)

# # === Validator Chain ===
# validation_prompt = PromptTemplate.from_template(
#     """You are a validation engine for a grief support chatbot.

# User's message: "{query}"

# Only respond with one of:
# - "valid"
# - "nonsensical"
# - "unrelated"
# - "illogical"
# - "offensive"
# - "harmful"

# Your answer:"""
# )
# validator_chain = LLMChain(llm=llm, prompt=validation_prompt)

# # === System Prompt ===
# system_prompt = (
#     """
# Here's how you respond:
# - Use short, simple, human-sounding sentences.
# - Acknowledge emotion first.
# - Offer support or reflection, not lectures.
# - Ask gentle follow-up questions when needed.
# - Only include facts from provided context if relevant.
# - Never guess if you’re unsure — just say so kindly.

# Avoid sounding like a bot or giving long, polished essays.

# Always DEEPLY HUMANIZE YOUR RESPONSES.

# Now continue the conversation naturally. Keep your response in 2-3 lines maximum

# {context}"""
# )

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
# ])

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # === Session History ===
# session_store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     print("sessionid:", session_id)
#     if session_id not in session_store:
#         session_store[session_id] = ChatMessageHistory()
#     return session_store[session_id]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer"
# )

# def build_system_prompt(verdict: str):
#     print("verdict in prompt build", verdict)
#     advisory = ""
#     if verdict in ["illogical", "nonsensical"]:
#         advisory = (
#             f"The user's message was flagged as '{verdict}'. "
#             "If something doesn’t make sense, clarify kindly. Be gentle and make sure their feelings are not hurt\n\n"
#             "Just make sure you make no errors. You have to clarify if the query doesn't seem logical"
#             "You have to make sure that you clarify the mistake so that you dont sound stupid"
#             "{context}"
#         )
#         return advisory

#     return advisory + """
# - You are based in the United Kingdom
# - Use short, simple, human-sounding sentences.
# - Acknowledge emotion first.
# - Offer support or reflection, not lectures.
# - Ask gentle follow-up questions when needed.
# - Only include facts from provided context if relevant.
# - Never guess if you’re unsure — just say so kindly.

# Avoid sounding like a bot or giving long, polished essays.

# Always DEEPLY HUMANIZE YOUR RESPONSES.

# Now continue the conversation naturally. Keep your response in 2-3 lines maximum

# {context}"""

# groq_client = Groq(api_key=os.getenv("GROQ"))
# print("groq_client",groq_client)
# def validate_message(query: str) -> str:
#     print("Coming inside validate message")
#     response = groq_client.chat.completions.create(
#         model="qwen/qwen3-32b",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a validation engine for a grief support chatbot.\n"
#                            "Only return one of: valid, nonsensical, unrelated, illogical, offensive, harmful."
#             },
#             {
#                 "role": "user",
#                 "content": f'User message: "{query}"'
#             }
#         ],
#         temperature=0.6,
#         top_p=0.95
#     )
#     print("response of validation", response)
#     reasoning = getattr(response.choices[0], "reasoning", None)
#     answer = response.choices[0].message.content.strip().lower()

#     print("🧠 Validator Reasoning:\n", reasoning)
#     print("✅ Validator Verdict:", answer)

#     raw_answer = answer
#     cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

#     return cleaned_answer

# def moderate_content(text: str) -> tuple[bool, str | None]:

#     client = Groq(api_key=os.getenv("GROQ"))
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-Guard-4-12B",
#         messages=[
#             {
#                 "role": "user",
#                 "content": text
#             }
#         ]
#     )

#     result = response.choices[0].message.content.strip()
#     print("🛡️ Moderation result:", repr(result))

#     if result.startswith("safe"):
#         return True, None
#     elif result.startswith("unsafe"):
#         lines = result.splitlines()
#         if len(lines) > 1:
#             return False, lines[1].strip()
#         return False, "unknown"
#     else:
#         # Fallback if output is unclear
#         return False, "unclassified"

# def custom_moderate_content(text: str) -> dict:
#     result = classifier(text)[0]
#     label = result["label"].lower()
#     score = result["score"]

#     is_safe = label in ["neither", "suicidal"]  # allow suicidal content for support

#     return {
#         "is_safe": is_safe,
#         "label": label,
#         "score": round(score, 4),
#         "original_text": text
#     }

# def ask_bot(query, session_id="default"):
#     # Step 1: Validate the message using raw Groq SDK
#     verdict = validate_message(query)
#     print("verdict", verdict)
#     if verdict in ["unrelated", "offensive", "harmful"]:
#         return "🤖 I'm here to help with grief-related concerns. Could you ask something else?"

#     # Step 2: Triple extraction and KG update
#     triples = extract_triples(query)
#     store_kg(triples)

#     # Step 3: Get relevant documents from vector store
#     docs = retriever.get_relevant_documents(query)
#     if not docs or sum(len(doc.page_content.strip()) for doc in docs) < 100:
#         return "🤖 I'm not sure how to answer that based on what I know."

#     # Step 4: Load session history
#     history = get_session_history(session_id)

#     # Step 5: Dynamically build prompt with verdict
#     prompt_text = build_system_prompt(verdict)
#     print("prompt text", prompt_text)
#     dynamic_qa_prompt = ChatPromptTemplate.from_messages([
#         ("system", prompt_text),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ])
#     question_answer_chain = create_stuff_documents_chain(llm, dynamic_qa_prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     dynamic_conversational_chain = RunnableWithMessageHistory(
#         rag_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer"
#     )

#     # Step 6: Generate bot response
#     response = dynamic_conversational_chain.invoke(
#         {"input": query, "chat_history": history.messages},
#         config={"configurable": {"session_id": session_id}}
#     )

#     raw_answer = response["answer"]
#     cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
#     return cleaned_answer


# # === API Routes ===
# @app.route("/")
# def home():
#     return "GriefBot API is running."

# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.get_json()
#     query = data.get("question")
#     print("query:", query)
#     if not query:
#         return jsonify({"error": "Missing question"}), 400

#     session_id = data.get("session_id") or str(uuid4())

#     try:
#         response = ask_bot(query, session_id)
#         return jsonify({"response": response, "session_id": session_id})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/test-cors", methods=["GET"])
# def test_cors():
#     return jsonify({"message": "CORS is working!"})

# @app.route("/moderate", methods=["POST"])
# def moderate():
#     data = request.get_json()
#     text = data.get("content")
#     print("text received", text)
#     if not text:
#         return jsonify({"error": "Missing 'text' in request body"}), 400

#     result = custom_moderate_content(text)
#     return jsonify(result)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)


import os, time, json, logging
import re
import tempfile
import requests
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import numpy as np
import networkx as nx
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from groq import Groq

# ----------------- Env / Basics -----------------
load_dotenv()
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger("moderation")
log.setLevel(logging.INFO)

def on_azure_app_service() -> bool:
    return bool(os.getenv("WEBSITE_SITE_NAME"))

def ensure_dir(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except Exception as e:
        print(f"⚠️ Could not create {path}: {e}. Falling back to system temp.")
        t = Path(tempfile.gettempdir()) / "huggingface"
        t.mkdir(parents=True, exist_ok=True)
        return str(t)

# Cache dirs (harmless; we’re not downloading big models)
if on_azure_app_service():
    base_cache = Path("/home/data/.cache")
else:
    base_cache = Path.cwd() / ".cache"


def resolve_hf_cache_dir() -> Path:
    # Respect explicit env first
    if os.getenv("HF_HOME"):
        return Path(os.getenv("HF_HOME"))
    # Azure App Service has a writable /home
    if on_azure_app_service():
        return Path("/home/data/.cache")
    # Local/dev default
    return Path.home() / ".cache" / "huggingface"

cache_dir = resolve_hf_cache_dir()
try:
    cache_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Could not create {cache_dir}: {e}. Falling back to temp dir.")
    cache_dir = Path(tempfile.gettempdir()) / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
print("✅ HF cache directory:", cache_dir)

_classifier = None

hf_dir = ensure_dir(base_cache / "huggingface")
os.environ.setdefault("HF_HOME", hf_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_dir)
print("✅ HF cache directory:", hf_dir)

# ----------------- Flask -----------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("✅ CWD:", os.getcwd())
PDF_DIR = Path.cwd() / "pdfs"

# ----------------- Globals -----------------
_groq_client: Optional[Groq] = None
_KG = nx.DiGraph()

# Chat memory (in-memory per-process; **we rely on the session_id you pass**)
_session_store: Dict[str, ChatMessageHistory] = {}

# Simple in-memory TF-IDF index (no sklearn/scipy/faiss/chroma)
_docs: List[Document] = []
_vocab: Dict[str, int] = {}
_idf: Optional[np.ndarray] = None
_doc_matrix: Optional[np.ndarray] = None   # tf-idf dense matrix (n_docs x vocab)
_token_pattern = re.compile(r"[A-Za-z0-9']+")

# Model lists (configurable via env)
DEFAULT_QA_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
DEFAULT_VALIDATION_MODELS = [
    "openai/gpt-oss-20b",
]

def parse_model_list(env_name: str, default_list: List[str]) -> List[str]:
    raw = os.getenv(env_name, "")
    lst = [s.strip() for s in raw.split(",") if s.strip()]
    return lst or default_list

def safe_get_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("answer", "text", "output_text", "content"):
            if k in x and isinstance(x[k], str):
                return x[k]
        # fallback: first stringy value
        for v in x.values():
            if isinstance(v, str):
                return v
    return str(x)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_pattern.findall(text)]

# ----------------- Lazy initializers -----------------
def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ")
        if not api_key:
            raise ValueError("Missing GROQ API key. Set env var GROQ.")
        _groq_client = Groq(api_key=api_key)
        print("✅ Groq client ready")
    return _groq_client

def get_pdf_chunks() -> List[Document]:
    pdf_filename = PDF_DIR / "GriefBot.pdf"
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    if not pdf_filename.exists():
        print(f"⚠️ PDF not found at {pdf_filename}. Using fallback content.")
        text = (
            "Grief support basics: Acknowledge feelings, avoid fixing, ask gentle questions, "
            "encourage support networks, and seek professional help if risk is present."
        )
        return splitter.split_documents([Document(page_content=text)])

    try:
        loader = PyPDFLoader(str(pdf_filename))
        documents = loader.load()
        return splitter.split_documents(documents)
    except Exception as e:
        print("⚠️ PDF load error, using fallback doc:", e)
        return splitter.split_documents([Document(page_content="Grief support: listen first, reflect feelings, keep responses short and warm.")])

def build_tfidf_index():
    """Pure NumPy TF-IDF index."""
    global _docs, _vocab, _idf, _doc_matrix
    if _doc_matrix is not None and _idf is not None:
        return

    _docs = get_pdf_chunks()
    texts = [d.page_content for d in _docs]
    tokenized = [tokenize(t) for t in texts]

    # Build vocab
    vocab: Dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    _vocab = vocab
    V = len(vocab)
    N = len(texts)
    if N == 0 or V == 0:
        _idf = np.zeros((0,), dtype=np.float32)
        _doc_matrix = np.zeros((0, 0), dtype=np.float32)
        print("⚠️ Empty TF-IDF index")
        return

    # Term frequencies per doc
    tf = np.zeros((N, V), dtype=np.float32)
    df = np.zeros(V, dtype=np.int32)

    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts: Dict[int, int] = {}
        for t in tokens:
            j = vocab[t]
            counts[j] = counts.get(j, 0) + 1
        maxc = max(counts.values())
        for j, c in counts.items():
            tf[i, j] = c / maxc  # normalized term freq
        for j in counts:
            df[j] += 1

    # IDF
    idf = np.log((N + 1) / (df + 1)) + 1.0  # smoothed
    _idf = idf.astype(np.float32)

    # TF-IDF
    mat = tf * _idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    _doc_matrix = mat.astype(np.float32)
    print(f"✅ TF-IDF index ready: {N} docs, {V} terms")

def tfidf_vector_for_query(q: str) -> np.ndarray:
    tokens = tokenize(q)
    if not tokens or _idf is None or _doc_matrix is None or len(_vocab) == 0:
        return np.zeros((len(_vocab),), dtype=np.float32)
    counts: Dict[int, int] = {}
    for t in tokens:
        j = _vocab.get(t)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    if not counts:
        return np.zeros((len(_vocab),), dtype=np.float32)
    maxc = max(counts.values())
    vec = np.zeros((len(_vocab),), dtype=np.float32)
    for j, c in counts.items():
        vec[j] = (c / maxc) * _idf[j]
    n = np.linalg.norm(vec) + 1e-12
    return (vec / n).astype(np.float32)

def retrieve_docs(query: str, k: int = 4) -> List[Document]:
    if _doc_matrix is None:
        build_tfidf_index()
    if _doc_matrix.size == 0:
        return []
    qv = tfidf_vector_for_query(query)
    sims = _doc_matrix @ qv  # cosine because rows are normalized
    topk = np.argsort(-sims)[:k]
    return [_docs[i] for i in topk.tolist()]

# ----------------- KG / Prompts -----------------
def extract_triples_with_failover(text: str) -> List[tuple]:
    try:
        models = parse_model_list("GROQ_MODELS", DEFAULT_QA_MODELS)
        prompt = (
            "Extract all subject-relationship-object triples from the text below.\n\n"
            f'Text: "{text}"\n\n'
            "Format each triple like: (subject, relation, object)\n\n"
            "Only return the list of triples. Do not explain.\n\n"
            "Triples:"
        )
        messages = [{"role": "user", "content": prompt}]
        content, model_used = groq_chat_with_failover(messages, models, temperature=0, max_tokens=256)
        pattern = r"\(\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*\)"
        matches = re.findall(pattern, content or "")
        return [(s.strip(), r.strip(), o.strip()) for s, r, o in matches]
    except Exception as e:
        print("⚠️ Triple extraction failed:", e)
        return []

def store_kg(triples: List[tuple]):
    for s, r, o in triples:
        try:
            _KG.add_edge(s, o, label=r)
        except Exception as e:
            print("⚠️ KG store error:", e)

def build_system_prompt(verdict: str):
    if verdict in ["illogical", "nonsensical"]:
        return (
            f"The user's message was flagged as '{verdict}'. "
            "If something doesn’t make sense, clarify kindly. Be gentle and avoid shaming.\n"
            "Make no factual errors. If the query seems illogical, briefly clarify the misunderstanding.\n"
            "{context}"
        )
    return """
- You are based in the United Kingdom
- Use short, simple, human-sounding sentences.
- Acknowledge emotion first.
- Offer support or reflection, not lectures.
- Ask gentle follow-up questions when needed.
- Only include facts from provided context if relevant.
- Never guess if you’re unsure — just say so kindly.

Avoid sounding like a bot or giving long, polished essays.

Always DEEPLY HUMANIZE YOUR RESPONSES.

Now continue the conversation naturally. Keep your response in 2-3 lines maximum

{context}"""

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

# ----------------- Groq helpers (failover + backoff) -----------------
RETRY_STATUS_HINTS = ("over capacity", "rate", "timeout", "temporarily", "503", "429")

def groq_chat_with_failover(messages, models: List[str], temperature=0.6, top_p=0.95, max_tokens: Optional[int]=None):
    client = get_groq_client()
    last_err = None
    delay = 1.0
    for model in models:
        attempts = 0
        while attempts < 3:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=30,
                )
                try:
                    txt = resp.choices[0].message.content
                except Exception:
                    txt = getattr(resp.choices[0], "text", "")
                return (txt or ""), model
            except Exception as e:
                msg = str(e).lower()
                last_err = e
                transient = any(h in msg for h in RETRY_STATUS_HINTS)
                print(f"⚠️ Groq error on model '{model}' (attempt {attempts+1}): {e}")
                if not transient:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 8)
                attempts += 1
    raise last_err or RuntimeError("Groq call failed for all models")

# ----------------- Validation + Moderation -----------------
def validate_message(query: str) -> str:
    print("🔎 Validating message…")
    models = parse_model_list("GROQ_VALIDATION_MODELS", parse_model_list("GROQ_MODELS", DEFAULT_VALIDATION_MODELS))
    messages = [
        {"role": "system", "content":
            "You are a validation engine for a grief-support chatbot. "
            "Classify the user's message into exactly one category and reply with that single word only:\n"
            "- valid: about grief, loss, bereavement, or emotional support.\n"
            "- unrelated: a coherent request that is NOT about grief/loss/emotional support "
            "(e.g. general knowledge, coding, sports, weather, trivia).\n"
            "- nonsensical: gibberish or not understandable.\n"
            "- illogical: self-contradictory or logically impossible.\n"
            "- offensive: hateful, harassing, or abusive.\n"
            "- harmful: requests or promotes danger (violence, self-harm instructions, etc.).\n"
            "Respond with only one of: valid, nonsensical, unrelated, illogical, offensive, harmful."},
        {"role": "user", "content": f'User message: "{query}"'}
    ]
    # gpt-oss models spend tokens on an internal reasoning phase before the visible
    # verdict, so give enough headroom or the content comes back empty.
    txt, used = groq_chat_with_failover(messages, models, temperature=0.6, top_p=0.95, max_tokens=512)
    cleaned = re.sub(r"<think>.*?</think>", "", str(txt), flags=re.DOTALL).strip().lower()
    # Extract the verdict word robustly so extra text can't break the downstream match.
    labels = ["harmful", "offensive", "nonsensical", "illogical", "unrelated", "valid"]
    verdict = next((lbl for lbl in labels if re.search(rf"\b{lbl}\b", cleaned)), cleaned)
    print(f"✅ Validator verdict ({used}): {verdict}  (raw: {cleaned!r})")
    return verdict

def moderate_with_llamaguard(text: str) -> Tuple[bool, Optional[str]]:
    models = [os.getenv("GROQ_MODERATION_MODEL", "openai/gpt-oss-safeguard-20b")]
    # gpt-oss-safeguard follows an explicit policy and returns a verdict we parse below.
    system_policy = (
        "You are a content-moderation engine for a grief-support chatbot. "
        "Decide whether the user's message is SAFE to answer. "
        "Content is UNSAFE only if it involves hate, harassment, sexual content involving minors, "
        "instructions for violence or self-harm, or other clearly harmful requests. "
        "Expressions of grief, sadness, or suicidal feelings seeking support are SAFE. "
        "Reply with exactly one word on the first line: 'safe' or 'unsafe'. "
        "If unsafe, add a short reason on the next line."
    )
    messages = [
        {"role": "system", "content": system_policy},
        {"role": "user", "content": text},
    ]
    try:
        resp, used = groq_chat_with_failover(messages, models, temperature=0, max_tokens=512)
        res = (resp or "").strip()
        # Strip any <think>…</think> reasoning some models emit, then look at the verdict.
        res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip().lower()
        first_line = res.splitlines()[0].strip() if res else ""
        if first_line.startswith("safe") or (not first_line.startswith("unsafe") and "unsafe" not in res):
            return True, None
        # Unsafe: return the reason (line after the verdict if present, else the whole body).
        lines = [ln.strip() for ln in res.splitlines() if ln.strip()]
        reason = lines[1] if len(lines) > 1 else (first_line or "unsafe")
        return False, reason
    except Exception as e:
        print("⚠️ Moderation call failed (allowing request):", e)
        return True, None

# ----------------- Answering (MANUAL MEMORY) -----------------
def answer_with_model(model_name: str, query: str, docs: List[Document], history: ChatMessageHistory, verdict: str) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ"),
        model_name=model_name,
        temperature=0.6,
        top_p=0.95
    )
    prompt_text = build_system_prompt(verdict)

    # Prompt uses history + current user input
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Render the prompt to messages and call LLM directly to avoid wrapper issues
    rendered = qa_prompt.invoke({
        "chat_history": history.messages,   # <- THIS injects full prior convo
        "input": query,
        "context": docs                     # available to the prompt if you use {context}
    })
    # LLM call
    resp = llm.invoke(rendered.to_messages())
    text = safe_get_text(getattr(resp, "content", resp))
    return text


def ask_bot(query, session_id="default"):
    # Moderation (soft-fail)
    is_safe, reason = moderate_with_llamaguard(query)
    if not is_safe:
        return "🙏 I can’t help with that request. If you’re in distress, please consider reaching out to someone you trust or local support."

    verdict = validate_message(query)
    if verdict in ["unrelated", "offensive", "harmful"]:
        return "🤖 I’m here to help with grief-related concerns. Could you ask something else?"

    # Get session history
    history = get_session_history(session_id)

    # Update KG (best-effort)
    try:
        store_kg(extract_triples_with_failover(query))
    except Exception as e:
        print("⚠️ KG step failed (continuing):", e)

    # Retrieve docs
    docs = retrieve_docs(query, k=4)

    # Try QA models with failover
    models = parse_model_list("GROQ_MODELS", DEFAULT_QA_MODELS)
    last_err = None
    for m in models:
        try:
            # Generate answer with PRIOR history injected
            ans = answer_with_model(m, query, docs, history, verdict)
            cleaned = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()

            # **Persist this turn to memory** so follow-ups (e.g., “how old was she?”) work
            history.add_user_message(query)
            history.add_ai_message(cleaned)

            return cleaned
        except Exception as e:
            last_err = e
            print(f"⚠️ QA model failed '{m}': {e}")
            time.sleep(0.8)
            continue

    print("❌ All QA models failed:", last_err)
    return "😞 Our model endpoints are busy right now. Please try again shortly."


HF_MODEL_ID = os.getenv("HF_MODEL_ID", "arun86/hate-offensive-suicidal-bert1")
_classifier = None

def get_classifier():
    """
    Load the HF model once, on CPU, with anonymous download.
    Uses the writable cache dir you already set (HF_HOME).
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    if AutoTokenizer is None or AutoModelForSequenceClassification is None or pipeline is None:
        raise RuntimeError("transformers is not available. Check requirements install on Azure.")

    # Force anonymous (token=None) so old/expired tokens can't break downloads
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=None)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID, token=None)
    _classifier = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,           # CPU
        truncation=True
    )
    try:
        print("HF id2label:", mdl.config.id2label)
    except Exception:
        pass
    return _classifier

def _normalize_label(raw_label: str, id2label: Optional[Dict]=None) -> str:
    """
    Map model outputs to exactly: 'neither' | 'offensive' | 'hate' | 'suicidal'
    Handles LABEL_i, friendly strings, and your repo's 'hate_speech'.
    """
    lab = (raw_label or "").strip().lower()

    # LABEL_i -> id2label if available
    if lab.startswith("label_") and lab[6:].isdigit():
        idx = int(lab[6:])
        mapped = None
        if isinstance(id2label, dict):
            mapped = id2label.get(idx) or id2label.get(str(idx))
        lab = (str(mapped) if mapped is not None else lab).lower()

    # normalize common variants
    table = {
        "hate_speech": "hate",
        "neutral": "neither",
        "safe": "neither",
    }
    return table.get(lab, lab)

# ----------------- Routes -----------------
@app.route("/")
def root_ok():
    return "ok", 200

@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/warmup")
def warmup():
    try:
        build_tfidf_index()
        # optional ping to Groq
        try:
            groq_chat_with_failover(
                [{"role": "user", "content": "ping"}],
                parse_model_list("GROQ_MODELS", DEFAULT_QA_MODELS),
                temperature=0, max_tokens=4
            )
        except Exception as e:
            print("ℹ️ Warmup Groq failed (non-fatal):", e)
        return "warmed", 200
    except Exception as e:
        return f"warmup error: {e}", 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    # Use your frontend’s session_id (currentUserId). If absent, fall back to cookie/new UUID.
    session_id = data.get("session_id") or request.cookies.get("sid") or str(uuid4())
    try:
        response_text = ask_bot(query, session_id)
        resp = jsonify({"response": response_text, "session_id": session_id})
        if not request.cookies.get("sid"):
            resp.set_cookie("sid", session_id, max_age=60*60*24*7, httponly=False, samesite="Lax")
        return resp
    except Exception as e:
        print("❌ /ask error:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/moderate", methods=["POST"])
def moderate():
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Missing 'content'"}), 400

    try:
        clf = get_classifier()
        out = clf(content)
        top = out[0] if isinstance(out, list) else out

        id2label = None
        try:
            id2label = getattr(clf.model.config, "id2label", None)
        except Exception:
            pass

        label = _normalize_label(top.get("label"), id2label=id2label)
        # ensure one of our four
        if label not in {"neither", "offensive", "hate", "suicidal"}:
            # last resort: simple heuristics to fit one of the four
            t = content.lower()
            if any(k in label for k in ["suicid", "self-harm"]) or "kill myself" in t:
                label = "suicidal"
            elif "hate" in label:
                label = "hate"
            elif label in {"toxic","abusive","harassment","insult","offense","offencive"}:
                label = "offensive"
            else:
                label = "neither"

        return jsonify({"is_safe": (label == "neither"), "label": label}), 200

    except Exception as e:
        log.exception("Moderation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/test-cors", methods=["GET"])
def test_cors():
    return jsonify({"message": "CORS is working!"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
