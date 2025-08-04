import os, re, networkx as nx
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from uuid import uuid4

# Langchain Imports
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

load_dotenv()

# === Flask Setup ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust as needed

# === Document Setup ===
pdf_filename = os.path.join("pdfs", "GriefBot.pdf")
loader = PyPDFLoader(pdf_filename)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# === Vector Store ===
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === LLM Setup ===
api_key = os.getenv("GROQ")
if not api_key:
    raise ValueError("Missing GROQ API key")
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# === Knowledge Graph Setup ===
KG = nx.DiGraph()

triple_extraction_prompt = PromptTemplate.from_template(
    """Extract all subject-relationship-object triples from the text below.

Text: "{text}"

Format each triple like: (subject, relation, object)

Only return the list of triples. Do not explain.

Triples:"""
)
triple_extraction_chain = LLMChain(llm=llm, prompt=triple_extraction_prompt)

def extract_triples(text):
    try:
        raw_output = triple_extraction_chain.invoke({"text": text})["text"]
        pattern = r"\(\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*,\s*['\"]?([\w\s]+?)['\"]?\s*\)"
        matches = re.findall(pattern, raw_output)
        return [(s.strip(), r.strip(), o.strip()) for s, r, o in matches]
    except Exception as e:
        print("⚠️ Triple extraction failed:", e)
        return []

def store_kg(triples):
    for s, r, o in triples:
        KG.add_edge(s, o, label=r)

def get_kg_facts(entity=None):
    facts = []
    for u, v, d in KG.edges(data=True):
        if not entity or entity in (u, v):
            facts.append(f"{u} {d['label']} {v}")
    return "\n".join(facts)

# === Validator Chain ===
validation_prompt = PromptTemplate.from_template(
    """You are a validation engine for a grief support chatbot.

User's message: "{query}"

Only respond with one of:
- "valid"
- "nonsensical"
- "unrelated"
- "illogical"
- "offensive"
- "harmful"

Your answer:"""
)
validator_chain = LLMChain(llm=llm, prompt=validation_prompt)

# === System Prompt ===
system_prompt = (
    """You are a compassionate human therapist who supports clients through grief and emotional healing. 
You speak like a real person — calm, caring, and gently conversational. You avoid robotic or overly formal language.

Here's how you respond:
- Use short, simple, human-sounding sentences.
- Acknowledge emotion first.
- Offer support or reflection, not lectures.
- Ask gentle follow-up questions when needed.
- Only include facts from provided context if relevant.
- Never guess if you’re unsure — just say so kindly.

Avoid sounding like a bot or giving long, polished essays.

Example:
Client: I lost my dad two months ago.
Therapist: I'm really sorry you're going through that. Losing a parent can feel so heavy. Do you want to share what you’ve been feeling lately?

Client: He loved gardening. He spent hours in the backyard.
Therapist: That sounds like a beautiful memory. Do you still find yourself thinking about those moments?

Now continue the conversation naturally.

{context}"""
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# === Session History ===
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# === Core Ask Bot ===
def ask_bot(query, session_id="default"):
    verdict = validator_chain.invoke({"query": query})["text"].strip().lower()

    if verdict in ["nonsensical", "illogical"]:
        return "🤖 That seems a bit confusing. Could you clarify what you meant?"
    elif verdict in ["unrelated", "offensive", "harmful"]:
        return "🤖 I'm here to help with grief-related concerns. Could you ask something else?"

    triples = extract_triples(query)
    store_kg(triples)

    docs = retriever.get_relevant_documents(query)
    if not docs or sum(len(doc.page_content.strip()) for doc in docs) < 100:
        return "🤖 I'm not sure how to answer that based on what I know."

    history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": query, "chat_history": history.messages},
        config={"configurable": {"session_id": session_id}}
    )

    return response["answer"]

# === API Routes ===
@app.route("/")
def home():
    return "GriefBot API is running."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    session_id = data.get("session_id") or str(uuid4())

    try:
        response = ask_bot(query, session_id)
        return jsonify({"response": response, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test-cors", methods=["GET"])
def test_cors():
    return jsonify({"message": "CORS is working!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
