import os
import re
import time
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Fallback when model returns nothing or whitespace (never send empty WhatsApp bodies).
EMPTY_REPLY_FALLBACK = "Sorry, I didn't catch that — could you say that again?"


def ensure_non_empty_reply(reply: str | None) -> str:
    s = (reply or "").strip()
    return s if s else EMPTY_REPLY_FALLBACK


def ask_pdf_ai(db, question: str) -> str:
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful assistant answering only from the document context below.
If the context does not contain enough information to answer, say so briefly and suggest what to ask instead.
Do not invent facts not supported by the context.

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content


def send_message(to, message):
    url = f"https://graph.facebook.com/v18.0/{os.getenv('PHONE_NUMBER_ID')}/messages"

    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    response = requests.post(url, headers=headers, json=payload)
    print("SEND RESPONSE:", response.text)


def download_pdf(media_id):
    headers = {"Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}"}
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    res = requests.get(url, headers=headers).json()
    media_url = res.get("url")
    if not media_url:
        raise ValueError("No media URL in Graph API response")

    file_data = requests.get(media_url, headers=headers).content
    file_path = f"/tmp/{media_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(file_data)

    return file_path


# Fast path: obvious greetings / acknowledgments without an LLM call (lower latency).
_OBVIOUS_CASUAL = re.compile(
    r"^(hi|hello|hey|hii|hlo|yo|sup|thanks|thank you|thx|ty|ok|okay|k|cool|nice|"
    r"good morning|good afternoon|good evening|gm|bye|goodbye|see you|later)\b",
    re.I,
)


def _is_obvious_casual(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2 and t.lower() in ("hi", "ok", "👍", "🙂", "😊"):
        return True
    if len(t) > 120:
        return False
    return bool(_OBVIOUS_CASUAL.match(t))


def classify_message(msg: str) -> str:
    """
    Returns exactly one label: 'casual' or 'document'.
    """
    prompt = f"""Reply with exactly one word, either casual or document — nothing else.

- casual: greetings, thanks, small talk, chit-chat, or messages not asking about file content.
- document: questions about a PDF/document, policies, facts to look up in uploaded material, or anything that needs the document to answer.

Message: {msg}
"""
    res = (llm.invoke(prompt).content or "").strip().lower()
    # Take first token; tolerate "casual." or "document-related"
    first = res.split()[0] if res else ""
    if first.startswith("casual"):
        return "casual"
    if first.startswith("document"):
        return "document"
    # Safe default: treat as document-related so we can offer upload or RAG
    return "document"


def casual_reply(msg: str) -> str:
    prompt = f"""You are a friendly WhatsApp assistant. The user wrote:
"{msg}"

Reply in 1–3 short sentences. Be warm and natural. Do not mention PDFs or documents unless they did.
Do not use bullet points. Sound human, not like a system message.
"""
    res = llm.invoke(prompt)
    return (res.content or "").strip()


def reply_when_no_pdf() -> str:
    return (
        "I don’t have a document from you yet. "
        "Send me a PDF and I can answer questions from it — or just chat with me in the meantime!"
    )


def handle_text_message(user_message: str, has_pdf_db: bool, db) -> str:
    """
    Intent-first routing: casual always gets a chat reply; document + DB uses RAG;
    document without DB asks for a PDF politely.
    """
    text = (user_message or "").strip()
    if not text:
        return EMPTY_REPLY_FALLBACK

    if _is_obvious_casual(text):
        intent = "casual"
    else:
        intent = classify_message(text)

    if intent == "casual":
        return casual_reply(text)

    if has_pdf_db and db is not None:
        return ask_pdf_ai(db, text)

    return reply_when_no_pdf()


def handle_document_upload(sender: str, media_id: str, embeddings, user_dbs: dict) -> None:
    """
    Download PDF, chunk, build FAISS, store under sender. Sends user-facing status messages.
    """
    send_message(sender, "Got it! I’m processing your PDF…")
    start = time.time()
    try:
        file_path = download_pdf(media_id)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            send_message(sender, "That PDF looks empty or I couldn’t read it. Try another file?")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)

        if not chunks:
            send_message(sender, "I couldn’t split that PDF into usable text. Try a different file?")
            return

        db = FAISS.from_documents(chunks, embeddings)
        user_dbs[sender] = db
        print("PDF processing time (s):", time.time() - start)
        send_message(
            sender,
            "All set! Ask me anything about the document — or just say hi if you want to chat.",
        )
    except Exception as e:
        print("PDF ERROR:", e)
        send_message(sender, "Something went wrong while processing the PDF. Please try again.")
