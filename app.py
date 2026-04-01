from fastapi import FastAPI, Request, Response
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Query(BaseModel):
    question: str


app = FastAPI()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

def send_whatsapp_message(to, messgae):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"

    }

    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }

    requests.post(url, headers=headers, json=data)

print("Loading PDF...")

#loader

db = None

def load_db():
    global db

    if db is None:
        loader = PyPDFLoader("Test.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, embeddings)

    return db

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key= os.getenv("GROQ_API_KEY")
    )


print("SYSTEM READY!")

def ask_pdf_ai(question):
    db_instance = load_db()

    docs = db_instance.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below:

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
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }

    response = requests.post(url, headers=headers, json=payload)
    print("SEND RESPONSE:", response.text)


@app.get("/webhook")
async def verify(request: Request):
    params = request.query_params

    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    print("TOKEN FROM META:", verify_token)

    # Only respond when BOTH are present
    if verify_token and challenge:
        if verify_token == "mytoken":
            return Response(content=challenge, media_type="text/plain")

    # For all other cases → return 403 (NOT JSON)
    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    print("INCOMING:", data)

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
        sender = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]

        #  For now simple reply
        reply = ask_pdf_ai(message)

        send_message(sender, reply)

    except Exception as e:
        print("ERROR:", e)

    return {"status": "ok"}





if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
