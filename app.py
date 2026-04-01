from fastapi import FastAPI, Request
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


@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()

    try:
        message = data["entry"][0]["changes"][0]["Values"]["messages"][0]
        user_message = message["text"]["body"]
        sender = message["from"]
    except:
        return {"status": "no message"}

    
    db_instance = load_db
    docs = db_instance.similarity_search(user_message, k=5)
    context = "\n".join([d.page_content for d in docs])

    response = llm.invoke(f"""
    
    You are a Professional HR assitant.
    Understand the PDF, Check the Details.

    Context:
    {context}

    Question:
    {user_message}
    
    
    """)

    answer = response.content 
    send_whatsapp_message(sender, answer)
    return{"status": "ok"}


@app.get("/webhook")
def verify(request: Request):
    params = request.query_params

    if params.get("hub.verify_token") == "mytoken":
        return param.get("hub.challenge")

    return "verification failed"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
