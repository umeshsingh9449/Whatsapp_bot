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
import Functions as F

load_dotenv()


embeddings = FastEmbedEmbeddings()
user_dbs = {}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key= os.getenv("GROQ_API_KEY")
    )


class Query(BaseModel):
    question: str


app = FastAPI()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")




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
    data = await requests.json()
    
    print("Incoming:", data)

    try:
        message_data = data["entry"][0]["changes"][0]["values"]["messages"][0]
        sender = message_data["from"]

        # case 1: USER SENT PDF
        if "document" in message_data:
            doc = message_data["document"]
            media_id = doc["id"]
            send_message(sender, "PDF recived. Processing....")
            
            file_path = download_pdf(media_id)
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunks_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(chunks, embeddings)
            db = FAISS.from_documents(chunks, embeddings)
            user_db[sender] = db 
            send_message(sender, " PDF Processed! Now ask you Questions.")

        # Case2 : User SENT Text:
        elif "text" in message_data:
            user_message = message_data["text"]["body"]

            if sender in user_dbs:
                db = user_dbs[sender]
                reply = ask_pdf_ai(db, user_message)
            else:
                reply = "Pleader upload a PDF first."

            send_message(sender, reply)

    except Exception as e:
        print("ERROR:", e)

    return {"status": "ok"}
    






if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
