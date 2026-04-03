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


embeddings = FastEmbedEmbeddings()
user_dbs = {}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key= os.getenv("GROQ_API_KEY")
    )


class Query(BaseModel):
    question: str


app = FastAPI()

def send_whatsapp_message(to, message):
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





print("SYSTEM READY!")

def ask_pdf_ai(question):

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

def download_pdf(media_id):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}"
    }
    #set1: Get media URL
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    res = requests.get(url,headers=headers).json()
    media_rl = res["url"]

    #step2: Download file
    file_data = requests.get(media_url, headers=headers).content
    file_path = f"/tmp/{media_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(file_data)

    return file_path



if __name__ == "__main__":
    main_function()

