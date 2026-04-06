from fastapi import FastAPI, Request, Response
import os
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import Functions as F

load_dotenv()

embeddings = FastEmbedEmbeddings()
user_dbs = {}

app = FastAPI()


@app.get("/webhook")
async def verify(request: Request):
    params = request.query_params

    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    print("TOKEN FROM META:", verify_token)

    if verify_token and challenge:
        if verify_token == "mytoken":
            return Response(content=challenge, media_type="text/plain")

    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        print("Invalid JSON body:", e)
        return {"status": "ignored"}

    print("Incoming:", data)

    try:
        entry = data.get("entry") or []
        if not entry:
            return {"status": "ignored"}
        changes = (entry[0] or {}).get("changes") or []
        if not changes:
            return {"status": "ignored"}
        value = (changes[0] or {}).get("value") or {}

        messages = value.get("messages")
        if not messages:
            print("Non-message event:", value)
            return {"status": "ignored"}

        message_data = messages[0] if isinstance(messages, list) else None
        if not isinstance(message_data, dict):
            return {"status": "ignored"}

        sender = message_data.get("from")
        if not sender:
            return {"status": "ignored"}

        # Document upload
        if "document" in message_data:
            doc = message_data.get("document") or {}
            media_id = doc.get("id")
            if not media_id:
                return {"status": "ignored"}
            F.handle_document_upload(sender, str(media_id), embeddings, user_dbs)
            return {"status": "ok"}

        # Text message
        if "text" in message_data:
            text_obj = message_data.get("text") or {}
            user_message = text_obj.get("body")
            if user_message is None:
                return {"status": "ignored"}
            # Whitespace-only: do not send an empty or meaningless reply
            if not str(user_message).strip():
                return {"status": "ignored"}

            has_pdf = sender in user_dbs
            db = user_dbs.get(sender)
            reply = F.handle_text_message(str(user_message), has_pdf, db)
            reply = F.ensure_non_empty_reply(reply)
            F.send_message(sender, reply)
            return {"status": "ok"}

        return {"status": "ignored"}

    except Exception as e:
        print("ERROR:", e)
        return {"status": "error"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
