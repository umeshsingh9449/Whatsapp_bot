from collections import OrderedDict
from fastapi import FastAPI, Request, Response
import os
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import Functions as F

load_dotenv()

embeddings = FastEmbedEmbeddings()
user_dbs = {}

app = FastAPI()

# WhatsApp Cloud API: deduplicate by wamid (retries / duplicate deliveries).
_MAX_PROCESSED_WAMIDS = 8000
_processed_wamids: OrderedDict[str, None] = OrderedDict()


def _mark_wamid_processed(wamid: str) -> None:
    _processed_wamids[wamid] = None
    while len(_processed_wamids) > _MAX_PROCESSED_WAMIDS:
        _processed_wamids.popitem(last=False)


def _log_value_debug(value: dict) -> None:
    keys = sorted(value.keys())
    print(f"[webhook] value keys: {keys}")


def _log_non_message_value(value: dict) -> None:
    keys = sorted(value.keys())
    if "statuses" in value and "messages" not in value:
        print(f"[webhook] ignored: statuses-only or no messages (value keys: {keys})")
    elif not value.get("messages"):
        print(f"[webhook] ignored: no messages (value keys: {keys})")
    else:
        print(f"[webhook] ignored: empty messages list (value keys: {keys})")


def _should_process_root(data: dict) -> bool:
    obj = data.get("object")
    if obj != "whatsapp_business_account":
        print(f"[webhook] ignored: unexpected or missing object {obj!r}")
        return False
    return True


def _iter_message_events(data: dict):
    """Yield (value, message_dict) for each inbound message across all entry/changes."""
    for entry in data.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            _log_value_debug(value)
            messages = value.get("messages")
            if not isinstance(messages, list) or len(messages) == 0:
                _log_non_message_value(value)
                continue
            for msg in messages:
                if isinstance(msg, dict):
                    yield value, msg


def _normalize_phone_digits(s: str) -> str:
    return "".join(c for c in (s or "") if c.isdigit())


def _is_echo_from_business(from_id: str) -> bool:
    """Skip rare echo cases when WHATSAPP_BUSINESS_PHONE is set (digits match)."""
    configured = os.getenv("WHATSAPP_BUSINESS_PHONE")
    if not configured:
        return False
    return _normalize_phone_digits(from_id) == _normalize_phone_digits(configured)


def _validate_inbound_message(msg: dict) -> str | None:
    """
    Require id, from, type in (text, document), and matching payload keys.
    Returns 'text', 'document', or None.
    """
    if not isinstance(msg, dict):
        print("[webhook] ignored: message is not a dict")
        return None
    mid = msg.get("id")
    sender = msg.get("from")
    if not mid or not sender:
        print("[webhook] ignored: message missing id or from")
        return None
    mtype = msg.get("type")
    if mtype == "text":
        if "text" not in msg:
            print("[webhook] ignored: type text but no text payload")
            return None
        return "text"
    if mtype == "document":
        if "document" not in msg:
            print("[webhook] ignored: type document but no document payload")
            return None
        return "document"
    print(f"[webhook] ignored: unsupported message type {mtype!r}")
    return None


def _process_one_message(message_data: dict, kind: str) -> bool:
    """Run existing PDF / text logic. Returns True if handled (not skipped)."""
    sender = message_data.get("from")
    if not sender:
        return False

    if kind == "document":
        doc = message_data.get("document") or {}
        media_id = doc.get("id")
        if not media_id:
            print("[webhook] ignored: document missing media id")
            return False
        print(f"[webhook] processed: document from={sender}")
        F.handle_document_upload(str(sender), str(media_id), embeddings, user_dbs)
        return True

    # text
    text_obj = message_data.get("text") or {}
    user_message = text_obj.get("body")
    if user_message is None:
        print("[webhook] ignored: text body missing")
        return False
    if not str(user_message).strip():
        print("[webhook] ignored: text body empty/whitespace")
        return False

    has_pdf = sender in user_dbs
    db = user_dbs.get(sender)
    reply = F.handle_text_message(str(user_message), has_pdf, db)
    reply = F.ensure_non_empty_reply(reply)
    print(f"[webhook] processed: text from={sender}")
    F.send_message(sender, reply)
    return True


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
        print("[webhook] ignored: invalid JSON body:", e)
        return {"status": "ignored"}

    if not isinstance(data, dict):
        print("[webhook] ignored: root JSON is not an object")
        return {"status": "ignored"}

    print("[webhook] incoming payload (summary keys):", sorted(data.keys()))

    try:
        if not _should_process_root(data):
            return {"status": "ignored"}

        handled = 0
        for _value, msg in _iter_message_events(data):
            kind = _validate_inbound_message(msg)
            if not kind:
                continue

            wamid = msg.get("id")
            if wamid in _processed_wamids:
                print(f"[webhook] ignored: duplicate wamid {wamid!r}")
                continue

            sender = msg.get("from")
            if sender and _is_echo_from_business(str(sender)):
                print(
                    "[webhook] ignored: from matches WHATSAPP_BUSINESS_PHONE (echo guard)"
                )
                continue

            # Mark before heavy work so Meta retries do not double-reply.
            _mark_wamid_processed(str(wamid))

            if _process_one_message(msg, kind):
                handled += 1

        if handled:
            print(f"[webhook] done: handled {handled} message(s)")
            return {"status": "ok"}
        print("[webhook] done: nothing handled")
        return {"status": "ignored"}

    except Exception as e:
        print("[webhook] ERROR:", e)
        return {"status": "error"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
