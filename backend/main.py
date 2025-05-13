from fastapi import FastAPI, Request, HTTPException
from backend.rag_engine import answer_question
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/ask")
async def ask(req: Request):
    try:
        body = await req.json()
        question = body.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        answer = answer_question(question)
        return {"answer": answer}
    except Exception as e:
        logging.exception("Error processing request")
        raise HTTPException(status_code=500, detail="Internal Server Error")
