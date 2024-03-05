import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


#import RAG_pipeline
import LawSage_pipeline_GGUF


class QueryItem(BaseModel):
    query: str
    auth_token: str

app = FastAPI()


@app.post("/LAW-SAGE-GGUF")
async def LAW_SAGE_llamacpp_request(query_item: QueryItem):
    query = query_item.query 
    auth_token = query_item.auth_token
    result = LawSage_pipeline_GGUF.get_lawsage_llama_cpp_response(query)
    return {"result": result}


@app.post("/RAG")
async def RAG_request(query_item: QueryItem):
    query = query_item.query 
    auth_token = query_item.auth_token
    result = RAG_pipeline.get_RAG_response(query)
    return {"result": result}


@app.get("/CHECK", status_code=200)
async def read_root():
    return {
        "connection": True,
        "message": "Connection successful!"}


if __name__ == '__main__':
    uvicorn.run(app, port=8001, host='192.168.1.2')