import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
llm = get_hf_llm(temperature = 0.9)
genai_docs = r"E:\review_intern\1.chatbot\Knowledge-Langchain\rag_langchain_application\data_source"
#genai_docs = "./data_source/generative_ai"
# Chains
genai_chain = build_rag_chain (llm, data_dir=genai_docs, data_type="pdf")
# App FastAPI
app = FastAPI( 
    title="LangChain Server", 
    version="1.0", 
    description = "A simple api server using Langchain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Routes FastAPI
@app.get("/check")
async def check():
    return {"status": "ok"}
@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai (inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}
# Langserve Routes Playground
add_routes (app,
            genai_chain,
            playground_type="default",
            path=r"E:\review_intern\1.chatbot\Knowledge-Langchain\rag_langchain_application\data_source\generative_ai")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)