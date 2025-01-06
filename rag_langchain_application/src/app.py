import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

# App FastAPI
app = FastAPI( 
    title="LangChain Server", 
    version="1.0", 
    description = "A simple api server using Langchain's Runnable interfaces",
)

# Khởi tạo các biến global để lưu model và chain
llm = None
genai_chain = None

@app.on_event("startup")
async def startup_event():
    global llm, genai_chain
    # Khởi tạo model
    llm = get_hf_llm(temperature=0.9)
    # Khởi tạo chain
    genai_docs = os.path.join(os.path.dirname(__file__), "..", "data_source", "generative_ai")
    genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["+"],
)

#Routes FastAPI
@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    global genai_chain
    if genai_chain is None:
        return {"answer": "Model is still loading. Please try again in a moment."}
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# Langserve Routes Playground
@app.on_event("startup")
async def setup_langserve():
    global genai_chain
    if genai_chain is not None:
        add_routes(
            app,
            genai_chain,
            playground_type="default",
            path="/generative_ai"
        )