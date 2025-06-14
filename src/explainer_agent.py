# retrieval_agent.py
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# === Setup ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PDF_PATH = "PostgreSQL_(Postgres).pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "deepseek/deepseek-r1:free"

# === Load PDF ===
loader = PyPDFLoader(PDF_PATH)
documents = loader.load_and_split()

# === Embed and Store ===
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# === Initialize Chat Model ===
llm = ChatOpenAI(
    model=CHAT_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# === Create RetrievalQA Chain ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant helping users understand technical PDF documentation.
Use the following context to answer the question concisely and clearly:

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# === Interactive CLI ===
print("\n--- PDF Q&A System (powered by DeepSeek) ---")
print("Type 'exit' to end the conversation.")

while True:
    query = input("\nYou: ")
    if query.lower() == 'exit':
        print("Goodbye!")
        break
    result = qa_chain.invoke({"query": query})
    print("\nAI:", result['result'])
