import os
from dotenv import load_dotenv
load_dotenv()
from src.llm_config import get_chat_model, get_embeddings_model
from src.data_loader import load_and_split_pdf
from src.vector_store_manager import VectorStoreManager
from src.agent_tools import AgentTools
from src.explainer_agent import PDFAgentExplainer


def run_pdf_explainer_app():
    # 1. Configuration and Setup
    pdf_path = "PostgreSQL_(Postgres).pdf" # Ensure this file exists in the root directory

    # Create a dummy PDF if it doesn't exist for initial testing
    if not os.path.exists(pdf_path):
        print(f"'{pdf_path}' not found. Creating a dummy PDF for demonstration.")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_path)
        c.drawString(100, 750, "This is a sample document about OpenAI and LangChain.")
        c.drawString(100, 730, "OpenAI provides powerful language models like GPT-4o.")
        c.drawString(100, 710, "LangChain allows easy integration and building complex LLM applications.")
        c.drawString(100, 690, "Agents in LangChain can use tools to perform various tasks.")
        c.save()
        print(f"Dummy '{pdf_path}' created.")
    else:
        print(f"Using existing PDF: {pdf_path}")

    # Initialize LLM and Embeddings using OpenAI specific configurations
    chat_model = get_chat_model(model_name="deepseek/deepseek-r1:free") # You can change to "gpt-3.5-turbo" or "gpt-4o-mini"
    embeddings = get_embeddings_model()

    # 2. Data Loading and Vector Store Management
    print("\n--- Preparing PDF Data for Agent ---")
    documents = load_and_split_pdf(pdf_path)
    if not documents:
        print("Error: No documents loaded from PDF. Exiting.")
        return

    vector_manager = VectorStoreManager(embeddings_model=embeddings)
    vector_manager.create_or_load_vector_store(documents=documents)
    retriever = vector_manager.get_retriever()

    # 3. Initialize Agent Tools
    print("\n--- Initializing Agent Tools ---")
    tools_instance = AgentTools(retriever=retriever, chat_model=chat_model)
    agent_tools = [
        tools_instance.explain_pdf_content,
        tools_instance.general_knowledge_search
    ]

    # 4. Initialize and Run the PDF Explainer Agent
    print("\n--- Starting PDF Explainer Agent ---")
    pdf_explainer_agent = PDFAgentExplainer(chat_model=chat_model, tools=agent_tools)

    print("\nType your questions about the PDF or general topics. Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the PDF Explainer. Goodbye!")
            break

        pdf_explainer_agent.query_pdf(user_query)

    # Optional: Clean up the vector store on exit
    # vector_manager.delete_vector_store()

if __name__ == "__main__":
    run_pdf_explainer_app()