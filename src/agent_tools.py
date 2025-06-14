from langchain.tools import tool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

class AgentTools:
    def __init__(self, retriever, chat_model):
        self.retriever = retriever
        self.chat_model = chat_model
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> Runnable:
        """Creates a RAG chain for answering questions based on retrieved documents."""
        qa_system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Keep the answer concise and to the point.

            Context: {context}
            """),
            ("user", "{input}")
        ])
        document_chain = create_stuff_documents_chain(self.chat_model, qa_system_prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        return retrieval_chain

    @tool
    def explain_pdf_content(self, query: str) -> str:
        """
        Use this tool to answer questions about the content of the PDF document.
        Input should be a clear and concise question about the PDF.
        """
        print(f"\n--- Agent using 'explain_pdf_content' tool for query: '{query}' ---")
        try:
            response = self.qa_chain.invoke({"input": query})
            return response["answer"]
        except Exception as e:
            return f"An error occurred while explaining the PDF content: {e}"

    @tool
    def general_knowledge_search(self, query: str) -> str:
        """
        Use this tool to answer general knowledge questions not directly from the PDF.
        This tool simulates an external search.
        """
        print(f"\n--- Agent using 'general_knowledge_search' tool for query: '{query}' ---")
        # In a real application, this would integrate with a search API (e.g., Google Search, Tavily)
        return f"Simulated general knowledge search for '{query}': The answer is outside the PDF's scope or requires external research."


if __name__ == "__main__":
    from llm_config import get_chat_model, get_embeddings_model
    from vector_store_manager import VectorStoreManager
    from data_loader import load_and_split_pdf
    import os

    chat_model = get_chat_model()
    embeddings = get_embeddings_model()
    pdf_path = "Maximo76_Designer431_Report_Development_Guide_Rev8.pdf" # Ensure this file exists

    if not os.path.exists(pdf_path):
        print(f"Please place an 'example.pdf' in the root directory for testing.")
        exit()

    documents = load_and_split_pdf(pdf_path)
    if not documents:
        print("No documents loaded, cannot proceed with tool testing.")
        exit()

    vector_manager = VectorStoreManager(embeddings_model=embeddings)
    vector_manager.create_or_load_vector_store(documents=documents)
    retriever = vector_manager.get_retriever()

    tools_instance = AgentTools(retriever=retriever, chat_model=chat_model)

    print("\nTesting PDF explanation tool:")
    pdf_query = "What is LangChain mentioned in the document?"
    pdf_answer = tools_instance.explain_pdf_content(pdf_query)
    print(f"PDF Explanation: {pdf_answer}")

    print("\nTesting General Knowledge tool:")
    general_query = "What is the capital of France?"
    general_answer = tools_instance.general_knowledge_search(general_query)
    print(f"General Knowledge: {general_answer}")