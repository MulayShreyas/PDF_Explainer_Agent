# PDF_Explainer_Agent
The **PDF Explainer Agent** is an AI-powered tool that reads and answers questions from PDF documents using LLMs. It's built using [LangChain](https://www.langchain.com/), [OpenRouter](https://openrouter.ai/) (e.g., DeepSeek, Mixtral, etc.), and HuggingFace embeddings for semantic search.

---

## 🚀 Features

- Load and process PDF files
- Embed text chunks using HuggingFace embeddings
- Search relevant content using FAISS vector store
- Answer user queries using an LLM via OpenRouter
- Interactive local UI using `streamlit`

---

## 📁 Folder Structure

```
pdf_explainer_agent/
│
├── main.py                   # Entry point of the application
├── data.pdf                  # Your PDF file (replace as needed)
├── requirements.txt          # Python dependencies
├── .env                      # API keys and config (OpenRouter, HuggingFace)
└── README.md                 # This file
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf_explainer_agent.git
cd pdf_explainer_agent
```

### 2. Create & Activate Virtual Environment

```bash
# Using Anaconda
conda create -n pdf-agent python=3.11
conda activate pdf-agent
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Up `.env` File

Create a `.env` file in the root directory:

```ini
OPENROUTER_API_KEY=your_openrouter_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

Then, open your browser at [http://localhost:8501](http://localhost:8501)

---

## 📦 Dependencies

Key Python packages used:

- `langchain`
- `openai` (for OpenRouter)
- `streamlit`
- `PyPDF2` or `pdfminer.six`
- `faiss-cpu`
- `sentence-transformers`
- `python-dotenv`

Install them all using:

```bash
pip install -r requirements.txt
```

---

## ✍️ Example Usage

1. Upload or hardcode a PDF (`data.pdf`)
2. Ask questions like:
   - “What is the purpose of the report?”
   - “Summarize section 3”
3. The agent will retrieve relevant chunks and generate an answer.

---

## 🔐 Security Notes

- Do **not** commit your `.env` file.
- Never expose API keys publicly.
- Use only trusted models in OpenRouter.

---

## 📚 References

- [LangChain Docs](https://docs.langchain.com/)
- [OpenRouter](https://openrouter.ai/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)

---

## 🙋‍♂️ Author

Built by [Shreyas Mulay](https://github.com/shreyas-mulay).  
Need help? Open an issue or reach out directly.

---

## 📝 License

MIT License – do whatever you want but give credit.
