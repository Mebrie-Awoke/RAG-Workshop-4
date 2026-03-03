import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

load_dotenv()

# 1 Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# 2 Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# 3 Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4 Store in vector database
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k":3})

# 5 LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

print("\nRAG System Ready\n")

while True:
    query = input("Ask: ")

    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    you are a personal AI assistant.
Answer the question using the rules below:
1. Answer ONLY using the provided context.
2. If the context does not contain enough information, say:
   "The provided documents do not contain sufficient information to answer this."
3. Do NOT add outside knowledge.
4. Keep the answer clear, structured, and concise.
5. If helpful, format the response in bullet points.
6. Always cite the source of the information in the format: [Document X], where X is the document number from the provided context.
7. do not hallucinate or make up information. If the answer is not in the context, say: 
   "I don't know, pls refer other AI assistants or human experts for more information."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n", response.content)