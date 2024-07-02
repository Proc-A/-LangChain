import os
# import _pillow_heif
from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

llm = Ollama(
    model="llama3",
    temperature=0
)

loader = UnstructuredFileLoader("D:\zadanie/kniga-receptov-dentex.pdf")
documents = loader.load()


text_splitter = CharacterTextSplitter(separator="/n",
                                      chunk_size=1000,
                                      chunk_overlap=200)

text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

knowledge_base = FAISS.from_documents(text_chunks, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())

question = input("Введите ваш вопрос:")
response = qa_chain.invoke({"query": question})
print(response["result"])
