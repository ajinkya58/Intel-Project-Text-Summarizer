from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def retriver_(text, embedding, query):
    temp_text_path = "data/file.txt"
    with open(temp_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    documents = TextLoader(
        temp_text_path,
        encoding="utf-8",
    ).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)


    retriever = FAISS.from_documents(texts, embedding).similarity_search(query)
    print(retriever)

    return retriever


