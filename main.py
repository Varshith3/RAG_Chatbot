import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

# Initialize the LLM and Embeddings
llm = ChatCohere(model="command-a-03-2025", temperature=0, api_key=api_key)
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Load the SQuAD dataset
with open('train-v2.0.json', 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

# Preprocess the SQuAD data
documents = []
for entry in squad_data["data"]:
    title = entry.get("title", "")
    for paragraph in entry["paragraphs"][0:1]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answers = qa.get("answers", [])
            answer_texts = [a["text"] for a in answers]
            doc_text = f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswers: {answer_texts}"
            metadata = {
                "title": title,
                "context": context,
                "question": question,
                "answers": answer_texts
            }
        documents.append(Document(page_content=doc_text, metadata=metadata))
        

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Creating a vector store from the documents and saving the embeddings
db = FAISS.from_documents(documents, embeddings)



# Main function to retirve the answer from the vector store and generate a response from LLM
def get_answer(question):
    docs = db.similarity_search(question)
    prompt = f'''
    You are a helpful assistant. Answer the question based on the context provided.
    Question: {question}
    Context: {docs}
    Based on the context, provide a clear and concise answer to the question don't give multiple answers.
    If the information doesn't contain the answer, say "I don't have enough information to answer this question."
    '''
    response = llm.predict(prompt).strip()

    return response