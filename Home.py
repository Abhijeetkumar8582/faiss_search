from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI
load_dotenv('.env')

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')
loader = PyPDFLoader("Abhijeet.pdf")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
chain = RetrievalQA.from_chain_type(llm=OpenAI(
          ), chain_type="stuff", retriever=db.as_retriever())
        

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        data = request.json
        
        user_input = data['user_input']
  
        final_result = chain.invoke(user_input)
        print("Test")
        return(final_result)
    except Exception as e:
        return jsonify({'error': str(e)})