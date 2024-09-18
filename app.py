import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/hello', methods=['POST'])
def hello():
    req = request.form.get('req')

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()
    text_Splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    splits = text_Splitter.split_documents(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name= "all-MiniLM-L6-v2")
    db2 = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory="./chroma_db_new")
    db3 = Chroma(persist_directory="./chroma_db_new", embedding_function=embedding_function)

    result1 = db3.similarity_search(req, k=1)
    if req:
        print('Request for hello page received with req=%s' % req)
        return render_template('hello.html', req = result1)
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return redirect(url_for('index'))

if __name__ == '__main__':
   app.run()