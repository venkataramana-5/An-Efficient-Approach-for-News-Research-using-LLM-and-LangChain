import os
import streamlit as st
import pickle
import requests
import nltk
import re

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup

nltk.data.path.append(r"C:\Users\Snehita Bathula\AppData\Roaming\nltk_data")

# API Key
os.environ['OPENAI_API_KEY'] = 'XXXXXX.....'

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 News Research & Q&A Tool")

st.sidebar.header("🔗 Enter News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'documents' not in st.session_state:
    st.session_state.documents = []

def fetch_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.warning(f"⚠️ Failed to fetch content from: {url} (Status code: {response.status_code})")
    except Exception as e:
        st.error(f"❌ Error fetching {url}: {e}")
    return None

def process_articles(urls):
    documents = []
    for url in urls:
        html = fetch_content(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if len(text.strip()) < 100:
                st.warning(f"⚠️ Content too short or empty from URL: {url}")
            else:
                documents.append(Document(page_content=text, metadata={"source": url}))
        else:
            st.warning(f"⚠️ No content found at URL: {url}")
    return documents

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# Process articles
if st.sidebar.button("🔍 Process Articles") and urls:
    with st.spinner("Processing articles..."):
        docs = process_articles(urls)
        if docs:
            st.session_state.documents = docs
            st.session_state.vectorstore = create_vectorstore(docs)
            st.success("✅ Articles processed successfully!")
        else:
            st.error("❌ No valid content could be extracted from the provided URLs.")

# Prompt and LLM chain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
)
llm_chain = prompt | llm

# Helper functions for identifying special types of questions
def get_summary_request_index(question):
    match = re.search(r'url\s*(\d)', question.lower())
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(st.session_state.documents):
            return index
    return None

def get_keyword_from_question(question):
    match = re.search(r'keyword\s*:\s*(.+)', question.lower())
    if match:
        return match.group(1).strip()
    return None

# Main Q&A section
question = st.text_input("📝 Ask a question (e.g., 'Give me URL 1 summary' or 'Summarize based on keyword: gold'):")

if question:
    if st.session_state.vectorstore:
        # Check for summary by URL index
        summary_index = get_summary_request_index(question)

        if summary_index is not None:
            doc = st.session_state.documents[summary_index]
            summary_prompt = f"Please summarize the following article content:\n\n{doc.page_content}"
            with st.spinner("Generating summary..."):
                summary = llm.invoke(summary_prompt)
            st.subheader(f"📄 Summary of URL {summary_index + 1}")
            st.write(summary)
            st.markdown(f"🔗 [Original Article]({doc.metadata.get('source', '')})")

        # Check for keyword-based summary
        else:
            keyword = get_keyword_from_question(question)
            if keyword:
                matched_chunks = []
                for doc in st.session_state.documents:
                    if keyword.lower() in doc.page_content.lower():
                        matched_chunks.append(doc.page_content)

                if matched_chunks:
                    keyword_context = "\n\n".join(matched_chunks[:3])
                    keyword_prompt = f"Summarize the following content based on the keyword '{keyword}':\n\n{keyword_context}"
                    with st.spinner(f"Summarizing content related to '{keyword}'..."):
                        keyword_summary = llm.invoke(keyword_prompt)
                    st.subheader(f"🔑 Keyword-Based Summary: {keyword}")
                    st.write(keyword_summary)
                else:
                    st.warning(f"⚠️ No content found related to keyword: '{keyword}'")

            # General question
            else:
                with st.spinner("Thinking..."):
                    retrieved_docs = st.session_state.vectorstore.similarity_search(question, k=3)
                    if not retrieved_docs:
                        st.error("❌ No relevant information found. Try rephrasing your question.")
                    else:
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        sources = [doc.metadata.get("source", "Unknown source") for doc in retrieved_docs]
                        answer = llm_chain.invoke({"context": context, "question": question})

                        if answer.strip() == "":
                            st.warning("⚠️ The model could not generate an answer. Try refining your question.")
                        else:
                            st.subheader("📌 Answer:")
                            st.write(answer)

                            st.subheader("🔗 Sources:")
                            for src in set(sources):
                                st.markdown(f"[{src}]({src})")
    else:
        st.warning("⚠️ Please process articles first before asking questions.")
