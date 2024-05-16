import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=11000, chunk_overlap=1100)
    chunks = text_splitter.split_text(text)
    return chunks

def embedding_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")


def chain_prompt_question():

    prompt_template = """
    Give a detailed response to the provided question, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def get_context_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    context = new_db.similarity_search(user_question)

    chain = chain_prompt_question()

    
    response = chain(
        {"input_documents":context, "question": user_question}
        , return_only_outputs=True)

    st.write("Answer: ", response["output_text"])



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with pdf(s)")

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    btn = st.button("Submit & Process")
    if btn:
        with st.spinner("Processing..."):
            text = get_pdf_text(pdf_docs)
            text_chunks = chunk_text(text)
            embedding_vector(text_chunks)
            st.success("Done")

    user_question = st.text_input("Ask a question from the pdf")

    if user_question:
        get_context_question(user_question)

    
        



if __name__ == "__main__":
    main()

    
        





