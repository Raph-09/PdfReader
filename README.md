## Chat PDF Documentation

### Overview

This Streamlit application allows users to interact with PDF documents by uploading them, processing the text, and asking questions based on the content of the PDFs. The application leverages the Google Generative AI for embeddings and question-answering, and uses FAISS for efficient similarity search.

### Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.7+
- Streamlit
- LangChain
- Google Generative AI Python Client
- PyPDF2
- Python Dotenv
- FAISS

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/chat-pdf.git
   cd chat-pdf
   ```

2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory of your project and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

### Running the Application

To start the application, run:
```bash
streamlit run app.py
```

### Application Workflow

1. **Upload PDF Files**
   - Users can upload multiple PDF files using the file uploader widget.
   - After uploading, click on the "Submit & Process" button to start processing the PDFs.

2. **Process PDF Text**
   - The uploaded PDFs are read, and their text content is extracted.
   - The extracted text is then split into chunks using a recursive character splitter to ensure optimal processing.

3. **Generate Embeddings**
   - The text chunks are converted into embeddings using the Google Generative AI embedding model.
   - The embeddings are stored in a FAISS vector database for efficient similarity search.

4. **Ask Questions**
   - Users can input questions related to the content of the uploaded PDFs.
   - The application retrieves the most relevant context from the FAISS vector database.
   - A detailed response is generated using the Google Generative AI chat model and displayed to the user.

### Code Explanation

#### Importing Libraries and Configuring API
```python
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

#### Functions

- **`get_pdf_text(pdf_docs)`**: Extracts text from the uploaded PDF documents.
- **`chunk_text(text)`**: Splits the extracted text into smaller chunks for processing.
- **`embedding_vector(text_chunks)`**: Converts text chunks into embeddings and stores them in a FAISS vector database.
- **`chain_prompt_question()`**: Sets up the prompt template and initializes the question-answering chain.
- **`get_context_question(user_question)`**: Retrieves relevant context for the user's question and generates a detailed answer using the chat model.

#### Main Function

```python
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
```

### Conclusion

This application provides an intuitive interface for users to upload PDFs, process their content, and ask questions to extract relevant information. The integration with Google Generative AI ensures accurate and detailed responses based on the context of the PDFs.
