__import__('pysqlite3').connect(':memory:')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import tempfile
import os
from langchain_classic.chains import RetrievalQA


# 제목
st.title("ChatPDF")
st.write("---")

#OpenAI API Key 입력
api_key = st.text_input("OpenAI API Key를 입력해주세요", type="password")

#파일 업로드
uploaded_file = st.file_uploader("파일을 올려주세요", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 되면 동작하는 코드 
if uploaded_file is not None:
    with st.spinner("PDF를 분석하는 중입니다... (파일 크기에 따라 시간이 걸릴 수 있어요)"):
        pages = pdf_to_document(uploaded_file)

        #splitter
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 500,
            chunk_overlap = 50,
            length_function = len,
            is_separator_regex = False,
        )

        texts = text_splitter.split_documents(pages)

        #embedding
        embeddings_model = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")

        # load it into Chroma
        db = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings_model
            )

    st.success("PDF 분석 완료!")
    
    #Question
    st.header("PDF 질문하기")
    question = st.text_input("질문을 입력해주세요")

    if st.button("질문하기"):
       with st.spinner("답변을 생성하는 중입니다..."):
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(),
                return_source_documents=True
            )
            result = qa_chain.invoke({"query": question})
            st.write(result["result"])
