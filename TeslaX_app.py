import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Streamlit 클라우드에서는 st.secrets 사용
if os.getenv("STREAMLIT_SHARE"):
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    # 로컬 개발 환경에서는 .env 파일 사용
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 페이지 설정
st.set_page_config(
    page_title="TeslaX 챗봇",
    page_icon="🚗",
    layout="wide"
)

# Supabase 클라이언트 초기화
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI 설정
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    temperature=0.7, 
    model_name="gpt-3.5-turbo-16k",
    openai_api_key=OPENAI_API_KEY
)

# Supabase 벡터 스토어 설정
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# 대화형 체인 설정
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=True
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사이드바
with st.sidebar:
    st.title("🚗 TeslaX 챗봇")
    st.markdown("""
    ### 사용 방법
    1. 질문을 입력하세요
    2. 테슬라 매뉴얼을 기반으로 답변해드립니다
    3. 자세한 설명이 필요하면 추가 질문을 해주세요
    """)
    
    if st.button("대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 메인 화면
st.title("🚗 TeslaX 매뉴얼 도우미")

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])
        if "source_documents" in message:
            with st.expander("참고 문서"):
                for doc in message["source_documents"]:
                    st.markdown(f"- {doc.metadata.get('source', '알 수 없는 출처')}")

# 사용자 입력
if prompt := st.chat_input("테슬라에 대해 궁금한 점을 물어보세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                # QA 체인으로 답변 생성
                result = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                response = result["answer"]
                source_docs = result["source_documents"]
                
                # 답변 표시
                st.markdown(response)
                
                # 출처 표시
                if source_docs:
                    with st.expander("참고 문서"):
                        for doc in source_docs:
                            st.markdown(f"- {doc.metadata.get('source', '알 수 없는 출처')}")
                
                # 대화 기록 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": source_docs
                })
                st.session_state.chat_history.append((prompt, response))
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.error("죄송합니다. 다시 시도해주세요.") 