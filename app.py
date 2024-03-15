import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import tiktoken
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate



########################## pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.vectorstores import Pinecone as pinecone_vector
from pinecone import Pinecone


OPENAI_KEY = st.secrets['OPENAI_KEY']
PINECONE_KEY = st.secrets['PINECONE_KEY']

st.set_page_config(page_title="BrandChat", page_icon="🦜", layout="wide")
os.environ['PINECONE_API_KEY'] = PINECONE_KEY

pinecone_api_key = os.environ.get(PINECONE_KEY)
pinecone  = Pinecone(api_key=pinecone_api_key)

css='''
<style>
    section.main > div {max-width:956px}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

with open('./styles/1_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
            <div class="section_title">
            <span class="highlight">Creverse BrandChat </span>
            </div>
            """,unsafe_allow_html=True)

st.markdown("""
            <div style="font-size: 15px; padding: 10px;">Brandbook을 보면서 궁금했거나, 좀 더 알아보고 싶었던 것이 있으셨나요? 
            여기 <span style="font-weight: bold;">BrandChatbot</span>을 통해 질문해주세요</div>
            """, unsafe_allow_html=True)

avatars = {"human": "user", "ai": "assistant"}


st.chat_message(avatars['ai']).write('저는 BrandChat 입니다. 궁금하신 사항이 있으실 경우 말씀주세요')


# Create embeddings and store in vectordb
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model = "text-embedding-ada-002")


vectorstore = pinecone_vector.from_existing_index(index_name="test", embedding=embeddings_model)
vectorstore2 = vectorstore.as_retriever()


class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)




# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferWindowMemory(k =5, memory_key="chat_history", chat_memory=msgs, return_messages=True)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
tmpl= ChatPromptTemplate.from_template(template)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=OPENAI_KEY, temperature=0.3, streaming=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=vectorstore2, memory=memory, verbose=True
)


for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

user_query = st.chat_input("궁금하신게 있으시면 언제든 물어봐주세요!")

if user_query :
    st.chat_message("user").markdown(user_query)
    with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[stream_handler])
