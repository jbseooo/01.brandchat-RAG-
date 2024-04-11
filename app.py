import streamlit as st
import os
from pinecone import Pinecone

from langchain_community.vectorstores import Pinecone as pinecone_vector
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate,PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pymongo.mongo_client import MongoClient

OPENAI_KEY = st.secrets['OPENAI_KEY']
PINECONE_KEY = st.secrets['PINECONE_KEY']
MONGO_URI = st.secrets['uri']

st.set_page_config(page_title="BrandChat", page_icon="🦜", layout="wide")
os.environ['PINECONE_API_KEY'] = PINECONE_KEY

pinecone_api_key = os.environ.get(PINECONE_KEY)
pinecone  = Pinecone(api_key=pinecone_api_key)
URI = 
@st.cache_resource
def init_connection():
    return MongoClient(MONGO_URI)

db = init_connection()

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


embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model = "text-embedding-ada-002")
index2 = pinecone.Index("test")

bm25_encoder = BM25Encoder().load("./data/tokenized_corpus.json")
hybrid_retriever= PineconeHybridSearchRetriever(
    embeddings=embeddings_model, sparse_encoder=bm25_encoder, index=index2, top_k=int(2) , alpha=float(0.5))

## vector db load
# vectorstore = pinecone_vector.from_existing_index(index_name="test", embedding=embeddings_model)
# vectorstore2 = vectorstore.as_retriever()


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

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=OPENAI_KEY, temperature=0.3, streaming=True)



msgs = StreamlitChatMessageHistory(key= 'chat_history')
# memory 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, memory_key='chat_history', chat_memory=msgs, return_messages=False)

# chat history 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# messages 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []




# prompt 설정
template = """
저는 크레버스 FSO Brand MKT 본부에서 만들어진 BrandChat 입니다.
Remember the chat history when you answer

Chat History:
{chat_history}
Follow Up Input: {question}
"""


QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question","chat_history"])


system_template = """
저는 크레버스 FSO Brand MKT 본부에서 만들어진 BrandChat 입니다.
Remember the chat history when you answer
Chat History:
{chat_history}

----------------
{context}"""

# chat prompt templates
messages = [
SystemMessagePromptTemplate.from_template(system_template),
HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# display chat

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# retrievalchain 설정
qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=hybrid_retriever,
    memory=st.session_state.memory,
    verbose=True,
    get_chat_history = lambda h:h,
    return_source_documents=False,
    rephrase_question = False,
    condense_question_prompt = QA_PROMPT,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    chain_type = 'stuff',
)


# React to user input
if userquery := st.chat_input("BrandPlaybook에 대해 궁금하신 점이 있다면 편하게 질문해주세요!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(userquery)
    # user message 추가
    st.session_state.messages.append({"role": "user", "content": userquery})
    qu = {'question':userquery}
    db.ST_question.question.insert_one(qu)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(userquery, callbacks=[stream_handler])

        # assistant message 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        # memory 추가
        # st.session_state.memory.save_context({'input:':userquery}, {'output':response})
