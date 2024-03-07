import os
from langchain.llms import CTransformers
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "3.conversational_chat_bot"

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create LLM model
model_type = 'mistral'
llm = CTransformers(model="C:/Users/arasu/Workspace/Projects/GenAI/models/Mistral_quantized/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    model_type=model_type)

# # Initialize instructor embeddings using the Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name='C:/Users/arasu/Workspace/Projects/GenAI/embeddings/sentence-transformers_all-mpnet-base-v2/')
db_path = "vector_db"

def create_vector_db():
    # Load data from pdf
    raw_text = ""
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )
    for root, dirs, files in os.walk("docs"):        
        for file in files:
            if file.endswith(".pdf"):
                pdf = PdfReader("./docs/"+file)
                for i, page in enumerate(pdf.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content
    texts = text_splitter.split_text(raw_text)

    # Create a  vector database from 'text'
    vector_db = Chroma.from_texts(texts,embeddings,persist_directory=db_path)
    vector_db.persist()
    vector_db = None 

def get_qa_chain():
    # Load the vector database from the local folder
    vector_db = Chroma(persist_directory=db_path, embedding_function = embeddings)

    # Create a retriever for querying the vector database
    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    chat_history = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True,k=2

                                            )
    conv_qa = ConversationalRetrievalChain.from_llm(llm=llm,  # USing ChatOpenAI as the LLM
                                                    retriever=retriever,  # # set the vectorstore to do similarity search
                                                    memory=chat_history,
                                                    # Provide the buffer memory object to pass the conversation history to LLM.
                                                )
    return  conv_qa