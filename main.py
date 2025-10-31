import tempfile
import os
from dotenv import load_dotenv
from pytubefix import YouTube
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=_ipn0gG8OgI"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not os.path.exists("transcript.txt"):
    yt = YouTube(YOUTUBE_VIDEO)
    audio = yt.streams.filter(only_audio=True).first()

    whisper_model = whisper.load_model("turbo", "cuda")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcript.txt", "w") as file:
            file.write(transcription)
else:
    with open("transcript.txt", "r") as file:
        transcription = file.read()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", google_api_key = GEMINI_API_KEY)
template = """
Answer the question based on the context below. If you can't answer 
the question, reply "I don't know".

Context: {context}

Question: {question}
"""
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)

# try:
#     rs = chain.invoke({
#         "context": transcription,
#         "question": "How to play reroll camp 3-cost? Any other notable strategies?"
#     })
#     print(rs)
# except Exception as e:
#     print("Error:", e)

loader = TextLoader("transcript.txt")
text_documents = loader.load()
# print(text_documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
splitted = text_splitter.split_documents(text_documents)
# print(splitted)
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001", google_api_key = GEMINI_API_KEY)
# embedded_query = embeddings.embed_query("How to play reroll camp 3-cost?")

vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding=embeddings)

setup = RunnableParallel(context = vectorstore.as_retriever(), question = RunnablePassthrough())

chain = (
    setup
    | prompt
    | model
    | parser
)

rs = chain.invoke("How to play reroll camp 3-cost?")
print(rs)