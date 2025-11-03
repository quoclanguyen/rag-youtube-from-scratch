import streamlit as st
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

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ Please set GEMINI_API_KEY in your .env file")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YouTube Q&A Demo",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube Video Q&A Demo")
st.markdown("Upload a YouTube video (<30 minutes) and ask questions about its content!")

# Initialize session state
if "transcription_complete" not in st.session_state:
    st.session_state.transcription_complete = False
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None


def get_video_id(url):
    """Extract video ID from YouTube URL"""
    try:
        yt = YouTube(url)
        return yt.video_id
    except Exception as e:
        st.error(f"Error extracting video ID: {e}")
        return None


def check_video_length(url):
    """Check if video is less than 30 minutes"""
    try:
        yt = YouTube(url)
        length_seconds = yt.length
        length_minutes = length_seconds / 60
        return length_minutes <= 30, length_minutes, yt.title
    except Exception as e:
        st.error(f"Error checking video length: {e}")
        return False, 0, None


@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached)"""
    return whisper.load_model("base", "cuda")


def transcribe_video(url, video_id):
    """Download and transcribe YouTube video"""
    try:
        yt = YouTube(url)
        audio = yt.streams.filter(only_audio=True).first()
        
        if not audio:
            st.error("Could not find audio stream")
            return None
        
        whisper_model = load_whisper_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = audio.download(output_path=tmpdir)
            transcription = whisper_model.transcribe(file_path, fp16=False)["text"].strip()
            
            # Save transcript with video ID as identifier
            transcript_file = f"transcript_{video_id}.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            return transcription
    except Exception as e:
        st.error(f"Error transcribing video: {e}")
        return None


def setup_rag_chain(transcription, video_id):
    """Setup RAG chain with embeddings and vectorstore"""
    try:
        # Save transcript to file
        transcript_file = f"transcript_{video_id}.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        # Load and split documents
        loader = TextLoader(transcript_file)
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        splitted = text_splitter.split_documents(text_documents)
        
        # Create embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vectorstore = DocArrayInMemorySearch.from_documents(splitted, embedding=embeddings)
        
        # Setup RAG chain
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY
        )
        
        template = """Answer the question based on the context below. If you can't answer 
the question, reply "I don't know".

Context: {context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        parser = StrOutputParser()
        
        setup = RunnableParallel(
            context=vectorstore.as_retriever(),
            question=RunnablePassthrough()
        )
        
        chain = setup | prompt | model | parser
        
        return vectorstore, chain
    except Exception as e:
        st.error(f"Error setting up RAG chain: {e}")
        return None, None


# Sidebar for video input
with st.sidebar:
    st.header("📥 Video Input")
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube video URL here"
    )
    
    if youtube_url:
        if st.button("Process Video", type="primary"):
            with st.spinner("Checking video..."):
                video_id = get_video_id(youtube_url)
                if video_id:
                    is_valid, length, title = check_video_length(youtube_url)
                    
                    if not is_valid:
                        st.error(f"❌ Video is {length:.1f} minutes long. Please use a video shorter than 30 minutes.")
                    else:
                        st.success(f"✅ Video: {title}")
                        st.info(f"Duration: {length:.1f} minutes")
                        
                        # Check if transcript already exists
                        transcript_file = f"transcript_{video_id}.txt"
                        if os.path.exists(transcript_file):
                            st.info("📄 Using cached transcript")
                            with open(transcript_file, "r", encoding="utf-8") as f:
                                transcription = f.read()
                        else:
                            with st.spinner("Downloading audio..."):
                                pass
                            with st.spinner("Transcribing video (this may take a few minutes)..."):
                                transcription = transcribe_video(youtube_url, video_id)
                        
                        if transcription:
                            st.session_state.transcription_complete = True
                            st.session_state.current_video_id = video_id
                            
                            with st.spinner("Setting up RAG system..."):
                                vectorstore, chain = setup_rag_chain(transcription, video_id)
                                if vectorstore and chain:
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.chain = chain
                                    st.success("✅ Ready to answer questions!")
                        else:
                            st.error("Failed to transcribe video")

# Main area for Q&A
if st.session_state.transcription_complete and st.session_state.chain:
    st.header("💬 Ask Questions")
    
    # Display video info
    if youtube_url:
        try:
            yt = YouTube(youtube_url)
            st.info(f"📹 **Current Video:** {yt.title}")
        except:
            pass
    
    # Question input
    question = st.text_input(
        "Enter your question",
        placeholder="e.g., What is the main topic discussed in the video?",
        help="Ask any question about the video content"
    )
    
    if question:
        if st.button("Get Answer", type="primary"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")
    
    # Instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Enter YouTube URL**: Paste a YouTube video link in the sidebar (video must be <30 minutes)
        2. **Process Video**: Click "Process Video" to download and transcribe the audio
        3. **Ask Questions**: Once processed, enter your questions in the main area
        4. **Get Answers**: The AI will answer based on the video transcript
        
        **Note**: The first time processing a video may take a few minutes for transcription.
        Subsequent questions on the same video are much faster!
        """)

