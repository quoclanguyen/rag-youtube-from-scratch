# YouTube Video Q&A RAG from scratch

A web-based RAG (Retrieval-Augmented Generation) application that allows you to ask questions about YouTube videos.

## Demo
<video src="https://github.com/user-attachments/assets/2a832422-2786-432f-8359-d024d9956ef9" controls width="100%"></video>

## Features

- 🎬 Download and transcribe YouTube videos (<30 minutes)
- 🤖 AI-powered question answering using Gemini
- 💾 Automatic transcript caching
- 🚀 Fast subsequent queries after initial processing

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for Whisper transcription)
- Gemini API key

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

If you're using CUDA, also install PyTorch with CUDA support:

```bash
pip install -r requirements_cuda.txt
```

2. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Enter YouTube URL**: Paste a YouTube video link in the sidebar (video must be <30 minutes)
2. **Process Video**: Click "Process Video" to download and transcribe the audio
3. **Ask Questions**: Once processed, enter your questions in the main area
4. **Get Answers**: The AI will answer based on the video transcript

## Notes

- First-time video processing may take a few minutes for transcription
- Transcripts are cached locally as `transcript_{video_id}.txt`
- Subsequent questions on the same video are much faster!

## Troubleshooting

- **CUDA errors**: Make sure you have CUDA installed and compatible PyTorch version
- **API errors**: Verify your `GEMINI_API_KEY` is set correctly in the `.env` file
- **Video too long**: Only videos shorter than 30 minutes are supported
