SmartLearn AI

Video Summarization & Intelligent Notes Generator

SmartLearn AI is a Streamlit-based educational application that converts lecture videos into useful study material.

Features

🎥 Upload an educational video

🔗 Process educational YouTube videos

🎙️ Convert speech into text using Whisper AI

✨ Generate an AI-based summary

📚 Generate intelligent study notes

🧠 Generate MCQs for practice

📄 Generate PDF study material

🎯 Generate exam revision material

🔄 Start a new lecture without restarting the application

Technologies Used

Python

Streamlit

OpenAI Whisper

Hugging Face Transformers

DistilBART

T5 Question Generation

ReportLab

yt-dlp

FFmpeg

Project Workflow

Video / YouTube Link
        ↓
   Speech-to-Text
      (Whisper)
        ↓
    Transcript
        ↓
   AI Summary
        ↓
 Intelligent Notes
        ↓
      MCQs
        ↓
    PDF Study Material
        ↓
   Exam Revision

Installation

1. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\Activate.ps1

2. Install Python dependencies

pip install -r requirements.txt

3. Install FFmpeg

FFmpeg must be installed separately and available in the system PATH because Whisper uses it to process audio/video.

Check the installation with:

ffmpeg -version

4. Run the application

streamlit run app.py

The application will open in the browser at the local Streamlit address.

How to Use

Open SmartLearn AI.

Paste a YouTube educational video link or upload a video file.

Generate the transcript.

Generate the AI summary.

Generate intelligent notes.

Generate MCQs.

Download the generated PDF study material.

Use Exam Revision for quick revision.

Project Structure

SmartLearn_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── logo.png              # Optional
│
└── sample_videos/
    └── test_lecture.mp4      # Optional demo video

Important Notes

The first run may take longer because AI models need to be downloaded.

Whisper requires FFmpeg.

Large video files and downloaded AI models should not be uploaded to GitHub.

Keep API keys and other secrets out of the project repository.

Future Scope

Support for more video sources

Better multilingual transcription and summaries

More advanced question generation

Personalized learning recommendations

Topic-wise progress tracking

Cloud deployment