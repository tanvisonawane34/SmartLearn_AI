import streamlit as st
import whisper
import tempfile
import os
import re
import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

# NEW: YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="SmartLearn AI",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM UI STYLE
# =========================================================

st.markdown(
    """
    <style>
    /* =====================================================
       SMARTLEARN AI - PRESENTATION THEME
       Lavender + Purple + Teal + Warm Gold
       ===================================================== */

    .main-title {
        padding: 34px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #E8F1FF 0%, #EEF0FF 48%, #F8FBFF 100%);
        border: 1px solid #C9D8F3;
        text-align: center;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(42, 72, 150, 0.10);
    }

    .main-title h1 {
        color: #6F63B8;
        font-size: 44px;
        margin-bottom: 7px;
        letter-spacing: -0.7px;
    }

    .main-title h3 {
        margin-top: 4px;
        color: #3F4658;
        font-size: 22px;
    }

    .main-title p {
        color: #6B7280;
        font-size: 16px;
        margin: 9px auto 0 auto;
        max-width: 850px;
    }

    .feature-card {
        padding: 11px 13px;
        margin: 7px 0;
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #E4E1EF;
        color: #3F4658;
        transition: all 0.2s ease;
    }

    .feature-card:hover {
        background: #F1EEFA;
        border-color: #CFC7E8;
        transform: translateX(2px);
    }

    .source-card {
        padding: 17px 19px;
        border-radius: 15px;
        background: linear-gradient(135deg, #F8F6FC 0%, #FFFFFF 100%);
        border: 1px solid #E2DDED;
        margin: 8px 0 12px 0;
        box-shadow: 0 4px 13px rgba(45, 71, 130, 0.07);
    }

    .source-card .source-title {
        color: #7668B8;
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .source-card .source-text {
        color: #72798A;
        font-size: 14px;
        margin: 0;
    }

    .workflow-strip {
        padding: 16px 18px;
        border-radius: 16px;
        background: linear-gradient(90deg, #F4F0FB, #F8F3F7, #EFFAF8);
        border: 1px solid #E4DEED;
        text-align: center;
        color: #4B5563;
        font-size: 14px;
        margin: 8px 0 22px 0;
        box-shadow: 0 4px 14px rgba(53, 72, 135, 0.06);
    }

    .workflow-strip b {
        color: #7567B5;
    }

    .section-label {
        color: #665A9F;
        font-size: 25px;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .summary-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #F1EDFA, #F8F5FB);
        border: 1px solid #DDD5EA;
        border-left: 5px solid #8878C7;
        margin: 10px 0 18px 0;
        color: #46505F;
        line-height: 1.65;
        box-shadow: 0 4px 13px rgba(54, 82, 160, 0.06);
    }

    .note-card {
        padding: 13px 16px;
        border-radius: 11px;
        background: #EEF9F7;
        border: 1px solid #CFE9E4;
        border-left: 5px solid #18A6A6;
        margin: 8px 0;
        color: #46505F;
        line-height: 1.5;
    }

    .mcq-card {
        padding: 19px 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #F6F1FB, #FFFFFF);
        border: 1px solid #DDD2EB;
        border-left: 5px solid #7567D9;
        margin: 13px 0;
        box-shadow: 0 4px 13px rgba(91, 75, 155, 0.06);
    }

    .mcq-card h4 {
        color: #6D5AA5;
        margin-top: 0;
    }

    .mcq-card p {
        color: #4B5563;
        margin: 7px 0;
    }

    .revision-card {
        padding: 13px 16px;
        border-radius: 11px;
        background: #FFF8EE;
        border: 1px solid #F0DEC1;
        border-left: 5px solid #E09A2D;
        margin: 8px 0;
        color: #5B4630;
    }

    .footer-card {
        text-align: center;
        color: #68758D;
        padding: 20px 10px 12px 10px;
        line-height: 1.7;
    }

    /* Streamlit controls */
    div.stButton > button {
        border-radius: 11px;
        border: 1px solid #D7CFE8;
        background: linear-gradient(135deg, #7567C8, #8A70C7);
        color: white;
        font-weight: 700;
        min-height: 44px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(56, 78, 170, 0.14);
    }

    div.stButton > button:hover {
        border-color: #7562B5;
        color: white;
        background: linear-gradient(135deg, #6E62BE, #8269C4);
        box-shadow: 0 6px 15px rgba(56, 78, 170, 0.22);
        transform: translateY(-1px);
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
        border: 1px solid #DDD9E5;
        background: #FDFBFF;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #8B7CC4;
        box-shadow: 0 0 0 1px #8B7CC4;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #F7F3FB, #FBF7FA);
        border: 1px dashed #CFC5DE;
        border-radius: 12px;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        border: 1px solid #DDD9E5;
        background: #FFFDFF;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F1ECFA 0%, #F8F5FB 55%, #EDF9F6 100%);
        border-right: 1px solid #E1DCE8;
    }

    [data-testid="stSidebar"] h1 {
        color: #7668B8;
    }

    [data-testid="stSidebar"] h3 {
        color: #4D465F;
    }

    hr {
        border-color: #E0DAE7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE
# =========================================================

if "transcript" not in st.session_state:
    st.session_state["transcript"] = None

if "summary" not in st.session_state:
    st.session_state["summary"] = None

if "notes" not in st.session_state:
    st.session_state["notes"] = None

if "mcqs" not in st.session_state:
    st.session_state["mcqs"] = None

if "revision" not in st.session_state:
    st.session_state["revision"] = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎓 SmartLearn AI")
    st.caption("Video Summarization & Intelligent Notes Generator")

    st.divider()

    st.markdown("### ✨ Features")

    st.markdown(
        '<div class="feature-card">🎙️ Speech-to-Text</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="feature-card">✨ AI Summary</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="feature-card">📚 Intelligent Notes</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="feature-card">🧠 MCQs</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="feature-card">📄 PDF Study Material</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="feature-card">🎯 Exam Revision</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🔄 How It Works")
    st.write("1. Upload a video or paste a YouTube link")
    st.write("2. Generate the transcript")
    st.write("3. Generate summary and notes")
    st.write("4. Generate MCQs")
    st.write("5. Download study material as PDF")

    st.divider()

    st.caption("AI-powered educational learning assistant")

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🎓 SmartLearn AI</h1>
        <h3>Video Summarization & Intelligent Notes Generator</h3>
        <p>
            Transform educational videos into transcripts, summaries,
            intelligent notes, MCQs and exam revision material.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if st.button(
    "🔄 Start New Lecture",
    use_container_width=True
):
    st.session_state.clear()
    st.rerun()

st.markdown(
    """
    <div class="workflow-strip">
        🎥 <b>Video</b>
        &nbsp;→&nbsp;
        🎙️ <b>Transcript</b>
        &nbsp;→&nbsp;
        ✨ <b>Summary</b>
        &nbsp;→&nbsp;
        📚 <b>Notes</b>
        &nbsp;→&nbsp;
        🧠 <b>MCQs</b>
        &nbsp;→&nbsp;
        📄 <b>PDF</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# WHISPER
# =========================================================

@st.cache_resource
def load_whisper():

    return whisper.load_model("base")


# =========================================================
# SUMMARIZER
# =========================================================

@st.cache_resource
def load_summarizer():

    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# =========================================================
# QUESTION GENERATOR
# =========================================================

@st.cache_resource
def load_question_model():

    model_name = "valhalla/t5-small-qg-hl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# =========================================================
# GENERATE QUESTION
# =========================================================

def make_question(
    tokenizer,
    model,
    sentence
):

    answer = None

    patterns = [

        r"^(.+?)\s+is\s+(.+)$",

        r"^(.+?)\s+are\s+(.+)$",

        r"^(.+?)\s+was\s+(.+)$",

        r"^(.+?)\s+were\s+(.+)$",

        r"^(.+?)\s+has\s+(.+)$",

        r"^(.+?)\s+have\s+(.+)$"

    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            sentence,
            re.IGNORECASE
        )

        if match:

            answer = match.group(1).strip()

            break

    if not answer:

        words = sentence.split()

        if len(words) >= 3:

            answer = " ".join(
                words[:3]
            )

        else:

            return None, None

    prompt = (
        "answer: "
        + answer
        + " context: "
        + sentence
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    question_ids = model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4
    )

    question = tokenizer.decode(
        question_ids[0],
        skip_special_tokens=True
    )

    question = question.strip()

    if not question:

        return None, None

    return question, answer


# =========================================================
# YOUTUBE VIDEO
# =========================================================

st.markdown(
    """
    <div class="source-card">
        <div class="source-title">🔗 YouTube Lecture</div>
        <p class="source-text">Paste an educational YouTube lecture link and convert its speech into a transcript.</p>
    </div>
    """,
    unsafe_allow_html=True
)

youtube_url = st.text_input(
    "Paste YouTube video link here"
)

if youtube_url:

    if st.button("📥 Process YouTube Video", use_container_width=True):

        with st.spinner(
            "Getting YouTube transcript..."
        ):

            try:

                # Extract YouTube video ID
                match = re.search(
                    r"(?:v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
                    youtube_url
                )

                if not match:

                    st.error(
                        "Please enter a valid YouTube video link."
                    )

                else:

                    video_id = match.group(1)

                    # Get transcript directly from YouTube captions
                    youtube_api = YouTubeTranscriptApi()

                    fetched_transcript = youtube_api.fetch(
                        video_id,
                        languages=["en", "hi", "mr"]
                    )

                    # Convert transcript into normal text
                    transcript = " ".join(
                        snippet.text
                        for snippet in fetched_transcript
                    )

                    if transcript.strip():

                        st.session_state["transcript"] = transcript

                        st.session_state.pop(
                            "summary",
                            None
                        )

                        st.session_state.pop(
                            "notes",
                            None
                        )

                        st.session_state.pop(
                            "mcqs",
                            None
                        )

                        st.session_state.pop(
                            "revision",
                            None
                        )

                        st.success(
                            "YouTube transcript generated successfully!"
                        )

                    else:

                        st.warning(
                            "No transcript was available for this video."
                        )

            except Exception as e:

                st.error(
                    "Could not get the YouTube transcript."
                )

                st.info(
                    "This video may not have accessible captions."
                )


# =========================================================
# VIDEO UPLOAD
# =========================================================

st.markdown(
    """
    <div class="source-card">
        <div class="source-title">📹 Upload Your Video</div>
        <p class="source-text">Upload an educational video file and use Whisper AI to generate its transcript.</p>
    </div>
    """,
    unsafe_allow_html=True
)

video_file = st.file_uploader(
    "Choose an educational video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)

if video_file is not None:

    st.success(
        "Video uploaded successfully!"
    )

    st.video(video_file)

    if st.button("🎙️ Generate Transcript", use_container_width=True):

        with st.spinner(
            "Loading Whisper AI..."
        ):

            whisper_model = load_whisper()

        with st.spinner(
            "Converting speech into text..."
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_video:

                temp_video.write(
                    video_file.read()
                )

                video_path = temp_video.name

            result = whisper_model.transcribe(
                video_path
            )

            transcript = result["text"]

            os.remove(video_path)

        st.session_state["transcript"] = transcript

        st.session_state.pop(
            "summary",
            None
        )

        st.session_state.pop(
            "notes",
            None
        )

        st.session_state.pop(
            "mcqs",
            None
        )

        st.success(
            "Transcript generated successfully!"
        )


# =========================================================
# TRANSCRIPT
# =========================================================

if "transcript" in st.session_state:

    st.markdown('<div class="section-label">📝 Transcript</div>', unsafe_allow_html=True)

    st.text_area(
        "Generated Transcript",
        st.session_state["transcript"],
        height=300
    )

    st.divider()

    # =====================================================
    # SUMMARY
    # =====================================================

    if st.button("✨ Generate AI Summary", use_container_width=True):

        with st.spinner(
            "Loading AI summarization model..."
        ):

            tokenizer, summarizer_model = (
                load_summarizer()
            )

        with st.spinner(
            "Generating AI summary..."
        ):

            text = st.session_state["transcript"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            summary_ids = summarizer_model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                do_sample=False
            )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.session_state["summary"] = summary

        st.success(
            "AI summary generated successfully!"
        )


    if "summary" in st.session_state:

        st.markdown('<div class="section-label">📌 AI Summary</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="summary-card">
                <b>📌 Generated Summary</b><br><br>
                {st.session_state["summary"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # =====================================================
    # INTELLIGENT NOTES
    # =====================================================

    st.markdown('<div class="section-label">📚 Intelligent Notes</div>', unsafe_allow_html=True)

    if st.button("📝 Generate Study Notes", use_container_width=True):

        with st.spinner(
            "Generating intelligent notes..."
        ):

            tokenizer, summarizer_model = (
                load_summarizer()
            )

            text = st.session_state["transcript"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            notes_ids = summarizer_model.generate(
                inputs["input_ids"],
                max_length=180,
                min_length=40,
                do_sample=False
            )

            notes_text = tokenizer.decode(
                notes_ids[0],
                skip_special_tokens=True
            )

        sentences = notes_text.split(".")

        clean_notes = []

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:

                clean_notes.append(
                    sentence
                )

        st.session_state["notes"] = clean_notes

        st.success(
            "Study notes generated successfully!"
        )


    if "notes" in st.session_state:

        for note in st.session_state["notes"]:

            st.markdown(
                f"""
                <div class="note-card">
                    📌 {note}
                </div>
                """,
                unsafe_allow_html=True
            )


    st.divider()


    # =====================================================
    # MCQs
    # =====================================================

    st.markdown('<div class="section-label">🧠 Questions & MCQs</div>', unsafe_allow_html=True)

    if st.button("🎯 Generate MCQs", use_container_width=True):

        with st.spinner(
            "Generating MCQs from the video..."
        ):

            question_tokenizer, question_model = (
                load_question_model()
            )

            transcript = (
                st.session_state["transcript"]
            )

            sentences = re.split(
                r"[.!?]",
                transcript
            )

            useful_sentences = []

            for sentence in sentences:

                sentence = sentence.strip()

                if len(sentence.split()) >= 6:

                    useful_sentences.append(
                        sentence
                    )

            generated_mcqs = []

            for sentence in useful_sentences:

                if len(generated_mcqs) >= 5:

                    break

                question, correct_answer = make_question(
                    question_tokenizer,
                    question_model,
                    sentence
                )

                if not question:

                    continue

                if not correct_answer:

                    continue

                possible_answers = []

                for other_sentence in useful_sentences:

                    if other_sentence == sentence:

                        continue

                    match = re.match(
                        r"^(.+?)\s+(?:is|are|was|were|has|have)\s+(.+)$",
                        other_sentence,
                        re.IGNORECASE
                    )

                    if match:

                        candidate = (
                            match.group(1).strip()
                        )

                        if (
                            candidate.lower()
                            != correct_answer.lower()
                        ):

                            if candidate not in possible_answers:

                                possible_answers.append(
                                    candidate
                                )

                if len(possible_answers) < 3:

                    for other_sentence in useful_sentences:

                        words = other_sentence.split()

                        if len(words) >= 2:

                            candidate = (
                                " ".join(words[:3])
                            )

                            if (
                                candidate.lower()
                                != correct_answer.lower()
                            ):

                                if candidate not in possible_answers:

                                    possible_answers.append(
                                        candidate
                                    )

                if len(possible_answers) < 3:

                    continue

                wrong_answers = (
                    possible_answers[:3]
                )

                options = [
                    correct_answer,
                    wrong_answers[0],
                    wrong_answers[1],
                    wrong_answers[2]
                ]

                random.shuffle(
                    options
                )

                correct_letter = ""

                for index, option in enumerate(
                    options
                ):

                    if option.lower() == (
                        correct_answer.lower()
                    ):

                        correct_letter = chr(
                            65 + index
                        )

                generated_mcqs.append(
                    {
                        "question": question,
                        "options": options,
                        "answer": correct_letter
                    }
                )

            st.session_state["mcqs"] = (
                generated_mcqs
            )

        if generated_mcqs:

            st.success(
                f"{len(generated_mcqs)} MCQs generated successfully!"
            )

        else:

            st.warning(
                "No MCQs could be generated from "
                "the available transcript."
            )


    if "mcqs" in st.session_state:

        for i, mcq in enumerate(
            st.session_state["mcqs"],
            start=1
        ):

            st.markdown(
                f"""
                <div class="mcq-card">
                    <h4>Q{i}. {mcq['question']}</h4>
                    <p><b>A)</b> {mcq['options'][0]}</p>
                    <p><b>B)</b> {mcq['options'][1]}</p>
                    <p><b>C)</b> {mcq['options'][2]}</p>
                    <p><b>D)</b> {mcq['options'][3]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                f"✅ Correct Answer: {mcq['answer']}"
            )


    # =====================================================
    # PDF
    # =====================================================

    st.markdown('<div class="section-label">📄 Study Material PDF</div>', unsafe_allow_html=True)

    if st.button("📥 Generate PDF", use_container_width=True):

        pdf_file = (
            "SmartLearn_AI_Study_Material.pdf"
        )

        document = SimpleDocTemplate(
            pdf_file
        )

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "SmartLearn AI - Study Material",
                styles["Title"]
            )
        )

        story.append(
            Spacer(1, 20)
        )

        story.append(
            Paragraph(
                "Transcript",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                escape(st.session_state["transcript"]),
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(1, 20)
        )

        if "summary" in st.session_state:

            story.append(
                Paragraph(
                    "AI Summary",
                    styles["Heading2"]
                )
            )

            story.append(
                Paragraph(
                    escape(st.session_state["summary"]),
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 20)
            )

        if "notes" in st.session_state:

            story.append(
                Paragraph(
                    "Intelligent Notes",
                    styles["Heading2"]
                )
            )

            for note in st.session_state["notes"]:

                story.append(
                    Paragraph(
                        "• " + escape(note),
                        styles["BodyText"]
                    )
                )

                story.append(
                    Spacer(1, 5)
                )

        if "mcqs" in st.session_state:

            story.append(
                Paragraph(
                    "MCQs",
                    styles["Heading2"]
                )
            )

            for i, mcq in enumerate(
                st.session_state["mcqs"],
                start=1
            ):

                story.append(
                    Paragraph(
                        f"Q{i}. {escape(mcq['question'])}",
                        styles["BodyText"]
                    )
                )

                for index, option in enumerate(
                    mcq["options"]
                ):

                    letter = chr(
                        65 + index
                    )

                    story.append(
                        Paragraph(
                            f"{letter}) {escape(option)}",
                            styles["BodyText"]
                        )
                    )

                story.append(
                    Paragraph(
                        f"Correct Answer: "
                        f"{escape(mcq['answer'])}",
                        styles["BodyText"]
                    )
                )

                story.append(
                    Spacer(1, 15)
                )

        document.build(
            story
        )

        with open(
            pdf_file,
            "rb"
        ) as file:

            pdf_data = file.read()

        st.success(
            "PDF generated successfully!"
        )

        st.download_button(
            label="⬇️ Download Study Material PDF",

            data=pdf_data,

            file_name=
            "SmartLearn_AI_Study_Material.pdf",

            mime="application/pdf"
        )


    # =====================================================
    # EXAM REVISION MODE
    # =====================================================

    st.divider()

    st.markdown('<div class="section-label">🎯 Exam Revision Mode</div>', unsafe_allow_html=True)

    st.caption(
        "Generate a quick revision sheet from your lecture."
    )

    if st.button(
        "⚡ Generate Revision Sheet",
        use_container_width=True
    ):

        with st.spinner(
            "Preparing your exam revision sheet..."
        ):

            tokenizer, summarizer_model = (
                load_summarizer()
            )

            text = st.session_state["transcript"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            revision_ids = summarizer_model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=60,
                do_sample=False
            )

            revision_text = tokenizer.decode(
                revision_ids[0],
                skip_special_tokens=True
            )

        sentences = revision_text.split(".")

        revision_points = []

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:

                revision_points.append(
                    sentence
                )

        st.session_state["revision"] = revision_points

        st.success(
            "Revision sheet generated successfully!"
        )

    if st.session_state["revision"]:

        st.markdown(
            "### 📌 Important Revision Points"
        )

        for point in st.session_state["revision"]:

            st.markdown(
                f"""
                <div class="revision-card">
                    ⭐ {point}
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-card">
        🎓 <b>SmartLearn AI</b><br>
        AI-powered learning assistant for students
    </div>
    """,
    unsafe_allow_html=True
)
