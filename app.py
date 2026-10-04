import streamlit as st
import whisper
import tempfile
import os
import re
import random
from html import escape

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartLearn AI",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f7f2ff, #eefcff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #6c4ab6;
    font-weight: 800;
}

h2, h3 {
    color: #6845a8;
}

.subtitle {
    color: #667085;
    font-size: 18px;
    margin-bottom: 25px;
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(100, 70, 150, 0.10);
    margin-bottom: 15px;
}

.success-box {
    background: #e9fff5;
    border-left: 5px solid #20a779;
    padding: 14px;
    border-radius: 10px;
}

.info-box {
    background: #f1edff;
    border-left: 5px solid #7956c7;
    padding: 14px;
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "transcript": None,
    "summary": None,
    "notes": None,
    "mcqs": None,
    "revision": None,
    "source_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# RESET GENERATED CONTENT
# ============================================================

def reset_generated_content():

    st.session_state.summary = None
    st.session_state.notes = None
    st.session_state.mcqs = None
    st.session_state.revision = None


# ============================================================
# LOAD WHISPER ONLY ONCE
# ============================================================

@st.cache_resource(show_spinner="Loading Whisper AI...")
def load_whisper_model():

    return whisper.load_model("base")


# ============================================================
# LOAD SUMMARIZER ONLY ONCE
# ============================================================

@st.cache_resource(show_spinner="Loading AI Summary Model...")
def load_summary_model():

    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# ============================================================
# SAFE TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_youtube_video_id(url):

    if not url:
        return None

    url = url.strip()

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&\s]+)",
        r"(?:youtu\.be/)([^?\s]+)",
        r"(?:youtube\.com/shorts/)([^?\s]+)",
        r"(?:youtube\.com/embed/)([^?\s]+)",
        r"(?:youtube\.com/live/)([^?\s]+)",
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# ============================================================
# YOUTUBE TRANSCRIPT
# ============================================================

def get_youtube_transcript(video_id):

    api = YouTubeTranscriptApi()

    # First try preferred languages
    try:

        fetched = api.fetch(
            video_id,
            languages=["en", "hi", "mr"]
        )

        texts = []

        for item in fetched:

            text = getattr(item, "text", None)

            if text is None and isinstance(item, dict):
                text = item.get("text", "")

            if text:
                texts.append(text)

        transcript = " ".join(texts)

        if transcript.strip():
            return clean_text(transcript)

    except Exception:
        pass


    # Second method: list available transcripts
    try:

        transcript_list = api.list(video_id)

        available = list(transcript_list)

        if not available:
            return None


        # Preferred languages
        selected = None

        for language in ["en", "hi", "mr"]:

            for transcript in available:

                language_code = getattr(
                    transcript,
                    "language_code",
                    ""
                )

                if language_code == language:

                    selected = transcript

                    break

            if selected:
                break


        # If preferred language not found
        if selected is None:
            selected = available[0]


        fetched = selected.fetch()

        texts = []

        for item in fetched:

            text = getattr(item, "text", None)

            if text is None and isinstance(item, dict):
                text = item.get("text", "")

            if text:
                texts.append(text)


        transcript = " ".join(texts)

        if transcript.strip():
            return clean_text(transcript)

        return None

    except Exception as e:

        raise Exception(
            "YouTube transcript could not be accessed. "
            "The video may not have captions or YouTube may be "
            "blocking transcript access."
        ) from e


# ============================================================
# SAFE SUMMARY FUNCTION
# ============================================================

def generate_summary(text):

    text = clean_text(text)

    if not text:
        return "No transcript available."


    try:

        tokenizer, model = load_summary_model()

        # Keep input manageable
        text_for_model = text[:12000]

        inputs = tokenizer(
            text_for_model,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        input_length = inputs["input_ids"].shape[1]

        # Safe lengths
        max_length = min(
            150,
            max(30, input_length)
        )

        min_length = min(
            30,
            max(5, max_length // 3)
        )

        if min_length >= max_length:
            min_length = max(5, max_length // 2)


        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True
        )


        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )


        return clean_text(summary)


    except Exception as e:

        # Fallback instead of crashing
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 30
        ]

        if sentences:

            fallback = " ".join(sentences[:5])

            return clean_text(fallback)

        return text[:1000]


# ============================================================
# INTELLIGENT NOTES
# ============================================================

def generate_notes(text):

    text = clean_text(text)

    if not text:
        return "No notes available."


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    useful_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) >= 5:

            useful_sentences.append(sentence)


    if not useful_sentences:

        return "- " + text[:1000]


    # Maximum 10 useful points
    useful_sentences = useful_sentences[:10]


    notes = []

    for sentence in useful_sentences:

        sentence = sentence.strip()

        if not sentence.endswith("."):
            sentence += "."

        notes.append("- " + sentence)


    return "\n".join(notes)


# ============================================================
# MCQ HELPERS
# ============================================================

def shorten_answer(text, max_words=18):

    words = text.split()

    if len(words) > max_words:

        return " ".join(words[:max_words]) + "..."

    return text


# ============================================================
# EXTRACT QUESTION + ANSWER
# ============================================================

def extract_question_answer(sentence):

    sentence = clean_text(sentence)

    if len(sentence.split()) < 6:
        return None


    # Pattern:
    # Photosynthesis is the process...
    match = re.match(
        r"(.+?)\s+(is|are|was|were|has|have)\s+(.+)",
        sentence,
        re.IGNORECASE
    )


    if match:

        subject = match.group(1).strip(" ,.-")
        verb = match.group(2).lower()
        answer = match.group(3).strip(" ,.-")


        if len(subject.split()) <= 12 and len(answer.split()) >= 2:

            if verb in ["is", "was", "has"]:
                question_word = "What is"
            else:
                question_word = "What are"


            question = f"{question_word} {subject}?"


            return {
                "question": question,
                "answer": shorten_answer(answer)
            }


    return None


# ============================================================
# GENERATE MCQs
# ============================================================

def generate_mcqs(text):

    text = clean_text(text)

    if not text:
        return []


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    candidates = []


    # --------------------------------------------------------
    # First collect meaningful question-answer pairs
    # --------------------------------------------------------

    for sentence in sentences:

        result = extract_question_answer(sentence)

        if result:

            answer = result["answer"]

            if answer not in [
                item["answer"]
                for item in candidates
            ]:

                candidates.append(result)


    # --------------------------------------------------------
    # Fallback candidates from sentences
    # --------------------------------------------------------

    if len(candidates) < 4:

        for sentence in sentences:

            sentence = clean_text(sentence)

            words = sentence.split()

            if len(words) < 7:
                continue


            # Use first meaningful phrase
            answer = " ".join(words[:8])

            if len(answer) < 15:
                continue


            if answer not in [
                item["answer"]
                for item in candidates
            ]:

                candidates.append({
                    "question":
                        "Which statement is mentioned in the lecture?",
                    "answer": answer
                })


    # Remove duplicates
    unique_candidates = []

    seen = set()

    for item in candidates:

        answer_key = item["answer"].lower()

        if answer_key not in seen:

            seen.add(answer_key)

            unique_candidates.append(item)


    candidates = unique_candidates


    # Need at least 4 answers
    if len(candidates) < 4:

        return []


    random.seed(42)

    mcqs = []

    # Maximum 5 questions
    number_of_questions = min(5, len(candidates))


    for index in range(number_of_questions):

        current = candidates[index]

        correct_answer = current["answer"]


        # Get distractors from other answers
        other_answers = [
            item["answer"]
            for j, item in enumerate(candidates)
            if j != index
            and item["answer"] != correct_answer
        ]


        # Remove duplicates
        other_answers = list(dict.fromkeys(other_answers))


        if len(other_answers) < 3:
            continue


        distractors = other_answers[:3]


        options = [
            correct_answer,
            distractors[0],
            distractors[1],
            distractors[2]
        ]


        # Remove accidental duplicate options
        options = list(dict.fromkeys(options))


        if len(options) < 4:
            continue


        random.shuffle(options)


        correct_letter = "ABCD"[
            options.index(correct_answer)
        ]


        mcqs.append({
            "question": current["question"],
            "options": options,
            "answer": correct_letter,
            "correct_text": correct_answer
        })


    return mcqs


# ============================================================
# REVISION MODE
# ============================================================

def generate_revision(text):

    text = clean_text(text)

    if not text:
        return "No revision material available."


    summary = generate_summary(text)

    notes = generate_notes(text)


    revision = f"""
## 📚 Exam Revision

### 🔹 Quick Summary

{summary}

### 🔹 Important Points

{notes}

### 🔹 Last-Minute Revision

Focus on:

• Main definitions  
• Important concepts  
• Key terms  
• Processes and explanations  
• Examples mentioned in the lecture  
• Differences between important concepts  

### 🔹 Exam Tip

Read the summary first, then revise the important points and finally practice the MCQs.
"""


    return revision.strip()


# ============================================================
# PDF GENERATION
# ============================================================

def create_pdf(transcript, summary, notes, revision):

    temp_pdf = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    temp_pdf.close()


    doc = SimpleDocTemplate(
        temp_pdf.name,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER


    story = []


    story.append(
        Paragraph(
            "SmartLearn AI - Study Material",
            title_style
        )
    )

    story.append(Spacer(1, 20))


    if summary:

        story.append(
            Paragraph(
                "<b>AI Summary</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                escape(summary).replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 15))


    if notes:

        story.append(
            Paragraph(
                "<b>Intelligent Notes</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                escape(notes).replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 15))


    if revision:

        story.append(
            Paragraph(
                "<b>Exam Revision</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                escape(revision).replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )


    doc.build(story)


    return temp_pdf.name


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 SmartLearn AI")

    st.write(
        "AI-powered learning assistant"
    )

    st.markdown("---")

    st.markdown("### ✨ Features")

    st.write("🎥 Video Transcription")
    st.write("🔗 YouTube Transcript")
    st.write("📝 AI Summary")
    st.write("📚 Intelligent Notes")
    st.write("❓ MCQ Generator")
    st.write("📖 Exam Revision")
    st.write("📄 PDF Study Material")

    st.markdown("---")

    if st.button(
        "🔄 Start New Lecture",
        use_container_width=True
    ):

        st.session_state.transcript = None
        st.session_state.summary = None
        st.session_state.notes = None
        st.session_state.mcqs = None
        st.session_state.revision = None
        st.session_state.source_name = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🎓 SmartLearn AI")

st.markdown(
    '<div class="subtitle">'
    'Video Summarization & Intelligent Notes Generator'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT TABS
# ============================================================

tab1, tab2 = st.tabs([
    "🎥 Upload Video",
    "🔗 YouTube Link"
])


# ============================================================
# UPLOAD VIDEO
# ============================================================

with tab1:

    st.markdown(
        "### 📤 Upload your lecture video"
    )


    video_file = st.file_uploader(
        "Choose a video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm"
        ]
    )


    if video_file is not None:

        st.video(video_file)


        if st.button(
            "🎙️ Generate Transcript",
            use_container_width=True
        ):

            reset_generated_content()


            suffix = os.path.splitext(
                video_file.name
            )[1].lower()


            if not suffix:
                suffix = ".mp4"


            temp_path = None


            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_video:

                    temp_video.write(
                        video_file.getbuffer()
                    )

                    temp_path = temp_video.name


                with st.spinner(
                    "🎙️ Transcribing your video..."
                ):

                    model = load_whisper_model()

                    result = model.transcribe(
                        temp_path,
                        fp16=False
                    )


                transcript = clean_text(
                    result.get("text", "")
                )


                if transcript:

                    st.session_state.transcript = transcript

                    st.session_state.source_name = (
                        video_file.name
                    )

                    st.success(
                        "✅ Video transcribed successfully!"
                    )

                else:

                    st.error(
                        "❌ No speech could be detected "
                        "in this video."
                    )


            except Exception as e:

                st.error(
                    "❌ Video transcription failed."
                )

                st.info(
                    "Make sure FFmpeg is installed "
                    "in your Streamlit deployment."
                )

                st.code(str(e))


            finally:

                if temp_path and os.path.exists(
                    temp_path
                ):

                    try:
                        os.remove(temp_path)
                    except:
                        pass


# ============================================================
# YOUTUBE
# ============================================================

with tab2:

    st.markdown(
        "### 🔗 Enter a YouTube lecture link"
    )


    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )


    if st.button(
        "📥 Get YouTube Transcript",
        use_container_width=True
    ):

        if not youtube_url.strip():

            st.warning(
                "⚠️ Please enter a YouTube URL."
            )

        else:

            video_id = extract_youtube_video_id(
                youtube_url
            )


            if not video_id:

                st.error(
                    "❌ Invalid YouTube URL."
                )

            else:

                reset_generated_content()


                try:

                    with st.spinner(
                        "🔗 Fetching YouTube transcript..."
                    ):

                        transcript = get_youtube_transcript(
                            video_id
                        )


                    if transcript:

                        st.session_state.transcript = transcript

                        st.session_state.source_name = (
                            "YouTube Lecture"
                        )

                        st.success(
                            "✅ YouTube transcript loaded!"
                        )

                    else:

                        st.warning(
                            "⚠️ No transcript/captions "
                            "were found for this video."
                        )


                except Exception as e:

                    st.error(
                        "❌ Could not get the YouTube transcript."
                    )

                    st.info(
                        "This usually means the video has "
                        "no captions or YouTube is blocking "
                        "transcript access."
                    )

                    st.code(str(e))


# ============================================================
# TRANSCRIPT SECTION
# ============================================================

if st.session_state.get("transcript"):

    st.markdown("---")

    st.markdown(
        "## 📜 Transcript"
    )


    with st.expander(
        "View Full Transcript",
        expanded=False
    ):

        st.write(
            st.session_state.transcript
        )


    st.markdown("---")


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "## 📝 AI Summary"
    )


    if st.button(
        "✨ Generate Summary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 Generating AI summary..."
            ):

                st.session_state.summary = (
                    generate_summary(
                        st.session_state.transcript
                    )
                )


        except Exception as e:

            st.error(
                "❌ Summary generation failed."
            )

            st.code(str(e))


    if st.session_state.summary:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True
        )

        st.write(
            st.session_state.summary
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # NOTES
    # ========================================================

    st.markdown(
        "## 📚 Intelligent Notes"
    )


    if st.button(
        "📌 Generate Notes",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "📚 Creating intelligent notes..."
            ):

                st.session_state.notes = (
                    generate_notes(
                        st.session_state.transcript
                    )
                )


        except Exception as e:

            st.error(
                "❌ Notes generation failed."
            )

            st.code(str(e))


    if st.session_state.notes:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            st.session_state.notes
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # MCQs
    # ========================================================

    st.markdown(
        "## ❓ Questions & MCQs"
    )


    if st.button(
        "🧠 Generate MCQs",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🧠 Generating MCQs..."
            ):

                st.session_state.mcqs = (
                    generate_mcqs(
                        st.session_state.transcript
                    )
                )


        except Exception as e:

            st.error(
                "❌ MCQ generation failed."
            )

            st.code(str(e))

            st.session_state.mcqs = []


    if st.session_state.mcqs:

        for i, mcq in enumerate(
            st.session_state.mcqs,
            start=1
        ):

            st.markdown(
                f"### Q{i}. {mcq['question']}"
            )


            letters = ["A", "B", "C", "D"]


            for letter, option in zip(
                letters,
                mcq["options"]
            ):

                st.write(
                    f"**{letter}.** {option}"
                )


            with st.expander(
                f"✅ Show Answer - Q{i}"
            ):

                st.write(
                    f"**Correct Answer: "
                    f"{mcq['answer']}. "
                    f"{mcq['correct_text']}**"
                )


            st.markdown("---")


    elif st.session_state.mcqs == []:

        st.warning(
            "⚠️ Not enough meaningful information "
            "was found to create 4-option MCQs."
        )


    # ========================================================
    # EXAM REVISION
    # ========================================================

    st.markdown(
        "## 📖 Exam Revision Mode"
    )


    if st.button(
        "🚀 Generate Revision Material",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "📖 Preparing exam revision material..."
            ):

                st.session_state.revision = (
                    generate_revision(
                        st.session_state.transcript
                    )
                )


        except Exception as e:

            st.error(
                "❌ Revision generation failed."
            )

            st.code(str(e))


    if st.session_state.revision:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            st.session_state.revision
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # PDF
    # ========================================================

    st.markdown(
        "## 📄 Study Material"
    )


    if st.button(
        "📄 Generate PDF Study Material",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "📄 Creating PDF..."
            ):

                pdf_path = create_pdf(
                    st.session_state.transcript,
                    st.session_state.summary,
                    st.session_state.notes,
                    st.session_state.revision
                )


            with open(
                pdf_path,
                "rb"
            ) as pdf_file:

                st.download_button(
                    label="⬇️ Download Study Material",
                    data=pdf_file,
                    file_name="SmartLearn_AI_Study_Material.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "❌ PDF generation failed."
            )

            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🎓 SmartLearn AI | AI-Powered Learning Assistant'
    '</div>',
    unsafe_allow_html=True
)
