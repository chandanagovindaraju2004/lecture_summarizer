import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI   # if you installed langchain-openai
# from langchain.chat_models import ChatOpenAI   # if only using langchain
import os
import re
from fpdf import FPDF

# Load environment variables
load_dotenv()

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="🎓 Smart Lecture Summarizer", page_icon="📚")
st.title("🎓 Smart Lecture Summarizer")

# ---------------------- Initialize session state ----------------------
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_study_plan" not in st.session_state:
    st.session_state.show_study_plan = False
if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False
if "last_video_url" not in st.session_state:
    st.session_state.last_video_url = ""

# ---------------------- Functions ----------------------
def get_youtube_transcript(video_url):
    long_form = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    short_form = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", video_url)

    if long_form:
        video_id = long_form.group(1)
    elif short_form:
        video_id = short_form.group(1)
    else:
        raise ValueError("Invalid YouTube URL format.")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])


def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)


def generate_pdf(text, filename="summary_notes.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename


def build_chat_prompt(summary, chat_history, new_question):
    prompt = f"You are a helpful assistant answering questions based on the following lecture summary:\n\n{summary}\n\n"
    if chat_history:
        prompt += "Here is the conversation so far:\n"
        for q, a in chat_history:
            prompt += f"Q: {q}\nA: {a}\n"
    prompt += f"\nQ: {new_question}\n\nAnswer:"
    return prompt

# ---------------------- LLM Setup ----------------------
os.environ["OPENAI_API_KEY"] = "ur api"

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    openai_api_base="https://openrouter.ai/api/v1"
)

# Prompt Templates
lecture_prompt = """
You are a helpful study assistant.

Given the transcript of a lecture or tutorial:

{transcript_text}

Please:

1. Summarize it in bullet points.
2. Suggest smart, self-explanatory note headings and bullet points.

Format exactly as:
📋 Summary:
- Bullet 1
- Bullet 2

📝 Smart Notes:
Heading 1:
- Point A
- Point B

Heading 2:
- Point C
"""

summary_prompt_template = PromptTemplate(
    input_variables=["transcript_text"],
    template=lecture_prompt
)

# ---------------------- Streamlit UI ----------------------

# Step 1: YouTube URL input
video_url = st.text_input("📺 Enter YouTube lecture/tutorial URL:", value=st.session_state.last_video_url)

if video_url and video_url != st.session_state.last_video_url:
    # New URL entered, fetch transcript
    st.session_state.last_video_url = video_url
    st.session_state.summary_output = ""
    st.session_state.chat_history = []
    st.session_state.show_study_plan = False
    st.session_state.question_submitted = False

    with st.spinner("🔍 Fetching transcript..."):
        try:
            st.session_state.transcript_text = get_youtube_transcript(video_url)
            st.success("✅ Transcript fetched successfully!")
        except Exception as e:
            st.error(f"❌ Error fetching transcript: {e}")
            st.stop()

if st.session_state.transcript_text:
    st.text_area("🧾 Transcript Preview:", st.session_state.transcript_text[:2000] + "...", height=200)

    # Step 2: Summarize button
    if st.button("🧠 Summarize Transcript"):
        with st.spinner("⏳ Summarizing with Llama 3.3 70B..."):
            try:
                chain = summary_prompt_template | llm
                response = chain.invoke({"transcript_text": st.session_state.transcript_text})
                output = response.content if hasattr(response, "content") else str(response)
                st.session_state.summary_output = remove_emojis(output)
                st.session_state.chat_history = []  # reset chat when new summary
                st.session_state.show_study_plan = False
                st.session_state.question_submitted = False

                st.success("✅ Summary and Notes Generated!")
            except Exception as e:
                st.error(f"❌ LLM Error: {e}")

# Step 3: Show summary and chat input if summary exists
if st.session_state.summary_output:
    st.markdown("### 📋 Summary & 📝 Smart Notes")
    st.markdown(st.session_state.summary_output)

    pdf_file = generate_pdf(st.session_state.summary_output)
    with open(pdf_file, "rb") as file:
        st.download_button(
            label="💾 Download Summary as PDF",
            data=file,
            file_name="Smart_Lecture_Summary.pdf",
            mime="application/pdf"
        )

    st.divider()
    st.subheader("💬 Ask questions about the lecture summary")

    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input("Ask a question related to the lecture summary:", key="chat_input")
        submit_button = st.form_submit_button("❓ Get Answer")

    if submit_button:
        if question.strip() == "":
            st.warning("Please enter a question!")
        else:
            with st.spinner("🤖 Thinking..."):
                try:
                    chat_prompt = build_chat_prompt(st.session_state.summary_output, st.session_state.chat_history, question)
                    answer = llm.predict(chat_prompt)
                    st.session_state.chat_history.append((question, answer))
                    st.session_state.question_submitted = True
                except Exception as e:
                    st.error(f"❌ Error getting answer: {e}")

    # Display chat history (latest first)
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    # Step 4: Show "Create Study Plan" button and section if clicked
    if not st.session_state.show_study_plan:
        if st.button("🗓 Create Study Plan"):
            st.session_state.show_study_plan = True

    if st.session_state.show_study_plan:
        st.divider()
        st.subheader("📅 Personalized Study Plan Generator")

        exam_date = st.date_input("📅 Select your exam or deadline date:")
        daily_hours = st.slider("⏰ How many hours can you study per day?", 1, 12, 3)
        plan_type = st.selectbox("📚 What kind of plan do you prefer?", ["Balanced", "Revision-Heavy", "Practice-Heavy"])

        if st.button("📘 Generate Study Plan", key="generate_study_plan"):
            with st.spinner("⚙ Generating your study plan..."):
                try:
                    study_plan_prompt = """
You are an intelligent academic assistant.

Based on the following lecture summary, create a personalized, day-by-day study plan for the user.
Their exam or goal deadline is {exam_date}.
They can study {daily_hours} hours per day.
They prefer a {plan_type} study style.

Lecture Summary:
{summary}

Format the output clearly with daily tasks.
"""

                    study_plan_template = PromptTemplate(
                        input_variables=["summary", "exam_date", "daily_hours", "plan_type"],
                        template=study_plan_prompt
                    )

                    study_plan_chain = study_plan_template | llm
                    study_plan_output = study_plan_chain.invoke({
                        "summary": st.session_state.summary_output,
                        "exam_date": exam_date.strftime("%B %d, %Y"),
                        "daily_hours": daily_hours,
                        "plan_type": plan_type
                    })

                    plan_text = study_plan_output.content if hasattr(study_plan_output, "content") else str(study_plan_output)
                    clean_plan_text = remove_emojis(plan_text)

                    st.success("✅ Personalized Study Plan Generated!")
                    st.markdown(clean_plan_text)

                    plan_pdf_file = generate_pdf(clean_plan_text, filename="Personalized_Study_Plan.pdf")
                    with open(plan_pdf_file, "rb") as plan_file:
                        st.download_button(
                            label="📥 Download Study Plan as PDF",
                            data=plan_file,
                            file_name="Study_Plan.pdf",
                            mime="application/pdf"
                        )

                except Exception as e:
                    st.error(f"❌ Study Plan Error: {e}")
