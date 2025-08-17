# lecture_summarizer
# 🎓 Smart Lecture Summarizer  

An **AI-powered web application** that transforms long YouTube lectures and tutorials into **concise summaries, smart notes, and personalized study plans**. Built with **Streamlit, LangChain, OpenRouter (LLaMA 3.3), and NLP techniques**, this tool helps students and professionals save time, learn efficiently, and prepare better for exams.  

---

## 🚀 Features  
- **YouTube Transcript Extraction** – Automatically fetches transcripts using `youtube-transcript-api`.  
- **AI-Powered Summarization** – Generates structured summaries and smart notes with **LLaMA 3.3 via OpenRouter**.  
- **Interactive Q&A Chatbot** – Ask questions directly from the summarized lecture for contextual learning.  
- **Personalized Study Plan Generator** – Creates day-by-day study schedules based on deadlines, hours, and study style.  
- **Export to PDF** – Download summaries and study plans for offline use.  
- **Streamlit Web App** – User-friendly interface for seamless interaction.  

---

## 🛠️ Tech Stack  
- **Frontend / App**: [Streamlit](https://streamlit.io/)  
- **AI & NLP**: [LangChain](https://www.langchain.com/), [OpenRouter](https://openrouter.ai/) (LLaMA 3.3)  
- **APIs**: [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)  
- **Environment Management**: `python-dotenv`  
- **PDF Generation**: `fpdf`  
- **Deployment**: Google Colab + Ngrok / Streamlit Cloud  

---

## ⚡ Installation & Setup  

```bash
# Clone this repository
git clone https://github.com/your-username/lecture-summarizer.git
cd lecture-summarizer

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run final_trial.py
