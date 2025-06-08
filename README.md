# 💼 Resumate

## 📘 Overview

**Resumate** is a **Streamlit-based web application** designed to analyze resumes and generate **tailored interview questions** based on the job description and candidate's resume. The app uses **Google's Generative AI** (via `langchain-google-genai`) to perform deep analysis and provide insightful visualizations.

---

## ✨ Features

1. **📄 Resume Analysis**:

   * 🔍 Analyzes resumes against a given job description.
   * 📊 Breaks down technical skills, soft skills, experience relevance, and education alignment.
   * ✅ Highlights strengths & ⚠️ areas for improvement.

2. **🧠 Interview Question Generation**:

   * 💬 Generates tailored **technical**, **behavioral**, and **project-specific** interview questions.
   * 📌 Each question comes with context to explain its relevance.

3. **📈 Visualizations**:

   * 📉 Interactive charts (via Plotly) to show **match percentages**.
   * 🏆 Comparison table to **rank candidates** based on relevance.

4. **🔐 Secure API Key Management**:

   * 🗝️ Uses `.env` file to store & load your **Google API Key** securely.

---

## 🛠️ Prerequisites

Make sure you have the following installed:

* 🐍 Python 3.8+
* 📦 `streamlit`
* 🧠 `langchain-google-genai`
* 🔐 `python-dotenv`
* 📄 `PyPDF2`
* 🧮 `pandas`
* 📊 `plotly`

---

## ⚙️ Installation

1. 🔁 Clone the repository:

   ```bash
   git clone https://github.com/pharshithashetty/Resumate.git
   cd resume-analyzer
   ```

2. 🧪 Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. 🧬 Activate the virtual environment:

   * **Windows (CMD)**:

     ```bash
     .venv\Scripts\activate
     ```
   * **Windows (PowerShell)**:

     ```bash
     .venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux**:

     ```bash
     source .venv/bin/activate
     ```

4. 📥 Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. 🧾 Set your Google API Key in `.env`:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

---

## 🚀 Usage

1. ▶️ Run the app:

   ```bash
   streamlit run main_ai_interviewer.py
   ```

2. 🌐 Open the link in your browser (`http://localhost:8501`).

3. ✍️ **Input Job Description**:

   * Paste the JD in the provided textbox.

4. 📤 **Upload Resumes**:

   * Upload one or more **PDF resumes**.

5. 🧠 **Analyze**:

   * Click on **"Analyze Resumes"** to begin processing.

6. 📊 **View Results**:

   * See candidate rankings, skill breakdowns, and **auto-generated interview questions**.

---

## 📁 Project Structure

```
resume-analyzer/
├── .env                     # 🔐 API key file
├── main_ai_interviewer.py   # 🧠 Main Streamlit app
├── requirements.txt         # 📦 Dependencies
├── README.md                # 📘 Documentation
```

---

## 🧩 Dependencies

* `streamlit` 🖥️ — Web app interface
* `langchain-google-genai` 🧠 — Generative AI integration
* `python-dotenv` 🔐 — Environment management
* `PyPDF2` 📄 — Resume text extraction
* `pandas` 📊 — Data analysis
* `plotly` 📈 — Interactive charts

---

## 🤝 Contributing

We 💖 contributions!

1. 🍴 Fork the repo
2. 🌱 Create a feature/bugfix branch
3. ✅ Commit and push your changes
4. 🔁 Open a pull request

---

## 🙏 Acknowledgments

* 🌟 [Streamlit](https://streamlit.io/)
* 🔗 [LangChain](https://www.langchain.com/)
* 🤖 [Google Generative AI](https://developers.generativeai.google/)

---

## 📬 Contact

Got questions or suggestions? Reach out!

* 👤 **Harshitha P Shetty**
* 📧 **Email**: [pharshithashetty@gmail.com](mailto:pharshithashetty@gmail.com)


