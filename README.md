💼 Resumate
📘 Overview
Resumate is a Streamlit-based web application designed to analyze resumes and generate tailored interview questions based on the job description and candidate's resume. The app uses Google's Generative AI (via langchain-google-genai) to perform deep analysis and provide insightful visualizations.

✨ Features
📄 Resume Analysis:

🔍 Analyzes resumes against a given job description.

📊 Breaks down technical skills, soft skills, experience relevance, and education alignment.

✅ Highlights strengths & ⚠️ areas for improvement.

🧠 Interview Question Generation:

💬 Generates tailored technical, behavioral, and project-specific interview questions.

📌 Each question comes with context to explain its relevance.

📈 Visualizations:

📉 Interactive charts (via Plotly) to show match percentages.

🏆 Comparison table to rank candidates based on relevance.

🔐 Secure API Key Management:

🗝️ Uses .env file to store & load your Google API Key securely.

🛠️ Prerequisites
Make sure you have the following installed:

🐍 Python 3.8+

📦 streamlit

🧠 langchain-google-genai

🔐 python-dotenv

📄 PyPDF2

🧮 pandas

📊 plotly

⚙️ Installation
🔁 Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
🧪 Create a virtual environment:

bash
Copy
Edit
python -m venv .venv
🧬 Activate the virtual environment:

Windows (CMD):

bash
Copy
Edit
.venv\Scripts\activate
Windows (PowerShell):

bash
Copy
Edit
.venv\Scripts\Activate.ps1
macOS/Linux:

bash
Copy
Edit
source .venv/bin/activate
📥 Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🧾 Set your Google API Key in .env:

env
Copy
Edit
GOOGLE_API_KEY=your_api_key_here
🚀 Usage
▶️ Run the app:

bash
Copy
Edit
streamlit run main_ai_interviewer.py
🌐 Open the link in your browser (http://localhost:8501).

✍️ Input Job Description:

Paste the JD in the provided textbox.

📤 Upload Resumes:

Upload one or more PDF resumes.

🧠 Analyze:

Click on "Analyze Resumes" to begin processing.

📊 View Results:

See candidate rankings, skill breakdowns, and auto-generated interview questions.

📁 Project Structure
bash
Copy
Edit
resume-analyzer/
├── .env                     # 🔐 API key file
├── main_ai_interviewer.py   # 🧠 Main Streamlit app
├── requirements.txt         # 📦 Dependencies
├── README.md                # 📘 Documentation
🧩 Dependencies
streamlit 🖥️ — Web app interface

langchain-google-genai 🧠 — Generative AI integration

python-dotenv 🔐 — Environment management

PyPDF2 📄 — Resume text extraction

pandas 📊 — Data analysis

plotly 📈 — Interactive charts

🤝 Contributing
We 💖 contributions!

🍴 Fork the repo

🌱 Create a feature/bugfix branch

✅ Commit and push your changes

🔁 Open a pull request

🙏 Acknowledgments
🌟 Streamlit

🔗 LangChain

🤖 Google Generative AI

📬 Contact
Got questions or suggestions? Reach out!

👤 Harshitha P Shetty

📧 Email: pharshithashetty@gmail.com

