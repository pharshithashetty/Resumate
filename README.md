# Resumate

## Overview
This is a Streamlit-based web application designed to analyze resumes and generate tailored interview questions based on the job description and candidate's resume. The app uses Google's Generative AI (via `langchain-google-genai`) to perform the analysis and generate questions. It also provides visualizations to compare candidates and highlight their strengths and areas for improvement.

---

## Features
1. **Resume Analysis**:
   - Analyzes resumes against a job description.
   - Provides a detailed breakdown of technical skills, soft skills, experience relevance, and education alignment.
   - Highlights key strengths and areas for improvement.

2. **Interview Question Generation**:
   - Generates tailored technical, behavioral, and project-specific interview questions.
   - Provides context for each question to help interviewers understand its relevance.

3. **Visualizations**:
   - Interactive charts (using Plotly) to visualize candidate match percentages.
   - Comparison table to rank candidates based on their overall match.

4. **Secure API Key Management**:
   - Uses `.env` file to securely store and load the Google API key.

---

## Prerequisites
Before running the app, ensure you have the following installed:
- Python 3.8 or higher
- Streamlit
- `langchain-google-genai`
- `python-dotenv`
- `PyPDF2`
- `pandas`
- `plotly`

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/resume-analyzer.git
   cd resume-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows (Command Prompt):
     ```bash
     .venv\Scripts\activate
     ```
   - On Windows (PowerShell):
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory and add your Google API key:
   ```plaintext
   GOOGLE_API_KEY=your_api_key_here
   ```

---

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run main_ai_interviewer.py
   ```

2. Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

3. **Input Job Description**:
   - Paste the job description in the provided text area.

4. **Upload Resumes**:
   - Upload one or more resumes in PDF format.

5. **Analyze Resumes**:
   - Click the "Analyze Resumes" button to start the analysis.

6. **View Results**:
   - The app will display a comparison table, detailed analysis for each candidate, and recommended interview questions for the top candidates.

---

## Project Structure
```
resume-analyzer/
├── .env                     # Environment variables (API key)
├── main_ai_interviewer.py   # Main Streamlit app script
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
```

---

## Dependencies
The project uses the following Python packages:
- `streamlit`: For building the web app.
- `langchain-google-genai`: For interacting with Google's Generative AI models.
- `python-dotenv`: For loading environment variables from `.env`.
- `PyPDF2`: For extracting text from PDF files.
- `pandas`: For data manipulation and analysis.
- `plotly`: For creating interactive visualizations.

---

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

---

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the web app framework.
- [LangChain](https://www.langchain.com/) for the AI integration.
- [Google Generative AI](https://developers.generativeai.google/) for the AI models.

---

## Contact
For questions or feedback, please reach out to:
- Harshitha P Shetty  
- **Email**: pharshithashetty@gmail.com  
---
