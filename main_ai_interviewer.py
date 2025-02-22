import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import PyPDF2
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the .env file
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

# Set the API key in the environment
os.environ["GOOGLE_API_KEY"] = google_api_key

# Streamlit app styling
st.markdown(
    """
    <style>
    /* Dark background for the entire app */
    body {
        background: #121212;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }

    /* Custom font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Sleek hover effect for buttons */
    .stButton>button {
        background: #1e1e1e;
        color: #ffffff;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for expanders */
    .stExpander {
        background: #1e1e1e;
        transition: all 0.3s ease;
    }

    .stExpander:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for file uploader */
    .stFileUploader {
        background: #1e1e1e;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for text areas */
    .stTextArea textarea {
        background: #1e1e1e;
        color: #ffffff;
        border: none;
        transition: all 0.3s ease;
    }

    .stTextArea:hover textarea {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for dataframes */
    .stDataFrame {
        background: #1e1e1e;
        transition: all 0.3s ease;
    }

    .stDataFrame:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for plots */
    .stPlotlyChart {
        background: #1e1e1e;
        transition: all 0.3s ease;
    }

    .stPlotlyChart:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Sleek hover effect for tabs */
    .stTabs [role="tab"] {
        background: #1e1e1e;
        color: #ffffff;
        border: none;
        transition: all 0.3s ease;
    }

    .stTabs [role="tab"]:hover {
        background: #2a2a2a;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: #2a2a2a;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Pydantic models for structured output
class ResumeAnalysis(BaseModel):
    technical_skills_match: Dict[str, float] = Field(description="Dictionary of technical skills and match percentage")
    soft_skills_match: Dict[str, float] = Field(description="Dictionary of soft skills and match percentage")
    experience_relevance: float = Field(description="Percentage of relevant experience")
    education_match: float = Field(description="Education requirement match percentage")
    overall_match: float = Field(description="Overall candidate match percentage")
    strengths: List[str] = Field(description="Key strengths of the candidate")
    gaps: List[str] = Field(description="Areas for improvement")
    detailed_feedback: str = Field(description="Detailed analysis and recommendations")

class InterviewQuestions(BaseModel):
    technical_questions: List[Dict[str, str]] = Field(description="List of technical questions with their context")
    behavioral_questions: List[Dict[str, str]] = Field(description="List of behavioral questions with their context")
    project_questions: List[Dict[str, str]] = Field(description="List of questions about past projects and experience")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to create the analysis prompt
def create_analysis_prompt():
    template = """
    You are an expert AI talent recruiter specialized in Generative AI and Machine Learning roles. 
    Analyze the following job description and resume in detail.
    
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text}
    
    Provide a comprehensive analysis focusing on:
    1. Technical Skills Match:
       - Core ML/AI skills
       - Programming languages
       - Frameworks and tools
       - Cloud and infrastructure
    
    2. Soft Skills Match:
       - Communication
       - Leadership
       - Problem-solving
       - Collaboration
    
    3. Experience Analysis:
       - Relevance to role
       - Project complexity
       - Impact and achievements
    
    4. Education Alignment
    
    5. Detailed Strengths and Gaps
    
    Provide your analysis in a structured format that matches the following schema:
    {format_instructions}
    
    Be specific and quantitative in your assessment. Include percentages for matches 
    and provide detailed reasoning for your evaluations.
    """
    
    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

# Function to analyze the resume
def analyze_resume(llm, prompt, parser, jd_text, resume_text):
    formatted_prompt = prompt.format(
        jd_text=jd_text,
        resume_text=resume_text
    )
    
    response = llm.predict(formatted_prompt)
    return parser.parse(response)

# Function to create visualizations
def create_visualization(analysis_result):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}],
               [{'type': 'bar', 'colspan': 2}, None]],
        subplot_titles=('Technical Skills', 'Soft Skills', 'Detailed Scores')
    )
    
    tech_skills_avg = sum(analysis_result.technical_skills_match.values()) / len(analysis_result.technical_skills_match)
    fig.add_trace(
        go.Pie(labels=['Match', 'Gap'],
               values=[tech_skills_avg, 100-tech_skills_avg],
               name="Technical Skills"),
        row=1, col=1
    )
    
    soft_skills_avg = sum(analysis_result.soft_skills_match.values()) / len(analysis_result.soft_skills_match)
    fig.add_trace(
        go.Pie(labels=['Match', 'Gap'],
               values=[soft_skills_avg, 100-soft_skills_avg],
               name="Soft Skills"),
        row=1, col=2
    )

    scores = {
        'Technical Skills': tech_skills_avg,
        'Soft Skills': soft_skills_avg,
        'Experience': analysis_result.experience_relevance,
        'Education': analysis_result.education_match,
        'Overall': analysis_result.overall_match
    }
    
    fig.add_trace(
        go.Bar(x=list(scores.keys()),
               y=list(scores.values()),
               name="Detailed Scores"),
        row=2, col=1
    )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

# Function to create the interview questions prompt
def create_interview_questions_prompt():
    template = """
    You are an expert technical interviewer for AI/ML positions. Based on the following job description and candidate's resume,
    generate relevant interview questions that will help assess the candidate's fit for the role.
    
    Job Description:
    {jd_text}
    
    Candidate's Resume:
    {resume_text}
    
    Key Areas to Focus:
    1. Technical Skills highlighted in their resume
    2. Past projects and their complexity
    3. Behavioral aspects based on role requirements
    
    For each question, provide:
    - The question itself
    - Context for why this question is relevant
    
    Generate a mix of:
    - Technical questions targeting their specific skills
    - Behavioral questions related to their experience
    - Project-specific questions based on their past work
    
    Provide your output in the following JSON format:
    {{
        "technical_questions": [
            {{
                "question": "Question 1",
                "context": "Context for Question 1"
            }},
            {{
                "question": "Question 2",
                "context": "Context for Question 2"
            }}
        ],
        "behavioral_questions": [
            {{
                "question": "Question 1",
                "context": "Context for Question 1"
            }},
            {{
                "question": "Question 2",
                "context": "Context for Question 2"
            }}
        ],
        "project_questions": [
            {{
                "question": "Question 1",
                "context": "Context for Question 1"
            }},
            {{
                "question": "Question 2",
                "context": "Context for Question 2"
            }}
        ]
    }}
    
    Be specific and ensure all fields are included.
    """
    
    parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

# Function to generate interview questions
def generate_interview_questions(llm, prompt, parser, jd_text, resume_text):
    formatted_prompt = prompt.format(
        jd_text=jd_text,
        resume_text=resume_text
    )
    
    response = llm.predict(formatted_prompt)
    
    try:
        return parser.parse(response)
    except Exception as e:
        st.error(f"Error parsing interview questions: {str(e)}")
        st.write("Raw API response:")
        st.write(response)  # Display the raw response for debugging
        return None

# Function to select top candidates
def select_top_candidates(results, num_candidates=2):
    ranked_candidates = sorted(
        results,
        key=lambda x: (
            x['analysis'].overall_match,
            sum(x['analysis'].technical_skills_match.values()) / len(x['analysis'].technical_skills_match)
        ),
        reverse=True
    )
    return ranked_candidates[:num_candidates]

# Function to display interview questions
def display_interview_questions(questions):
    if not questions:
        st.error("No interview questions were generated. Please check the input and try again.")
        return
    
    st.write("### Technical Questions")
    for i, q in enumerate(questions.technical_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")
    
    st.write("### Behavioral Questions")
    for i, q in enumerate(questions.behavioral_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")
    
    st.write("### Project Experience Questions")
    for i, q in enumerate(questions.project_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")

# Main function
def main():
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-pro",
    )
    
    prompt, parser = create_analysis_prompt()
    interview_prompt, interview_parser = create_interview_questions_prompt()

    st.header("Job Description💼")
    jd_text = st.text_area("Paste the job description here:", height=200)
    
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF format)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if st.button("Analyze Resumes") and jd_text and uploaded_files:
        all_results = []
        
        with st.spinner('Analyzing resumes... This may take a few minutes.'):
            for file in uploaded_files:
                st.write(f"Processing: {file.name}")
                
                resume_text = extract_text_from_pdf(file)
                if resume_text:
                    try:
                        analysis = analyze_resume(llm, prompt, parser, jd_text, resume_text)
                        all_results.append({
                            'filename': file.name,
                            'resume_text': resume_text,
                            'analysis': analysis
                        })
                    except Exception as e:
                        st.error(f"Error analyzing {file.name}: {str(e)}")
        
        if all_results:
            st.header("Analysis Results📑")
            
            comparison_data = []
            for result in all_results:
                analysis = result['analysis']
                comparison_data.append({
                    'Candidate': result['filename'],
                    'Overall Match': f"{analysis.overall_match:.1f}%",
                    'Technical Skills': f"{sum(analysis.technical_skills_match.values()) / len(analysis.technical_skills_match):.1f}%",
                    'Soft Skills': f"{sum(analysis.soft_skills_match.values()) / len(analysis.soft_skills_match):.1f}%",
                    'Experience': f"{analysis.experience_relevance:.1f}%",
                    'Education': f"{analysis.education_match:.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.subheader("Candidates Comparison:")
            st.dataframe(comparison_df)
            
            for result in all_results:
                with st.expander(f"Detailed Analysis: {result['filename']}"):
                    analysis = result['analysis']
                    
                    fig = create_visualization(analysis)
                    st.plotly_chart(fig, key=f"plotly_chart_{result['filename']}")
                    
                    st.subheader("Technical Skills Breakdown")
                    tech_skills_df = pd.DataFrame.from_dict(
                        analysis.technical_skills_match,
                        orient='index',
                        columns=['Match Percentage']
                    )
                    st.dataframe(tech_skills_df)
                    
                    st.subheader("Soft Skills Breakdown")
                    soft_skills_df = pd.DataFrame.from_dict(
                        analysis.soft_skills_match,
                        orient='index',
                        columns=['Match Percentage']
                    )
                    st.dataframe(soft_skills_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Key Strengths💪")
                        for strength in analysis.strengths:
                            st.write(f"✓ {strength}")
                    
                    with col2:
                        st.subheader("Areas for Improvement📉")
                        for gap in analysis.gaps:
                            st.write(f"○ {gap}")
                    
                    st.subheader("Detailed Feedback")
                    st.write(analysis.detailed_feedback)
            
            top_candidates = select_top_candidates(all_results)
            
            st.header("Top Candidates🥇")
            
            candidate_tabs = st.tabs([f"Candidate: {candidate['filename']}" for candidate in top_candidates])
            
            for tab, candidate in zip(candidate_tabs, top_candidates):
                with tab:
                    st.subheader("Candidate Analysis")
                    fig = create_visualization(candidate['analysis'])
                    st.plotly_chart(fig, key=f"plotly_chart_top_{candidate['filename']}")
                    
                    st.subheader("Recommended Interview Questions💡")
                    with st.spinner("Generating tailored interview questions..."):
                        interview_questions = generate_interview_questions(
                            llm,
                            interview_prompt,
                            interview_parser,
                            jd_text,
                            candidate['resume_text']
                        )
                        if interview_questions:
                            display_interview_questions(interview_questions)
                        else:
                            st.error("Failed to generate interview questions. Please check the input and try again.")

if __name__ == "__main__":
    main()