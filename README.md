# 📄 Resume Lens

An intelligent resume analysis tool that leverages **AI and LLMs** to match your resume against job postings, identify skill gaps, and provide personalized improvement suggestions.

---

## 🎯 What Resume Lens Does

Resume Lens automates the job search process by:

1. **📋 Parsing Your Resume** - Extracts structured information (skills, experience, education) from your PDF resume
2. **🔍 Finding Relevant Jobs** - Searches for job postings matching your profile using industry job APIs
3. **🤖 Intelligent Matching** - Uses AI to analyze how well your resume matches each job
4. **💡 Smart Recommendations** - Identifies skill gaps and suggests actionable improvements
5. **🎓 Learning Path Generation** - Recommends resources and timelines to upskill for target roles

---

## 🏗️ Project Structure

```
Resume_lens/
├── Agents.py                 # Main orchestration logic (LangGraph workflow)
├── Parser.py                 # PDF parsing & resume data extraction
├── API_call.py              # Job search API integration (JSearch)
├── Retrival_models.py       # Vector embeddings & similarity search
├── State.py                 # Data models for workflow state management
├── Data_preparation.py      # Dataset processing utilities
├── prompts.py               # LLM system prompts for various tasks
├── main.py                  # Entry point script
├── pyproject.toml           # Project dependencies & metadata
├── requirments.txt          # Additional dependencies
├── my_chroma_db/            # Local vector database (ChromaDB)
└── README.md               # This file
```

---

## 🔑 Key Components Explained

### **1. Parser.py** - Resume Extraction
- **What it does**: Reads your PDF resume and extracts structured data
- **Extracts**: Education, experience, skills, projects, certifications
- **Uses**: PyMuPDF4LLM for PDF parsing + LLMs for structured extraction
- **Output**: A `Resume` object with organized, machine-readable resume data

### **2. API_call.py** - Job Search
- **What it does**: Searches for job postings matching your role
- **API Used**: JSearch (RapidAPI) - real-time job posting data
- **Search Parameters**: Job title, location (India), posting date (last month)
- **Returns**: List of job postings with description, company, link, employment type

### **3. Agents.py** - Main Workflow (LangGraph)
The core orchestration using **LangGraph** with these nodes:

| Node | Purpose |
|------|---------|
| **job_title_suggestion** | Analyzes your resume to recommend suitable job titles |
| **job_search** | Searches for jobs matching the suggested titles |
| **job_recommendation** | Compares your skills against each job's requirements |
| **referral** (optional) | Finds potential referral connections |

### **4. Retrival_models.py** - Smart Matching
- **What it does**: Compares resume skills with job requirements
- **Uses**: 
  - Vector embeddings (sentence-transformers) for semantic similarity
  - BM25 algorithm for keyword matching
  - ChromaDB for efficient vector storage
- **Output**: Matching scores and identified skill gaps

### **5. State.py** - Data Models
Defines the data structures that flow through the workflow:
- `Resume` - Structured resume data
- `JobDescriptionAnalysis` - Parsed job requirements
- `JobAnalysisState` - Workflow state for job matching
- `SuggestionAnalysisState` - Workflow state for improvement suggestions

### **6. prompts.py** - AI Instructions
Contains system prompts that guide the LLM to:
- Extract resume information accurately
- Suggest relevant job titles
- Match skills to requirements
- Generate improvement recommendations

---

## 📦 Installation

### **Prerequisites**
- Python 3.10+ (project requires 3.13+)
- Ollama (for local LLM) OR Google API Key (for Gemini)
- API Keys for JSearch and Tavily (optional for web search)

### **Step 1: Clone & Setup**
```bash
cd Resume_lens
pip install -r requirements.txt
```

### **Step 2: Environment Variables**
Create a `.env` file in the root directory:

```env
# For Google's Gemini API (recommended - no local server needed)
GOOGLE_API_KEY=your_api_key_here

# For job search API
RAPIDAPI_KEY=your_rapidapi_key_here

# For web search (optional)
TAVILY_API_KEY=your_tavily_key_here
```

### **Step 3: LLM Setup**

#### **Option A: Use Google Gemini (Recommended)**
- Get API key from [Google AI Studio](https://aistudio.google.com)
- Uncomment the Gemini section in `Parser.py` and `Agents.py`
- Comment out the Ollama sections

#### **Option B: Use Ollama (Local)**
1. **Install Ollama**: https://ollama.ai
2. **Start Ollama server**:
   ```bash
   ollama serve
   ```
3. **In another terminal, pull the model**:
   ```bash
   ollama pull llama3.2
   ```

---

## 🚀 How to Run

### **Basic Usage**
```bash
python3 Agents.py
```

### **What Happens Step-by-Step**

1. **Resume Parsing**
   ```
   Input: Your PDF resume (/Users/rahulbiju/Downloads/Rahul_Biju_Resume_1.pdf)
   ↓
   Process: Extracts text and structures it using LLM
   ↓
   Output: Resume object with skills, experience, education
   ```

2. **Job Title Suggestion**
   ```
   Input: Resume skills and experience
   ↓
   Process: LLM analyzes and suggests 3-5 relevant job titles
   ↓
   Output: List of job titles (e.g., "Senior AI Engineer", "ML Researcher")
   ```

3. **Job Search**
   ```
   Input: Suggested job titles
   ↓
   Process: API searches for matching jobs
   ↓
   Output: Job postings with descriptions and apply links
   ```

4. **Job Matching & Analysis**
   ```
   Input: Resume + Job postings
   ↓
   Process: 
   - Extracts job requirements
   - Compares with your skills (semantic + keyword matching)
   - Calculates match percentage (0-100)
   ↓
   Output: Match score, missing skills, missing tools
   ```

5. **Improvement Suggestions** (Optional)
   ```
   Input: Skill gaps identified
   ↓
   Process: LLM generates learning plan
   ↓
   Output: Recommended skills, resources, estimated time to learn
   ```

---

## 📊 Output Example

When you run the script, you'll get results like:

```python
{
    "resume": {
        "skills": ["Python", "Machine Learning", "Data Analysis"],
        "experience": [{"role": "Data Scientist", ...}],
        ...
    },
    "suggested_title": [
        "Senior Data Scientist",
        "ML Engineer",
        "AI Specialist"
    ],
    "job_descriptions": [
        {
            "job_title": "Senior Data Scientist",
            "employer": "Tech Corp",
            "skill_set_needed": ["Python", "TensorFlow", "SQL"],
            ...
        }
    ],
    "job_suggestion": [
        {
            "job_title": "Senior Data Scientist",
            "matching_score": 85,
            "missing_skills": ["TensorFlow", "Spark"],
            "missing_tools": ["Databricks"]
        }
    ]
}
```

---

## 🔧 Configuration

### **Modify Resume Path**
Edit the PDF path in `Agents.py` (line ~305):
```python
pdf_file_path = "/path/to/your/resume.pdf"
```

### **Adjust Job Search**
In `API_call.py`:
- Change location from "India" to your country
- Modify `date_posted` from "month" to "week" or "anytime"
- Adjust `num_pages` for more/fewer results

### **LLM Settings**
In `Parser.py` and `Agents.py`:
- **Temperature**: Lower (0.0) = more consistent, Higher (1.0) = more creative
- **Model**: Change from `llama3.2` to other Ollama models (llama2, mistral, etc.)

---

## 📚 Dependencies Overview

| Package | Purpose |
|---------|---------|
| **langgraph** | Workflow orchestration (graph-based agents) |
| **langchain-*** | LLM framework & integrations |
| **chromadb** | Vector database for embeddings |
| **sentence-transformers** | Create semantic embeddings |
| **rank-bm25** | Keyword-based text matching |
| **PyMuPDF4LLM** | PDF parsing |
| **requests** | HTTP API calls |
| **pydantic** | Data validation & models |

---

## 🐛 Troubleshooting

### **Error: Connection refused**
- **Cause**: Ollama server not running
- **Fix**: Run `ollama serve` in a separate terminal

### **Error: TypedDict from typing**
- **Cause**: Pydantic v2 incompatibility with Python < 3.12
- **Fix**: Already fixed - TypedDict imported from `typing_extensions`

### **Error: API Key not found**
- **Cause**: `.env` file not created or API key not set
- **Fix**: Create `.env` file with your API keys

### **Error: PDF not found**
- **Cause**: Resume path doesn't exist
- **Fix**: Update the file path in `Agents.py`

---

## 🎓 How the AI Works (Behind the Scenes)

1. **LLM Models**: Uses either Ollama (local) or Google Gemini (cloud) for text understanding
2. **Structured Output**: Forces LLM to return data in specific formats (using Pydantic)
3. **Vector Embeddings**: Converts text to numerical representations for similarity matching
4. **Semantic Search**: Finds conceptually similar skills even if wording differs
5. **Agent Workflow**: Chains multiple LLM calls with decision points (LangGraph)

---

## 📝 Example Workflow

```
Your Resume (PDF)
    ↓
[Parser] Extract: Skills, Experience, Education
    ↓
[Agents] Generate job title suggestions
    ↓
[API] Search for matching jobs
    ↓
[Retrieval] Match your skills with job requirements
    ↓
[LLM] Generate improvement suggestions
    ↓
Results: Match scores, gaps, learning path
```

---

## 🔐 Security Notes

- **API Keys**: Never commit `.env` file to git
- **Rate Limiting**: JSearch API has request limits - cache results if needed
- **Local Processing**: Using Ollama keeps your resume data locally

---

## 📖 Next Steps

1. Set up your environment variables
2. Prepare your resume in PDF format
3. Run `python3 Agents.py`
4. Review the results and improvement suggestions
5. Track your progress in upskilling

---

## 🤝 Contributing

Feel free to improve this project! Possible enhancements:
- Support for multiple resume formats (DOCX, TXT)
- Integration with LinkedIn API
- Resume tailoring (auto-rewrite resume for specific jobs)
- Real-time job alerts
- Salary predictions

---

## 📄 License

[Add your license here]

---

## ❓ Questions?

For detailed questions about specific components, check the docstrings in each Python file or create an issue.

**Happy job hunting! 🎯**
