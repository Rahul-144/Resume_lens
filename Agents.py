from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, TypedDict, Annotated
import operator
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # 
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from Parser import extract_resume_with_pymupdf
from Data_preparation import predict_job_title


load_dotenv()

search_tool = TavilySearch(max_results=2)
llm = ChatOllama(
    model="llama3.2",
    temperature = 0

)


class ExperienceAnalysisDetails(TypedDict):
    years_of_experience: int
    skill_set_used: List[str]
    tools_used: List[str]
    summary_of_experience: str

class ProjectAnalysisDetails(TypedDict):
    skill_set_used: List[str]
    tools_used: List[str]
    summary_of_project: str

class JobDescriptionAnalysis(TypedDict):
    experience_needed: str
    skill_set_needed: List[str]
    tools_needed: List[str]

class JobSuggestion(TypedDict):
    matching_score: int
    missing_skills: List[str]
    missing_tools: List[str]

class SuggestionForImprovement(TypedDict):
    skill_gaps: List[str]
    recommended_resources: List[str]
    estimated_weeks: int

class ResumeAnalysisState(TypedDict):
    resume_text: str                                        
    experience: List[ExperienceAnalysisDetails]
    projects: List[ProjectAnalysisDetails]
    job_descriptions: List[JobDescriptionAnalysis]
    job_suggestion: Optional[JobSuggestion]
    improvement_plan: Optional[SuggestionForImprovement]
    messages: Annotated[List, operator.add]                 



class ExperienceOutput(BaseModel):
    years_of_experience: int    = Field(description="Total years of professional experience, please don;t hallucinate")
    skill_set_used: List[str]   = Field(description="Programming languages and ML skills used")
    tools_used: List[str]       = Field(description="Frameworks, tools, platforms used")
    summary_of_experience: str  = Field(description="Two sentence summary of experience")

class ProjectOutput(BaseModel):
    skill_set_used: List[str]   = Field(description="Skills demonstrated in the project")
    tools_used: List[str]       = Field(description="Tools and frameworks used")
    summary_of_project: str     = Field(description="One sentence project summary")

class JobOutput(BaseModel):
    job_title: str
    skill_required: List[str]
    tools_needed: List[str]
    matching_score: int


class ExperienceListOutput(BaseModel):
    experiences: List[ExperienceOutput]

class ProjectListOutput(BaseModel):
    projects: List[ProjectOutput]

class JobListOutput(BaseModel):
    Jobs: List[JobOutput]



# ── Agent class ──────────────────────────────────────────────────────────────
# ✅ Fix 2: PascalCase, no (), llm injected, prompt defined

EXPERIENCE_SYSTEM_PROMPT = """
You are an expert resume analyst. Extract structured experience details 
from the resume text provided. Be specific about skills and tools.
"""
PROJECT_SYSTEM_PROMPT = """
You are an expert resume analyst. Extract structured project details 
from the resume text provided. Focus on technical skills demonstrated.
"""

class ResumeAgent:
    def __init__(self, llm, system_prompt: str = ""):
        self.llm = llm
        self.system_prompt = system_prompt
        self.search_tool = search_tool

   
    def experience_node(self, state: ResumeAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(ExperienceListOutput)

        messages = [
            SystemMessage(content=EXPERIENCE_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract experience details from:\n\n{state['resume_text'].experience}")
        ]

        result: ExperienceListOutput = structured_llm.invoke(messages)

        
        return {
            "experience": [e.model_dump() for e in result.experiences],
            "messages": [AIMessage(content=f"Extracted {len(result.experiences)} experience entries.")]
        }

    def project_node(self, state: ResumeAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(ProjectListOutput)

        messages = [
            SystemMessage(content=PROJECT_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract project details from:\n\n{state['resume_text']}")
        ]

        result: ProjectListOutput = structured_llm.invoke(messages)

        return {
            "projects": [p.model_dump() for p in result.projects],
            "messages": [AIMessage(content=f"Extracted {len(result.projects)} projects.")]
        }

class JobAgent:
    def __init__(self, search_tool,llm, system_prompt:str =""):
        self.job_title = job_title
        self.search = search_tool
        self.llm = llm 
        self.system_prompt = system_prompt

        
    def JobTitleSuggestionNode():
        job_title = predict_job_title()

    def JobSearchNode():
        pass 
    def JobRecommendationNode():
        pass


        
# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    agent = ResumeAgent(llm=llm)

    graph = StateGraph(ResumeAnalysisState)

    graph.add_node("experience_node", agent.experience_node)
    graph.add_node("project_node",    agent.project_node)

    graph.set_entry_point("experience_node")
    graph.add_edge("experience_node", "project_node")
    graph.add_edge("project_node", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_graph()
    pdf_file_path = "/Users/rahulbiju/Downloads/Rahul_Biju_Resume_1.pdf"
    try:
        extracted_data = extract_resume_with_pymupdf(pdf_file_path)
    except Exception as e:
        print(f"Extraction halted: {e}")
    print(extracted_data)
    initial_state: ResumeAnalysisState = {
        "resume_text":     extracted_data,
        "experience":      [],
        "projects":        [],
        "job_descriptions":[],
        "job_suggestion":  None,
        "improvement_plan":None,
        "messages":        [],
    }

    result = pipeline.invoke(initial_state)
    print("Experience:", result["experience"])
    print("Projects:",   result["projects"])