from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv
from typing import  List
import operator
from langchain_tavily import TavilySearch
from uuid import uuid4
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage  # 
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from Parser import extract_resume_with_pymupdf,Resume
from Data_preparation import predict_job_title
from State import ResumeAnalysisState, JobAnalysisState, ExperienceAnalysisDetails, ProjectAnalysisDetails, EducationalDetails, JobDescriptionAnalysis, JobSuggestion, SuggestionForImprovement

load_dotenv()

search_tool = TavilySearch(max_results=2)
llm = ChatOllama(
    model="llama3.2",
    temperature = 0

)
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

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
    job_title: str =Field(description="Job title given")
    employer: str =Field(description="company name")
    skill_required: List[str] =Field(description= "skill required for the job")
    tools_needed: List[str] =Field(description= "tools knowledge needed for the job ")
    experience_needed: int =Field(description="Minimum Experience Required for the job")
    matching_score: int =Field(description="Matching score of the resume and job")


class ExperienceListOutput(BaseModel):
    experiences: List[ExperienceOutput]

class ProjectListOutput(BaseModel):
    projects: List[ProjectOutput]

class JobListOutput(BaseModel):
    Jobs: List[JobOutput]



class ResumeAgent:
    def __init__(self, llm, system_prompt: str = ""):
        self.llm = llm
        self.system_prompt = system_prompt
    def experience_node(self, state: ResumeAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(ExperienceListOutput)
        experience_text = "\n\n".join(
            f"Role: {e.role}\nEmployer: {e.employer}\nDate: {e.date}\nDescription: {e.description}"
            for e in state['resume_text'].experience
        )
        messages = [
            SystemMessage(content=EXPERIENCE_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract experience details from:\n\n{experience_text}")
        ]
        result: ExperienceListOutput = structured_llm.invoke(messages)
        return {
            "experience": [e.model_dump() for e in result.experiences],
            "messages": [AIMessage(content=f"Extracted {len(result.experiences)} experience entries.")]
        }

    def project_node(self, state: ResumeAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(ProjectListOutput)
        project_text = "\n\n".join(
            f"Title: {p.project_title}\nSkills: {', '.join(p.skill_used)}\nDescription: {p.description}"
            for p in state['resume_text'].project
        )
        messages = [
            SystemMessage(content=PROJECT_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract project details from:\n\n{project_text}")
        ]
       
        result: ProjectListOutput = structured_llm.invoke(messages)

        return {
            "projects": [p.model_dump() for p in result.projects],
            "messages": [AIMessage(content=f"Extracted {len(result.projects)} projects.")]
        }

class JobAgent:
    def __init__(self, search_tool, llm, system_prompt: str = ""):
        self.search = search_tool
        self.llm = llm
        self.system_prompt = system_prompt

    def JobTitleSuggestionNode(self, state: JobAnalysisState) -> dict:
        
        # Build JD text from experience summaries
        experience_text = " ".join(
            e['summary_of_experience']
            for e in state['experience']
        )
        project_text = " ".join(
            p['summary_of_project']
            for p in state['projects']
        )
        # Build skills from experience
        skills = " ".join(
            skill
            for e in state['experience']
            for skill in e['skill_set_used'] + e['tools_used']
        ).join(
            skill            for p in state['projects']
            for skill in p['skill_set_used'] + p['tools_used']
        )

        # Build skills from projects
        
        # Query ChromaDB separately for experience and projects
        job_titles_exp  = predict_job_title(jd=experience_text, skills=skills, top_n=3)
        job_titles_proj = predict_job_title(jd=project_text, skills=skills, top_n=3)

        # Extract title strings from ChromaDB results
        titles_exp  = [meta["job_title"] for meta in job_titles_exp["metadatas"][0]]
        titles_proj = [meta["job_title"] for meta in job_titles_proj["metadatas"][0]]

        # Deduplicate while preserving order
        seen = set()
        all_titles = []
        for t in titles_exp + titles_proj:
            if t not in seen:
                seen.add(t)
                all_titles.append(t)

        return {
            "suggested_title": all_titles,
            "messages": [AIMessage(content=f"Predicted job titles: {', '.join(all_titles)}")]
        }

    def JobSearchNode(self, state: JobAnalysisState) -> dict:
    
        yoe = sum(e['years_of_experience'] for e in state['experience'])
        titles = state['suggested_title']

        job_descriptions = []

        # Search across multiple job platforms
        for title in titles:
            queries = [
                f"{title} job opening site:linkedin.com/jobs",
                f"{title} job opening site:indeed.com",
                f"{title} job opening site:glassdoor.com",
            ]

            for query in queries:
                results = self.search.invoke(query)
            for r in results:
                job_descriptions.append(
                    JobDescriptionAnalysis(
                        employer=r['employer'],
                        job_title=r['title'],
                        experience_needed=str(yoe),
                        skill_set_needed=[],
                        tools_needed=[],
                    )
                )

        return {
            "job_descriptions": job_descriptions,
            "messages": [AIMessage(content=f"Found {len(job_descriptions)} job postings for {title} across LinkedIn, Indeed and Glassdoor.")]
        }

    def JobRecommendationNode(self, state: JobAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(JobListOutput)

        candidate_skills = " ".join(
            skill
            for e in state['experience']
            for skill in e['skill_set_used'] + e['tools_used']
        )

        jd_text = "\n\n".join(
            f"Experience needed: {jd['experience_needed']}\n"
            f"Skills needed: {', '.join(jd['skill_set_needed'])}\n"
            f"Tools needed: {', '.join(jd['tools_needed'])}"
            for jd in state['job_descriptions']
        )

        messages = [
            SystemMessage(content="You are a job recommendation expert. Compare the candidate's skills with the job descriptions and recommend the best matching jobs with a matching score."),
            HumanMessage(content=(
                f"Candidate skills: {candidate_skills}\n\n"
                f"Job descriptions:\n{jd_text}\n\n"
                f"Recommended job title: {state['suggested_title']}\n\n"
                "Rank the jobs and provide matching scores."
            ))
        ]

        result: JobListOutput = structured_llm.invoke(messages)

        return {
            "job_suggestion": {
                "matching_score": result.Jobs[0].matching_score if result.Jobs else 0,
                "missing_skills": [],
                "missing_tools":  [],
            },
            "messages": [AIMessage(content=f"Recommended {len(result.Jobs)} jobs.")]
        }
    def ReferalNode(self,state: JobAnalysisState) -> dict:
        institutes =  " ".join(
            e['college_university']
            for e in state['education'])
        companies = " ".join(
            e["employers"]
            for e in state['experience']
        )
        targeted_companies = " ".join(
            jd['employer']            for jd in state['job_descriptions']
        )
        for target in targeted_companies.split():
            query = f"Find employer linkedin or email who working at {target} with education from {institutes} or had worked at {companies} or both on LinkedIn for a job at {target}."
            results = self.search.invoke(query)

            # Process results to find referral connections (not implemented here)
        return {
            "referral_connections": []
        }
        

class RecommendationAgent:
    def __init__(self,llm,search_tool,system_prompt = ""):
        self.llm = llm 
        self.search = search_tool
        self.system_prompt = system_prompt
    def BlogNode(self):
        pass
    def ResourceNode(self):
        pass
    def CourseNode(self):
        pass
    def TimelineNode(self):
        pass
    def GoalSettingNode(self):
        pass
    def FeedbackNode(self):
        pass
    def ImprovementSuggestionNode(self):
        pass
    def youTubeNode(self):
        pass
    
        
        
# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    agent = ResumeAgent(llm=llm)
    # agent_2 = JobAgent()


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
    initial_state: ResumeAnalysisState = {
        "resume_text":     extracted_data,
        "experience":      [],
        "projects":        [],
        "improvement_plan": None,
        "messages":        [],
    }

    resume_result = pipeline.invoke(initial_state)
    print("Experience:", resume_result["experience"])
    print("Projects:",   resume_result["projects"])