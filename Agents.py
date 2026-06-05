from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv
from typing import  List
import operator
from langchain_tavily import TavilySearch
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage  # 
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from Retrival_models import Retrieval
import pprint
from Parser import extract_resume_with_pymupdf
from State import  JobAnalysisState,  SuggestionAnalysisState, SuggestionForImprovement,JobDescriptionAnalysis
from prompts import (
    JOB_TITLE_SYSTEM_PROMPT,
    JOB_RECOMMENDATION_SYSTEM_PROMPT,
    IMPROVEMENT_SYSTEM_PROMPT,
    JD_EXTRACTION_PROMPT
)

from API_call import JSearchClient
load_dotenv()
job_client = JSearchClient()
search_tool = TavilySearch(max_results=2)

llm = ChatOllama(
    model="llama3.2",
    temperature = 0

    )
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     api_key=os.environ.get("GOOGLE_API_KEY") # Ensure this env variable is set

# )

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


class JDExtraction(BaseModel):
    experience_needed:    str            = Field(description="Years of experience required e.g. '3-5 years'. 'Not specified' if missing.")
    skill_set_needed:     List[str]      = Field(description="Skills explicitly required")
    tools_needed:         List[str]      = Field(description="Frameworks, platforms, and tools explicitly required")
    preferred_degree:     str            = Field(description="Preferred or required degree e.g. 'B.Tech/M.Tech in CS'. 'Not specified' if missing.")
    expertise_level:      str            = Field(description="Seniority level: 'Junior', 'Mid', 'Senior', or 'Lead'")
    other_preferences:    List[str]      = Field(description="Any other preferences mentioned e.g. 'startup experience', 'open source contributions', 'publication record'")


class JobOutput(BaseModel):
    job_title: str =Field(description="Job title given")
    employer: str =Field(description="company name")
    skill_required: List[str] =Field(description= "skill required for the job")
    tools_needed: List[str] =Field(description= "tools knowledge needed for the job ")
    experience_needed: int =Field(description="Minimum Experience Required for the job")
    link: str =Field(description="link to the job description")

class JobComparisonOutput(BaseModel):
    job_title: str =Field(description="Job title given")
    employer: str =Field(description="company name")
    matching_score: int =Field(description= "matching score between candidate and job requirements (0-100)")
    missing_skills: List[str] =Field(description= "skills that the candidate is missing for the job")
    missing_tools: List[str] =Field(description= "tools knowledge that the candidate is missing for the job ")

class JobListOutput(BaseModel):
    Jobs: List[JobOutput]
class JobListComparisonOutput(BaseModel):
    Jobs: List[JobComparisonOutput]
class ImprovementOutput(BaseModel):
    imporvement_plan: List[SuggestionForImprovement]

class JobAgent:
    def __init__(self, search_tool, llm, job_client, system_prompt: str = ""):
        self.search = search_tool
        self.llm = llm
        self.client = job_client
        self.system_prompt = system_prompt
    def _candidate_skills(self, state) -> str:
        resume= state['resume']

        experience_skills = " ".join(
            skill
            for e in resume.experience
            for skill in e.skill_used
        )
        project_skills = " ".join(
            skill         
            for p in resume.projects
            for skill in p.skill_used
        )

        return experience_skills + " " + project_skills
    def _candidate_experience(self, state) -> str:
        resume = state['resume']
        return " ".join(
            e.description
            for e in resume.experience
        )
    def _candidate_projects(self, state) -> str:
        resume = state['resume']
        return " ".join(
            p.description
            for p in resume.projects
        )

    def JobTitleSuggestionNode(self, state: JobAnalysisState) -> dict:
        
       
        experience_text = self._candidate_experience(state)

        project_text = self._candidate_projects(state)
     
        skills = self._candidate_skills(state)
        Retrieve = Retrieval()
        
        job_titles_exp  = Retrieve.RRF(experience_text, skills, top_n=3)
        
        job_titles_proj = Retrieve.RRF(project_text, skills, top_n=3)

        # Extract title strings from ChromaDB results
        titles_exp  = [hit["metadata"]["job_title"] for hit in job_titles_exp]
        titles_proj = [hit["metadata"]["job_title"] for hit in job_titles_proj]
        print(f"Experience-based titles: {titles_exp}")
        print(f"Project-based titles: {titles_proj}")

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
        titles = state['suggested_title']
        extract_llm = self.llm.with_structured_output(JDExtraction)
        job_descriptions_all = []

        for title in titles[:2]:                        
            job_details = self.client.search(title=title, num_pages=1)

            for job in job_details[:2]:               
                jd_text = job.get('job_description', '')
                if not jd_text:                         
                    continue

                try:
                    extracted: JDExtraction = extract_llm.invoke([
                        SystemMessage(content=JD_EXTRACTION_PROMPT),  
                        HumanMessage(content=f"Extract structured information from this job description:\n\n{jd_text[:3000]}")
                    ])
                except Exception as e:
                    print(f"JD extraction failed for {job.get('job_title')}: {e}")
                    continue                            # ✅ skip bad JD, don't crash

                # ✅ Merge API fields + LLM fields into final TypedDict schema
                job_descriptions_all.append(JobDescriptionAnalysis(
                    job_title=          job.get('job_title', ''),
                    employer=           job.get('employer_name', ''),
                    employment_type=    job.get('job_employment_type', ''),
                    apply_link=         job.get('job_apply_link', ''),
                    experience_needed=  extracted.experience_needed,
                    skill_set_needed=   extracted.skill_set_needed,
                    tools_needed=       extracted.tools_needed,
                    preferred_degree=   extracted.preferred_degree,
                    expertise_level=    extracted.expertise_level,
                    other_preferences=  extracted.other_preferences,
                ))

        return {
            "job_descriptions": job_descriptions_all,
            "messages": [AIMessage(content=f"Found {len(job_descriptions_all)} job postings for {', '.join(titles[:2])}.")]
        }
    def JobRecommendationNode(self, state: JobAnalysisState) -> dict:
        structured_llm = self.llm.with_structured_output(JobListComparisonOutput)

        candidate_skills = self._candidate_skills(state)
        jd = state['job_descriptions']

        job_desc_texts = []
        for jd in state['job_descriptions']:
            job_desc_texts.append(
                f"Title: {jd['job_title']}\n"
                f"Employer: {jd['employer']}\n"
                f"Experience Needed: {jd['experience_needed']} years\n"
                f"Skills Needed: {', '.join(jd['skill_set_needed'])}\n"
                f"Tools Needed: {', '.join(jd['tools_needed'])}\n"
            )
        all_jds_text = "\n\n".join(job_desc_texts)
    
        messages = [
        SystemMessage(content=JOB_RECOMMENDATION_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Candidate skills: {candidate_skills}\n\n"
            f"Recommended target job title: {state['suggested_title']}\n\n"
            f"Here are ALL the available Job Descriptions:\n"
            f"{all_jds_text}\n\n"
            
        ))
    ]

        result: JobListComparisonOutput = structured_llm.invoke(messages)

        return {
            "job_suggestion": result,
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
            jd['employer'] 
            for jd in state['job_descriptions']
        )
        for target in targeted_companies.split():
            query = f"Find employer linkedin or email who working at {target} with education from {institutes} or had worked at {companies} or both on LinkedIn for a job at {target}."
            results = self.search.invoke(query)
        print(f"Referral search results: {results}")

        return {
            "referral_connections": []
        }
        

# class RecommendationAgent:
#     def __init__(self,llm,search_tool,system_prompt = ""):
#         self.llm = llm 
#         self.search = search_tool
#         self.system_prompt = system_prompt
#     def BlogNode(self,state: SuggestionAnalysisState)-> dict:
#         pass    
        
#     def ResourceNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def CourseNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def TimelineNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def GoalSettingNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def FeedbackNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def ImprovementSuggestionNode(self,state: SuggestionAnalysisState) -> dict:
#         pass
#     def youTubeNode(self,state: SuggestionAnalysisState) -> dict:
#         pass

        
        
# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
   
    agent = JobAgent(search_tool=search_tool, llm=llm, job_client=job_client)
    

    graph = StateGraph(JobAnalysisState)

    graph.add_node("job_title_suggestion", agent.JobTitleSuggestionNode)
    graph.add_node("job_search", agent.JobSearchNode)
    graph.add_node("job_recommendation",    agent.JobRecommendationNode)

    # graph.add_node("referral", agent.ReferalNode)

    graph.set_entry_point("job_title_suggestion")
    graph.add_edge("job_title_suggestion", "job_search")
    graph.add_edge("job_search", "job_recommendation")
    # graph.add_edge("job_recommendation", "referral")
    return graph.compile()


if __name__ == "__main__":
    pipeline = build_graph()
    pdf_file_path = "/Users/rahulbiju/Downloads/Rahul_Biju_Resume_1.pdf"
    try:
        extracted_data = extract_resume_with_pymupdf(pdf_file_path)
    except Exception as e:
        print(f"Extraction halted: {e}")
    initial_state = JobAnalysisState(
        resume= extracted_data,
        education=extracted_data.education,
        projects=extracted_data.projects,
        experience=extracted_data.experience,
        job_descriptions=[],
        job_suggestion=None,
    )

    resume_result = pipeline.invoke(initial_state)
    print("Final Result:")
    pprint.pprint(resume_result)