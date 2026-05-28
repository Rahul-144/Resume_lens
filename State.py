
from typing import TypedDict,List,Optional


class ExperienceAnalysisDetails(TypedDict):
    years_of_experience: int
    skill_set_used: List[str]
    tools_used: List[str]
    summary_of_experience: str

class ProjectAnalysisDetails(TypedDict):
    skill_set_used: List[str]
    tools_used: List[str]
    summary_of_project: str

class EducationalDetails(TypedDict):
    college_university: str
    degree: str
    stream: str
    CGPA: Optional[float]
    

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
    resume_text: Resume
    experience: List[ExperienceAnalysisDetails]
    projects: List[ProjectAnalysisDetails]
    education:List[EducationalDetails]
    improvement_plan: Optional[SuggestionForImprovement]

    messages: Annotated[List, operator.add]                 

class JobAnalysisState(TypedDict):
    experience: List[ExperienceAnalysisDetails]
    project: List[ProjectAnalysisDetails]
    education: List[EducationalDetails]
    job_descriptions: List[JobDescriptionAnalysis]
    job_suggestion: Optional[JobSuggestion]