from typing import List, Optional, Annotated, TypedDict
import operator
from langchain_core.messages import AnyMessage
from Parser import Resume  
# ─────────────────────────────────────────────────────────────────────────────
# Sub-schemas  (only what isn't already in Resume)
# ─────────────────────────────────────────────────────────────────────────────

class JobDescriptionAnalysis(TypedDict):
   
    job_title:            str
    employer:             str
    employment_type:      str
    apply_link:           str
    experience_needed:    str
    skill_set_needed:     List[str]
    tools_needed:         List[str]
    preferred_degree:     str
    expertise_level:      str
    other_preferences:    List[str]

class JobSuggestion(TypedDict):
    matching_score: int
    missing_skills: List[str]
    missing_tools:  List[str]

class SuggestionForImprovement(TypedDict):
    skill_gaps:             List[str]
    recommended_resources:  List[str]
    estimated_weeks:        int


# ─────────────────────────────────────────────────────────────────────────────
# Graph states
# ─────────────────────────────────────────────────────────────────────────────


class JobAnalysisState(TypedDict):
    resume:               Resume 
    suggested_title:      List[str]         
    job_descriptions:     List[JobDescriptionAnalysis]
    job_suggestion:       Optional[JobSuggestion]
    education:            List[dict]  # Extracted from resume for referral searches
    experience:           List[dict]  # Extracted from resume for referral searches
    projects:             List[dict]  # Extracted from resume for referral searches
    referral_connections: List[dict]


class SuggestionAnalysisState(TypedDict):
    resume:               Resume          
    suggested_title:      List[str]
    job_descriptions:     List[JobDescriptionAnalysis]
    job_suggestion:       Optional[JobSuggestion]
    improvement_plan:     Optional[SuggestionForImprovement]
    resources:            List[str]
    timeline:             List[str]
    goals:                List[str]
    messages:             Annotated[List[AnyMessage], operator.add]