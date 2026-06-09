from typing import List, Optional,Dict,TypedDict
from pydantic import BaseModel, Field, ConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from dotenv import load_dotenv
import os
from cache import load_from_cache, save_to_cache
from prompts import (
    RESUME_SYSTEM_PROMPT
)

load_dotenv()


class Education(BaseModel):
    date: str = Field(
        description="The date range or graduation year (e.g., '2019-2023', 'May 2021'). Fall back to 'Not Specified' if missing."
    )
    college_university: str = Field(
        description="Name of the university, college, institute, or school."
    )
    stream: str = Field(
        description="Degree title, major, or specialization (e.g., 'B.Tech in IT', 'M.S. in Computer Science')."
    )
    Specialization: Optional[str] = Field(
        description="Any specific focus area or specialization within the degree. Use null if not provided."
    )
    CGPA: Optional[float] = Field(
        description="Cumulative GPA or percentage if explicitly mentioned. Use null if not provided."
    )
    Thesis_title: Optional[str] = Field(
        description="Title of the thesis or dissertation if mentioned. Use null if not provided."
    )


class Experience(BaseModel):
    date: str = Field(
        description="Employment period (e.g., 'Jan 2021 – Mar 2023'). Fall back to 'Not Specified' if missing."
    )
    years_of_experience: float = Field(
        description="Total years of experience in this role. If only dates are provided, calculate the duration in years.If only months give it as fraction of year. If neither is provided, set to 0."
    )
    role: str = Field(
        description="Job title or position held (e.g., 'Software Engineer', 'Data Analyst')."
    )
    employer: str = Field(
        description="Company or organization name."
    )
    skill_used: List[str] = Field(
        description="List of skills and tools explicitly mentioned in this experience entry."
    )
    description: str = Field(
        description="Summary of responsibilities, achievements, or technologies used in this role."
    )
class Projects(BaseModel):
    project_title: str = Field(
        description="The title of the project"
    )
    skill_used: List[str] = Field(
        description="List of skills and tools used in the project"
    )
    description: str = Field(
        description="Brief summary of what the project does and its outcome"
    )

class Publication(BaseModel):
    date: str = Field(
        description="Publication or submission year (e.g., '2024')."
    )
    title: str = Field(
        description="Full official title of the research paper or manuscript."
    )
    conference_journal: str = Field(
        description="Name of the journal or conference where published. Do NOT put the paper title here."
    )


class Resume(BaseModel):
    # Allow both 'name' and 'Name' to populate this field

    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(alias="Name", description="Full name of the candidate.")
    professional_summary: Optional[str] = Field(description="professional summary of the candidate")
    education: List[Education] = Field(default=[], description="All academic background entries.")
    experience: List[Experience] = Field(default=[], description="All professional work history entries.")
    projects: List[Projects] = Field(default=[], description="All projects entries")
    skills: List[str] = Field(default=[], description="Technical or soft skills.")
    certifications: List[str] = Field(default=[], description="Certifications or credentials.")
    publication: Optional[List[Publication]] = Field(default=[], description="Research publications if any.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.environ.get("GOOGLE_API_KEY") # Ensure this env variable is set

)
# llm = ChatOllama(
#     model="llama3.2",
#     temperature = 0
# )
structured_llm = llm.with_structured_output(Resume)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        RESUME_SYSTEM_PROMPT
    ),
    ("user", "Extract structured data from the following resume:\n\n{resume_text}"),
])

extraction_chain = prompt | structured_llm

MAX_CHARS = 12000  # Stay within llama3.2's context window


def extract_resume_with_pymupdf(pdf_path: str) -> Resume:
    # Try to load from cache first
    cached_data = load_from_cache(pdf_path)
    if cached_data is not None:
        return cached_data
    

    loader = PyMuPDF4LLMLoader(pdf_path)
    pages = loader.load()
    markdown_text = "\n".join(page.page_content for page in pages)

    if len(markdown_text) > MAX_CHARS:
        print(f"[Warning] Resume text trimmed from {len(markdown_text)} to {MAX_CHARS} chars.")
        markdown_text = markdown_text[:MAX_CHARS]

    result = extraction_chain.invoke({"resume_text": markdown_text})
    save_to_cache(pdf_path, result)


    if result is None:
        raise ValueError("Structured extraction returned None — the LLM may have failed to parse the schema.")

    return result

