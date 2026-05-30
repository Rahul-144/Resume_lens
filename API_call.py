import requests
import os
from typing import List, Optional

class JSearchClient:
    BASE_URL = "https://jsearch.p.rapidapi.com/search"
    HEADERS  = {
        "X-RapidAPI-Key":  os.environ.get("RAPIDAPI_KEY"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    def search(self, title: str, num_pages: int = 1) -> list:
        params = {
            "query": f"{title} jobs in India",  
            "page": "1",
            "num_pages": str(num_pages),
            "date_posted": "month",
        }
        try:
            response = requests.get(
                self.BASE_URL,
                headers=self.HEADERS,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            
            # This API returns a list of dictionaries inside the "data" field
            raw_jobs_list = response.json().get("data", [])
            cleaned_jobs = []

            for job in raw_jobs_list:
                # Safely extract and format only the keys you explicitly requested
                extracted_data = {
                    "job_title": job.get("job_title", ""),
                    "employer_name": job.get("employer_name", ""),
                    "job_employment_types": job.get("job_employment_types", []),
                    "job_apply_link": job.get("job_apply_link", ""),
                    "job_description": job.get("job_description", "")
                }
                cleaned_jobs.append(extracted_data)
                
            return cleaned_jobs
            
        except Exception as e:
            print(f"JSearch error: {e}")
            return []