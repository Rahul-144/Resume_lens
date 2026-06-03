from datasets import load_dataset

def process_GenAI_dataset(dataset_name: str = "cmagganas/GenAI-job-postings-Dataset"):

    # 1. Load the dataset
    dataset = load_dataset(dataset_name)
    
    # 2. Parse raw dataset into structured job postings
    job_posting = []
    for data in dataset["train"]:
        job_postings_sample = {
            "job_title": data["job_title"],
        "position_level": data["position_level"],
        "domain": data["use_case"],
    }
    job_postings_sample["job_description"] = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in data['job_posting'].split("\n\n")
        if ":" in line  
    }
    job_posting.append(job_postings_sample) 

    # 3. Define the targeting keys (Converted to a set for speed)
    needed_keys = {'Requirements', 'Preferred Qualifications', 'Job Requirements', 'Job summary', 'Preferred qualifications', 'Primary Responsibilities',
                    'Overview', 'Key Qualifications', 'Key Responsibilities', 'Job Duties', 'Company Description', 'Job Overview',
                    'As an AI Consultant with a focus on natural language processing, you will be responsible for helping clients improve their business operations through the use of AI technologies. Your primary responsibilities will include', 
                    'The ideal candidate will have', 'Preferred', 'Job Description for a Mid-level Deep Learning Engineer specializing in Reinforcement Learning', 'Education', 
                    'Your primary responsibilities include', 'Based on existing job postings and industry standards, here is a sample job description for a senior Robotics Engineer role', 'Job Purpose', 'Company Overview', 'Job Summary', 'About the Role', 
                    'Job Responsibilities',  'Key Requirements', 'Required Qualifications', 'Your core responsibilities will include the following', 'Qualifications', 'Job Description',  'Position Type', 'Roles and Responsibilities',
                    "Based on our company's needs and requirements, here is an example job description for a Robotics Engineer role for a mid-level position using recommender system", 'Key responsibilities may include', 'Summary', 'Qualifications and Requirements', 'Responsibilities'}
    needed_keys_lower = {k.lower() for k in needed_keys}


    # 4. Prepare separate aligned arrays for ChromaDB
    documents = []
    metadatas = []
    ids = []
    documents_BM25 = []
    
    prefix = dataset_name.replace("/", "_")    
    # 5. Process and flatten into individual chunks
    for posting_idx, posting in enumerate(job_posting, start=1):
        job_desc = posting["job_description"]
        filtered_desc = {
                key: value
                for key, value in job_desc.items()
                if key.lower().strip() in needed_keys_lower   # ✅ case insensitive
            }
        if filtered_desc:
            for chunk_idx, (key, value) in enumerate(filtered_desc.items()):
                documents_BM25.append({"text": f"{key}_{value}", "metadata": {
                    "job_title": posting["job_title"],
                    "position_level": posting["position_level"],
                    "domain": posting["domain"],
                    "section_type": key  # Keeps track of what part of the job description this is
                }})
                
                # Append directly to individual flat lists required by ChromaDB
                documents.append(f"{key}: {value}")
                metadatas.append({
                    "job_title": posting["job_title"],
                    "position_level": posting["position_level"],
                    "domain": posting["domain"],
                    "section_type": key  # Keeps track of what part of the job description this is
                })
                ids.append(f"{prefix}_{posting_idx}_{chunk_idx}")
    return documents, metadatas, ids, documents_BM25

def process_Nexgen_dataset(dataset_name: str = "NxtGenIntern/job_titles_and_descriptions"):
    dataset = load_dataset(dataset_name)
    documents = []
    metadatas = []
    ids = []
    documents_BM25 = []
    
    prefix = dataset_name.replace("/", "_")    
    for idx, row in enumerate(dataset["train"]):
        combined_text = (
            f"Job Description: {row['Job Description']} | Skills: {row['Skills']}"
        )
        documents.append(combined_text)
        metadatas.append({
            "job_title": row["Job Title"],
        })
        ids.append(f"{prefix}_{idx}")
        documents_BM25.append({'text': combined_text, 'metadata': {
            "job_title": row["Job Title"],
            
        }})
    return documents, metadatas, ids, documents_BM25