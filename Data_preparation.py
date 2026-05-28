import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

def update_job_database(dataset_name, collection_obj):
    """Loads a Hugging Face dataset and upserts it into ChromaDB."""
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name)

    documents = []
    metadatas = []
    ids = []

    prefix = dataset_name.replace("/","_")
    for idx, row in enumerate(dataset["train"]):
        combined_text = (
            f"Job Description: {row['Job Description']} | Skills: {row['Skills']}"
        )
        documents.append(combined_text)
        metadatas.append({"job_title": row["Job Title"]})

        
        ids.append(f"{prefix}_{idx}")

    print(f"Syncing {len(documents)} records to database...")

    collection_obj.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print("Database sync complete!")


chroma_client = chromadb.PersistentClient(path="./my_chroma_db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="job_listings", embedding_function=sentence_transformer_ef
)

print(collection.count())
if collection.count() == 0:
    update_job_database("NxtGenIntern/job_titles_and_descriptions", collection)
else:
    print(f"Database already loaded ({collection.count()} records). Skipping sync.")


def predict_job_title(jd: str, skills: str, top_n: int = 1):
    """Queries ChromaDB using JD and Skills to return matching titles."""
    
    query_text = f"Job Description: {jd} | Skills: {skills}"

 
    results = collection.query(query_texts=[query_text], n_results=top_n)


    return results




