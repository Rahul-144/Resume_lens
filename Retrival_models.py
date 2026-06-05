import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from Data_preparation import process_GenAI_dataset, process_Nexgen_dataset
 


class Retrieval:
    def __init__(self):
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                                    model_name="all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./my_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="job_listings", embedding_function=self.sentence_transformer_ef
        )
        self.documents, self.metadatas, self.ids, self.documents_BM25 = process_GenAI_dataset()
        self.documents_Nexgen, self.metadatas_Nexgen, self.ids_Nexgen, self.documents_BM25_Nexgen = process_Nexgen_dataset()

    def update_database(self):
        
        if self.collection.count() == 0:
            self.collection.upsert(documents=self.documents, metadatas=self.metadatas, ids=self.ids)
            self.collection.upsert(documents=self.documents_Nexgen, metadatas=self.metadatas_Nexgen, ids=self.ids_Nexgen)
        elif self.collection.count() == len(self.documents_Nexgen):
            self.collection.upsert(documents=self.documents, metadatas=self.metadatas, ids=self.ids)
        else:
            print("Database already contains all documents. No update needed.")

    
    def predict_job_title(self, jd: str, skills: str, top_n: int):
        query_text = f"Job Description: {jd} | Skills: {skills}"
        results = self.collection.query(query_texts=[query_text], n_results=top_n)
        return results
    
    def search_with_metadata(self, jd: str, skills: str, top_n: int):
        tokenized_query = f"{jd} | {skills}".lower().split(" ")
        documents = self.documents_BM25 + self.documents_BM25_Nexgen
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in documents]
       
        bm25 = BM25Okapi(tokenized_corpus)

        # Get relevancy scores for all corpus indices
        scores = bm25.get_scores(tokenized_query)
        
        # Pair index positions with scores and sort them highest to lowest
        ranked_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        
        hits = []
        for index, score in ranked_indices:
            if score > 0:  # Filter out completely irrelevant results
                # FIX: Map properties correctly out of your centralized documents array
                hits.append({
                    "score": round(score, 4),
                    "text": documents[index]["text"],
                    "metadata": documents[index]["metadata"]
                })
        return hits
    
    def RRF(self, jd: str, skills: str, top_n: int = 2, K: int = 60):
        if jd.strip() == "" and skills.strip() == "":
            return []
        BM25_results = self.search_with_metadata(jd=jd, skills=skills, top_n=top_n)
        Chroma_results = self.predict_job_title(jd=jd, skills=skills, top_n=top_n)

        merge_result: dict = {}     
    
        for index, hit in enumerate(BM25_results):
            doc_key = f"{hit['metadata']['job_title']}_{hit['text'][:40]}" # Unique key based on job title and text snippet
            
            rrf_score = 1 / (K + index + 1)  # RRF score calculation
            merge_result[doc_key] = {
                "text":hit["text"],
                "metadata": hit["metadata"],
                "rrf_score": rrf_score
            }

        if Chroma_results and Chroma_results["documents"]:
            chroma_docs  = Chroma_results["documents"][0]   # flat list of strings
            chroma_metas = Chroma_results["metadatas"][0]   # flat list of dicts

            for index, (text, meta) in enumerate(zip(chroma_docs, chroma_metas)):
                doc_key   = f"{meta['job_title']}_{text[:40]}"
                rrf_score = 1 / (K + index + 1)

                # ✅ Fix 2 — keys() with parentheses
                if doc_key not in merge_result.keys():
                    merge_result[doc_key] = {
                        "text":      text,
                        "metadata":  meta,
                        "rrf_score": rrf_score,
                    }
                else:
                    merge_result[doc_key]["rrf_score"] += rrf_score
        final_ranked = sorted(
            merge_result.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        return [
            {"text": item["text"], "metadata": item["metadata"]}
            for item in final_ranked[:top_n]
        ]