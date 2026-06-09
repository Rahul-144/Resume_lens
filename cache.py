import pickle
import hashlib
import pathlib
import os

CACHE_DIR = pathlib.Path(".cache/resume")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _pdf_hash(pdf_path: str) -> str:
    """Generate a hash for the PDF file to use as a cache key."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
def load_from_cache(pdf_path: str):
    """Load extracted resume data from cache if available."""
    cache_key = _pdf_hash(pdf_path)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            print(f"[Cache] Loaded extracted data for {pdf_path} from cache.")
            return pickle.load(f)
    return None
def save_to_cache(pdf_path: str, data):
    """Save extracted resume data to cache."""
    cache_key = _pdf_hash(pdf_path)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
        print(f"[Cache] Saved extracted data for {pdf_path} to cache.")

