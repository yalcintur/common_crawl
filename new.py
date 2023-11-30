import hashlib
import json
import os
from multiprocessing import Pool, cpu_count
from google.cloud import storage
from tempfile import NamedTemporaryFile
from tqdm import tqdm

# Set Google Cloud credentials and create a client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/yalcin/my-service-account.json"
client = storage.Client()

hash_cache = {}

def hash_text(text):
    hash_key = text.strip().lower()
    if hash_key not in hash_cache:
        hash_cache[hash_key] = hashlib.md5(hash_key.encode('utf-8')).hexdigest()
    return hash_cache[hash_key]

def process_document(doc):
    text = doc.get('text')
    if not text:
        return None

    lines = text.split("\n")
    hashed_lines = {hash_text(line) for line in lines}
    unique_lines = [line for line in lines if hash_text(line) in hashed_lines]
    doc['text'] = "\n".join(unique_lines)
    return doc

def get_all_jsonl_files(bucket_name, prefix):
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.json')]

def download_blob_to_tempfile(bucket_name, blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    temp_file = NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    return temp_file.name

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def write_to_output(bucket_name, output_dir, local_output_dir, data):
    os.makedirs(local_output_dir, exist_ok=True)
    file_index = 1
    for i in range(0, len(data), 1000):
        batch = data[i:i+1000]
        local_path = os.path.join(local_output_dir, f'clean_docs_{file_index}.jsonl')
        with open(local_path, 'w', encoding='utf-8') as file:
            for item in batch:
                json.dump(item, file, ensure_ascii=False)
                file.write('\n')
        file_index += 1

def main():
    bucket_name = "hyperbee-turkish-scraper"
    input_dir = "processed_data/"
    output_dir = "clean_data/"
    local_output_dir = "/home/yalcin/clean_data"

    jsonl_files = get_all_jsonl_files(bucket_name, input_dir)
    processed_docs = []

    # Adjust pool size if needed
    pool_size = cpu_count()
    with Pool(pool_size) as pool:
        for blob_name in tqdm(jsonl_files, desc="Processing files"):
            local_file_path = download_blob_to_tempfile(bucket_name, blob_name)
            docs = list(read_jsonl(local_file_path))
            processed_results = pool.map(process_document, docs)
            cleaned_docs = filter(None, processed_results)
            processed_docs.extend(cleaned_docs)
            os.remove(local_file_path)

    write_to_output(bucket_name, output_dir, local_output_dir, processed_docs)

if __name__ == "__main__":
    main()
