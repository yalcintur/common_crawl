import hashlib
import json
import os
from multiprocessing import Pool, cpu_count
from google.cloud import storage
from tqdm import tqdm 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/yalcin/my-service-account.json"
client = storage.Client()

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def process_document(doc):
    text = doc.get('text')
    if not text:
        print(f"Invalid document (no text field): {doc}")
        return None

    lines = text.split("\n")
    line_hashes = {hash_text(line.strip().lower()): line for line in lines}
    unique_lines = [line for hash, line in line_hashes.items()]
    
    total_num = len(lines) - len(unique_lines)
    if total_num > 0:
        print(f"The total number of filtered lines is {total_num}")
    
    doc['text'] = "\n".join(unique_lines)
    return doc

def get_all_jsonl_files(bucket_name, prefix):
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.json')]

def read_jsonl(bucket_name, file_path):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text(encoding='utf-8')
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {line_number} in file {file_path}: {e}")
            print(f"Problematic line content (up to 200 chars): {line[:200]}")
            continue

def write_jsonl_to_gcs(bucket_name, output_dir, data):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(output_dir + '/clean_docs.jsonl')
    blob_contents = '\n'.join(json.dumps(item, ensure_ascii=False) for item in data)
    blob.upload_from_string(blob_contents, 'text/plain')

def write_jsonl_to_local(output_dir, data):
    local_path = os.path.join(output_dir, 'clean_docs.jsonl')
    with open(local_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def main():
    bucket_name = "hyperbee-turkish-scraper"
    input_dir = "processed_data/"
    output_dir = "clean_data/"
    local_output_dir = "/home/yalcin/clean_data"

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    jsonl_files = get_all_jsonl_files(bucket_name, input_dir)
    processed_docs = []

    with Pool(cpu_count()) as pool:
        for file_path in tqdm(jsonl_files, desc="Processing files"):  # wrap jsonl_files with tqdm
            docs = list(read_jsonl(bucket_name, file_path))
            processed_results = pool.map(process_document, docs)
            processed_docs.extend(filter(None, processed_results))

    print(f"Total processed documents: {len(processed_docs)}")
    write_jsonl_to_gcs(bucket_name, output_dir, processed_docs)
    write_jsonl_to_local(local_output_dir, processed_docs)
    print(f"Output written to {output_dir} on GCS and {local_output_dir} locally")

if __name__ == "__main__":
    main()