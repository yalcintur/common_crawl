import argparse
import hashlib
import json
import os
from multiprocessing import Pool, cpu_count
import nltk
import jieba
import pandas as pd
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing JSONL files")
    parser.add_argument("--output", default="./", help="Output directory")
    return parser.parse_args()

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def document_hash(doc):
    return hashlib.md5(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()

def is_repetition_removal(text, duplicate_line_fraction=0.5, duplicate_line_character_faction=3):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    line_count = 0
    dup_line = 0
    dup_line_chars = 0
    visit_lines = {}
    for line in text.split("\n"):
        line_hash = hash_text(line)
        if line_hash in visit_lines:
            dup_line += 1
            dup_line_chars += len(line)
        visit_lines[line_hash] = True
        line_count += 1

    if float(dup_line) / line_count > duplicate_line_fraction:
        return True

    if float(dup_line_chars) / len(text) > duplicate_line_character_faction:
        return True

    top_ngram_character_fractions = [
        (2, 0.20),
        (3, 0.17),
        (4, 0.15),
    ]
    for ngram, threshold in top_ngram_character_fractions:
        try:
            word_list = list(jieba.cut(text))
            bgs = nltk.ngrams(word_list, ngram)
            fdist = nltk.FreqDist(bgs)
            for word_list, repeat in fdist.items():
                char_count = sum([len(word) for word in word_list])
                if char_count * (repeat - 1) / len(text) > threshold:
                    return True
        except Exception as e:
            print(f"Error in ngram analysis: {e}")

    duplicate_ngram_character_fractions = [
        (5, 0.27),
        (6, 0.25),
        (7, 0.23),
        (8, 0.20),
        (9, 0.18),
        (10, 0.15),
    ]
    for ngram, threshold in duplicate_ngram_character_fractions:
        try:
            fdist = {}
            word_list = list(jieba.cut(text))
            mark = [0] * len(word_list)
            for i in range(len(word_list) - ngram + 1):
                bag = tuple(word_list[i: i + ngram])
                if bag in fdist:
                    for j in range(i, i + ngram):
                        mark[j] = len(word_list[j])
                    fdist[bag] += 1
                else:
                    fdist[bag] = 1

            if sum(mark) / float(len(text)) > threshold:
                return True
        except Exception as e:
            print(f"Error in duplicate ngram analysis: {e}")

    return False

def process_document(doc):
    if not isinstance(doc, dict) or "text" not in doc:
        raise ValueError("Document must be a dictionary with a 'text' key")
    return doc, is_repetition_removal(doc["text"])

def read_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")

def process_documents_in_chunks(docs, chunk_size=10000):
    chunk = []
    for doc in docs:
        chunk.append(doc)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def write_parquet(data, output_dir, base_file_name, rows_per_file):
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory {output_dir}")

    file_count = 0
    for i in tqdm(range(0, len(data), rows_per_file), desc=f"Writing {base_file_name}"):
        try:
            chunk = data[i:i + rows_per_file]
            df = pd.DataFrame(chunk)
            file_path = os.path.join(output_dir, f"{base_file_name}_{file_count}.parquet")
            df.to_parquet(file_path, index=False)
            file_count += 1
        except Exception as e:
            print(f"Error writing parquet file {file_path}: {e}")

def create_minhash(doc):
    """Create a MinHash object for a given document."""
    m = MinHash()
    for d in nltk.word_tokenize(doc):
        m.update(d.encode('utf8'))
    return m
# 79.41 - 78.7 = 0.71 in 40 mins

#2.200.000 ın 220 mins
#16.500.000 in 1600mıns  15
def deduplicate_docs_with_minhash(docs, num_perm=128, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicates = set()
    total_docs = len(docs)
    for i, doc in tqdm(enumerate(docs), desc="Building MinHash for each document"):
        minhash = create_minhash(doc['text'])
        lsh.insert(f"doc_{i}", minhash)
        result = lsh.query(minhash)
        if len(result) > 1:
            duplicates.update(result)
            if len(duplicates) % 1000 == 0:
                print(f"Total docs processed {i}")
                print(f"Total duplicates {len(duplicates)}")

    return [doc for i, doc in enumerate(docs) if f"doc_{i}" not in duplicates]
def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist")

    os.makedirs(args.output, exist_ok=True)
    file_count = 0
    seen_hashes = set()
    total_docs = 0
    duplicate_docs = 0
    repetitive_docs = 0

    docs = list()

    for file_name in tqdm(os.listdir(args.input_dir), desc="Reading JSONL Files"):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(args.input_dir, file_name)
            docs += list(read_jsonl(file_path))
            total_docs += len(docs)
    try:
        unique_docs = deduplicate_docs_with_minhash(docs)

        duplicate_docs += (len(docs) - len(unique_docs))

        print(f"Total documents processed: {total_docs}")
        print(f"Duplicate documents removed: {duplicate_docs}")
        print(f"Documents removed due to repetition: {repetitive_docs}")

        for chunk in process_documents_in_chunks(unique_docs, chunk_size=500000):
            with Pool(cpu_count()) as pool:
                results = list(tqdm(pool.imap(process_document, chunk), total=len(chunk), desc="Processing Documents"))

            clean_docs = [doc for doc, is_bad in results if not is_bad]
            bad_docs = [doc for doc, is_bad in results if is_bad]
            repetitive_docs += len(bad_docs)

            write_parquet(clean_docs, args.output, f"clean_docs_{file_count}", len(clean_docs))
            write_parquet(bad_docs, args.output, f"bad_docs_{file_count}", len(bad_docs))

            file_count += 1
            print(f"Total documents processed: {total_docs}")
            print(f"Duplicate documents removed: {duplicate_docs}")
            print(f"Documents removed due to repetition: {repetitive_docs}")

    except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Total documents processed: {total_docs}")
    print(f"Duplicate documents removed: {duplicate_docs}")
    print(f"Documents removed due to repetition: {repetitive_docs}")

if __name__ == "__main__":
    try:
        nltk.download('punkt')
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    main()
