import argparse
import json
import os
from queue import Queue
import nltk

from tqdm import tqdm
from datasketch import MinHashLSH
from datetime import datetime

from doc_processor import DocProcessor
from models import DocHash
from typing import List


def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument(
        "--input_dir", required=True, help="Input directory containing JSONL files"
    )
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--lines_per_file", default=50000, help="Lines per file")

    return parser.parse_args()


def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                yield json.loads(line)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")


def makedirsifnotexists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def all_keys(doc_hashes: List[DocHash]):
    for doc_hash in doc_hashes:
        yield doc_hash.url


def deduplicate(doc_hashes: List[DocHash]):
    lsh = MinHashLSH(threshold=0.6, num_perm=128)

    for doc_hash in doc_hashes:
        try:
            lsh.insert(doc_hash.url, doc_hash.hash)
        except ValueError as e:
            pass
        except Exception as e:
            print(f"Error inserting {doc_hash.url}: {e}")

        result = lsh.query(doc_hash.hash)

        if not (len(result) > 1):
            yield doc_hash.url


def main(args):
    SOURCE_DIR = args.input_dir
    TARGET_DIR = args.output
    LINES_PER_FILE = args.lines_per_file

    makedirsifnotexists(TARGET_DIR)

    docs_queue = Queue()

    for file_name in tqdm(os.listdir(SOURCE_DIR), desc="Reading JSONL Files"):
        if file_name.endswith(".json"):
            file_path = os.path.join(SOURCE_DIR, file_name)
            for doc in read_jsonl(file_path):
                docs_queue.put(doc)

    print(f"Total number of files: {docs_queue.qsize()}")

    doc_processor = DocProcessor(docs_queue)

    doc_hashes = doc_processor.create_document_hashes()

    temp = []

    for deduplicated_url in tqdm(deduplicate(doc_hashes)):
        if deduplicated_url is not None:
            temp.append(deduplicated_url)

        if len(temp) >= LINES_PER_FILE:
            ## Write it line by line

            human_readable_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            with open(
                os.path.join(
                    TARGET_DIR, f"deduplicated_urls_{human_readable_time}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                for url in temp:
                    f.write(url + "\n")

            temp = []

    print("Done.")


if __name__ == "__main__":
    try:
        nltk.download("punkt")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

    args = parse_args()

    main(args)
