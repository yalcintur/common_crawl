import glob
import json
import concurrent.futures
import multiprocessing
import time
import os
import uuid
import jieba
import nltk
import hashlib
import pandas as pd
import logging
import argparse

from queue import Queue
from typing import List
from tqdm import tqdm

jieba.setLogLevel(logging.INFO)


def read_filenames_from_dir(dir_path: str, extension: str) -> List[str]:
    return glob.glob(f"{dir_path}/*.{extension}")


class Entry:
    def __init__(
        self, url="", timestamp="", content_language="", content_type="", text=""
    ):
        self.url = url
        self.timestamp = timestamp
        self.content_language = content_language
        self.content_type = content_type
        self.text = text

    def to_json(self):
        return json.dumps(
            {
                "url": self.url,
                "timestamp": self.timestamp,
                "content_language": self.content_language,
                "content_type": self.content_type,
                "text": self.text,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


class EntryReader:
    def __init__(self, keys: set):
        self.keys = keys

    def _process_file(self, file_path):
        entries = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if data.get("url") in self.keys:
                        entry = Entry.from_json(line)
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def read(
        self,
        filenames: List[str],
        queue: Queue,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._process_file, filenames)

            for result in results:
                for entry in result:
                    queue.put(entry)


def is_repetetive(
    text, duplicate_line_fraction=0.5, duplicate_line_character_faction=0.3
):
    def hash_text(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

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
                bag = tuple(word_list[i : i + ngram])
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


def process_entry(entry: Entry) -> Entry:
    if is_repetetive(entry.text):
        return None

    return entry


class EntryProcessor:
    def process(self, entry_queue: Queue):
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            progress = tqdm(total=entry_queue.qsize(), desc="Setting up processes")

            results = []

            while True:
                entry = entry_queue.get()

                if entry is None:
                    break

                result = pool.apply_async(process_entry, args=(entry,))
                results.append(result)

                entry_queue.task_done()
                progress.update(1)

            processed_entries = []

            for result in tqdm(results, desc="Processing Entries"):
                result = result.get()

                if result is not None:
                    processed_entries.append(result)

            return processed_entries


def read_url_keys_from_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() != ""]


def partion_list(list, n):
    for i in range(0, len(list), n):
        yield list[i : i + n]


def divide_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def save_df_to_parquet(df, file_path, log=True):
    start_time = time.time()
    df.to_parquet(file_path, compression="gzip")
    if log:
        print(f"Saved {file_path} in {time.time() - start_time} seconds.")


def makedirsifnotexists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def extract_and_derepetation(
    filenames: List[str],
    url_keys: set,
    output_dir: str,
    entries_per_file: int,
):
    entry_queue = Queue()
    reader = EntryReader(url_keys)

    reader.read(
        filenames=filenames,
        queue=entry_queue,
    )

    entry_queue.put(None)  # Poison pill

    entry_processor = EntryProcessor()
    processed_entries = entry_processor.process(entry_queue)

    del entry_queue

    for processed_entries_batch in tqdm(
        partion_list(processed_entries, entries_per_file), desc="Saving Parquet Files"
    ):
        df = pd.DataFrame([entry.__dict__ for entry in processed_entries_batch])

        file_name = str(uuid.uuid4()) + ".parquet"

        save_df_to_parquet(df, os.path.join(output_dir, file_name), log=False)


def main(args):
    SOURCE_DIR = args.source_dir  ## Folder where common crawl json files are stored
    KEY_FILES__DIR = (
        args.key_files_dir
    )  ## Folder where the .txt outputs of the dedup script are stored
    OUTPUT_DIR = args.output  ## Folder where the parquet files will be stored

    ENTRIES_PER_FILE = args.lines_per_file  ## How many entries per parquet file
    BATCH_COUNT = (
        args.batch_count
    )  ## How many batches to process, adjust according to memory usage

    makedirsifnotexists(OUTPUT_DIR)

    url_key_file_paths = read_filenames_from_dir(KEY_FILES__DIR, "txt")

    url_keys = set()
    for url_key_file_path in tqdm(url_key_file_paths, desc="Reading URL Key files"):
        url_keys.update(read_url_keys_from_file(url_key_file_path))

    del url_key_file_paths

    filenames = read_filenames_from_dir(SOURCE_DIR, "json")

    batch_filenames = list(divide_list(filenames, BATCH_COUNT))
    del filenames

    for batch_id, batch_filenames in enumerate(batch_filenames):
        print(f"Processing batch {batch_id + 1} of {BATCH_COUNT}")

        extract_and_derepetation(
            filenames=batch_filenames,
            url_keys=url_keys,
            output_dir=OUTPUT_DIR,
            entries_per_file=ENTRIES_PER_FILE,
        )

    print("Done")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Folder where common crawl json files are stored",
    )
    parser.add_argument(
        "--key_files_dir",
        type=str,
        required=True,
        help="Folder where the .txt outputs of the dedup script are stored",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder where the parquet files will be stored",
    )
    parser.add_argument(
        "--lines_per_file",
        type=int,
        default=50000,
        help="How many entries per parquet file",
    )
    parser.add_argument(
        "--batch_count",
        type=int,
        default=5,
        help="How many batches to process, adjust according to memory usage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
