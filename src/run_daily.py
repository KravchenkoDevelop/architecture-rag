import os
import time
import datetime as dt

from parser.crawler import run_crawl
from index.faiss_indexer import run_index

INTERVAL_SEC = int(os.environ.get("INTERVAL_SEC", str(24 * 60 * 60)))

def main() -> None:
    while True:
        started = dt.datetime.now().isoformat(timespec="seconds")
        print(f"[RUN] {started} start daily pipeline")

        try:
            run_crawl()
            run_index()
        except Exception as e:
            print(f"[ERR] pipeline failed: {e}")

        finished = dt.datetime.now().isoformat(timespec="seconds")
        print(f"[SLEEP] {finished} sleep {INTERVAL_SEC}s")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
