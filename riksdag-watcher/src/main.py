import logging
import os
import time

import riksdag
import state
import downloader
import notify

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/data/state.db")
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/downloads")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "300"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "50"))


def poll_once() -> None:
    logger.info("Polling Riksdag speech list")
    try:
        speeches = riksdag.list_speeches(page_size=PAGE_SIZE)
    except Exception as e:
        logger.error("Failed to fetch speech list: %s", e)
        return

    new_count = 0
    for speech in speeches:
        if state.is_known(DB_PATH, speech.anforande_id):
            continue
        state.insert_speech(DB_PATH, speech)
        new_count += 1

    if new_count:
        logger.info("Found %d new speech(es)", new_count)
    else:
        logger.debug("No new speeches")

    process_new_speeches()


def process_new_speeches() -> None:
    for row in state.get_pending(DB_PATH, "seen"):
        anforande_id = row["anforande_id"]
        detail = riksdag.get_speech_detail(anforande_id)
        if detail is None:
            state.set_status(DB_PATH, anforande_id, "error", error_msg="fetch_detail_failed")
            continue

        video_url = riksdag.extract_video_url(detail)
        if not video_url:
            state.set_status(DB_PATH, anforande_id, "no_video")
            logger.debug("No video for %s", anforande_id)
            continue

        state.set_video_url(DB_PATH, anforande_id, video_url)

    for row in state.get_pending(DB_PATH, "has_video"):
        anforande_id = row["anforande_id"]
        video_url = row["video_url"]

        state.set_status(DB_PATH, anforande_id, "downloading")
        local_path = downloader.download_video(video_url, DOWNLOAD_DIR, anforande_id)

        if local_path is None:
            state.set_status(DB_PATH, anforande_id, "error", error_msg="download_failed")
            continue

        state.set_status(DB_PATH, anforande_id, "downloaded", local_video=local_path)

    for row in state.get_pending(DB_PATH, "downloaded"):
        anforande_id = row["anforande_id"]
        local_path = row["local_video"]

        ok = notify.notify_webhooks(row, local_path)
        if ok:
            state.set_status(DB_PATH, anforande_id, "notified")
        else:
            state.set_status(DB_PATH, anforande_id, "notify_failed")


def main() -> None:
    state.init_db(DB_PATH)
    logger.info(
        "Starting Riksdag watcher (interval=%ds, db=%s, downloads=%s)",
        POLL_INTERVAL,
        DB_PATH,
        DOWNLOAD_DIR,
    )
    while True:
        poll_once()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
