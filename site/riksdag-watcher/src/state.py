import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS speeches (
    anforande_id    TEXT PRIMARY KEY,
    dok_id          TEXT,
    talare          TEXT,
    parti           TEXT,
    datum           TEXT,
    debattnamn      TEXT,
    anf_nummer      TEXT,
    anforande_url   TEXT,
    video_url       TEXT,
    local_video     TEXT,
    status          TEXT NOT NULL DEFAULT 'seen',
    error_msg       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TRIGGER IF NOT EXISTS speeches_updated
AFTER UPDATE ON speeches
BEGIN
    UPDATE speeches SET updated_at = datetime('now') WHERE anforande_id = NEW.anforande_id;
END;
"""


@contextmanager
def _conn(db_path: str):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as con:
        con.executescript(SCHEMA)


def is_known(db_path: str, anforande_id: str) -> bool:
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT 1 FROM speeches WHERE anforande_id = ?", (anforande_id,)
        ).fetchone()
        return row is not None


def insert_speech(db_path: str, speech) -> None:
    with _conn(db_path) as con:
        con.execute(
            """INSERT OR IGNORE INTO speeches
               (anforande_id, dok_id, talare, parti, datum, debattnamn, anf_nummer, anforande_url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                speech.anforande_id,
                speech.dok_id,
                speech.talare,
                speech.parti,
                speech.datum,
                speech.debattnamn,
                speech.anf_nummer,
                speech.anforande_url_xml,
            ),
        )


def set_video_url(db_path: str, anforande_id: str, video_url: str) -> None:
    with _conn(db_path) as con:
        con.execute(
            "UPDATE speeches SET video_url = ?, status = 'has_video' WHERE anforande_id = ?",
            (video_url, anforande_id),
        )


def set_status(
    db_path: str,
    anforande_id: str,
    status: str,
    local_video: Optional[str] = None,
    error_msg: Optional[str] = None,
) -> None:
    with _conn(db_path) as con:
        con.execute(
            """UPDATE speeches
               SET status = ?, local_video = COALESCE(?, local_video), error_msg = ?
               WHERE anforande_id = ?""",
            (status, local_video, error_msg, anforande_id),
        )


def get_pending(db_path: str, status: str) -> list[sqlite3.Row]:
    with _conn(db_path) as con:
        return con.execute(
            "SELECT * FROM speeches WHERE status = ?", (status,)
        ).fetchall()
