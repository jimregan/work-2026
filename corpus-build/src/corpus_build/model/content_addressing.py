"""Content-addressing helpers.

Convention, not enforcement: a `LayerId.local_id` should be one of these
hashes wherever possible. The core model does not compute ids itself — it
has no notion of "the current content" of an artifact to hash from; whoever
constructs an id calls the appropriate function here.

Two cases:

- Content already exists: hash the bytes directly (`content_address`). This
  is the default for interior and terminal artifacts alike — a finished
  artifact is still identified by a hash of its bytes, even though its
  storage location may additionally carry a human-readable filename for
  consumers outside this library.
- Content does not exist yet: an intake/specification node ahead of a fetch
  (a URL, not yet retrieved) is addressed by its origin and acquisition
  date instead (`locator_address`). The date must be part of the address:
  the same origin fetched on two different days can yield different
  content, so two fetches of one URL are two distinct intake nodes, not one
  node whose content changed underneath a stable id.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime


def content_address(data: bytes) -> str:
    """SHA-256 hex digest of materialized content bytes."""
    return hashlib.sha256(data).hexdigest()


def locator_address(origin: str, at: datetime | date) -> str:
    """SHA-256 hex digest of an intake node's origin address and
    acquisition date, for artifacts declared before any content exists."""
    canonical = f"{at.isoformat()}\n{origin}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
