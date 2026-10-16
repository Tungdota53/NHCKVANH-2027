"""Execution-local verification cache, never a substitute for final full audit.

Unchanged stat signatures avoid repeated file reads between trials. This detects
ordinary edits/replacements, not adversarial metadata spoofing. Full verification
at execution start/end and public audits deliberately do not trust cached hashes.
"""

from pathlib import Path
import time

from research_pipeline.integrity import sha


def signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class VerifiedFileCache:
    def __init__(self):
        self.entries = {}
        self.reads = 0
        self.hits = 0
        self.bytes_read = 0
        self.seconds = 0.0

    def verify(self, path, expected, force=False):
        start = time.monotonic()
        path = Path(path).resolve()
        before = signature(path)
        key = str(path)
        if not force and self.entries.get(key) == (before, expected):
            self.hits += 1
        else:
            actual = sha(path)
            self.reads += 1
            self.bytes_read += before[2]
            if before != signature(path):
                raise ValueError(f"File changed while verifying: {path}")
            if actual != expected:
                raise ValueError(f"File content changed: {path}")
            self.entries[key] = (before, expected)
        self.seconds += time.monotonic() - start

    def stats(self):
        return {"hash_reads": self.reads, "metadata_cache_hits": self.hits,
                "bytes_hashed": self.bytes_read, "verification_seconds": self.seconds,
                "anti_tamper_guarantee": False}


# Identity lookups are frequent while ranking candidate grids. A stat-keyed
# bounded cache avoids rereading large model files, while final/live verification
# uses sha/VerifiedFileCache independently against locked expected bytes.
_identities = {}


def content_hash(path):
    path = Path(path).resolve()
    before = signature(path)
    key = str(path)
    previous = _identities.get(key)
    if previous and previous[0] == before:
        return previous[1]
    value = sha(path)
    if signature(path) != before:
        raise ValueError(f"Model changed while hashing: {path}")
    if len(_identities) >= 1024:
        _identities.clear()
    _identities[key] = (before, value)
    return value
