"""Portable module entrypoint, including non-editable package installations."""

import time

_workflow_origin = time.monotonic()

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main(origin=_workflow_origin))
