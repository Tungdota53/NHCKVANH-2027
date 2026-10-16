"""Menu by default; optimize is research v2, --data retains explicit legacy routing."""

import time

_workflow_origin = time.monotonic()
from research_pipeline.v2.cli import main

if __name__=="__main__":
    raise SystemExit(main(origin=_workflow_origin))
