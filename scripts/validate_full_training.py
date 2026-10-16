"""Exercise the public full-training path for a bounded number of epochs.

Writes a separate validation selection; never changes the user's recommended run.
"""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from yolo_tuner.cli import parser,train
from yolo_tuner.pipeline import new_session,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",required=True)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--device",default="auto")
    p.add_argument("--resume")
    p.add_argument("--output",default="runs/full_pipeline_validation")
    args=p.parse_args()
    selection=json.loads(Path(args.selection).read_text(encoding="utf-8"))
    folder=new_session(args.output)
    selection["validation_source_selection"]=str(Path(args.selection).resolve())
    selection["full_epochs"]=args.epochs
    selection["config"]["epochs"]=args.epochs
    write_json(folder/"validation_selection.json",selection)
    argv=["train","--selection",str(folder/"validation_selection.json"),"--device",args.device,
          "--timeout-hours","0.1","--output",str(folder/"training")]
    if args.resume:
        argv.extend(["--resume",args.resume])
    raise SystemExit(train(parser().parse_args(argv)))


if __name__=="__main__":
    main()
