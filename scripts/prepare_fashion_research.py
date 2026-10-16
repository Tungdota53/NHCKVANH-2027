"""Prepare a new real Fashion-MNIST domain with original labels and content-disjoint splits."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    from torchvision.datasets import FashionMNIST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="runs/datasets/fashion_research")
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--train-per-class",type=int,default=100)
    parser.add_argument("--val-per-class",type=int,default=30)
    parser.add_argument("--test-per-class",type=int,default=50)
    args=parser.parse_args(); root=Path(args.output).resolve()
    marker=root/"PROVENANCE.json"
    if marker.exists():
        previous=json.loads(marker.read_text(encoding="utf-8"))
        if any(previous[k]!=getattr(args,k) for k in ("seed","train_per_class","val_per_class","test_per_class")):
            raise ValueError("Existing dataset protocol differs; use a new output")
        for key,ids in previous["indices"].items():
            if any(not (root/key/f"{index:05d}.png").exists() for index in ids):
                raise ValueError("Prepared dataset is incomplete; inspect it before reuse")
        print(f"Already prepared: {root}"); return
    if min(args.train_per_class,args.val_per_class,args.test_per_class)<1:
        raise ValueError("Counts must be positive")
    # The official maintainer's HTTPS repository; torchvision checks official MD5s.
    class OfficialFashion(FashionMNIST):
        mirrors=["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    train=OfficialFashion(str(root/"raw"),train=True,download=True)
    test=OfficialFashion(str(root/"raw"),train=False,download=True)
    rng=random.Random(args.seed); used=set(); indices={}; skipped=0
    names=["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle-boot"]
    for label,name in enumerate(names):
        original_train=[i for i,y in enumerate(train.targets.tolist()) if y==label]
        original_test=[i for i,y in enumerate(test.targets.tolist()) if y==label]
        rng.shuffle(original_train); rng.shuffle(original_test)
        cursor=iter(original_train)
        for split,dataset,candidates,count in (("train",train,cursor,args.train_per_class),
                ("val",train,cursor,args.val_per_class),("test",test,iter(original_test),args.test_per_class)):
            folder=root/split/name; folder.mkdir(parents=True,exist_ok=True); chosen=[]
            for index in candidates:
                image,_=dataset[index]
                fingerprint=hashlib.sha256(image.tobytes()).hexdigest()
                if fingerprint in used:
                    skipped+=1; continue
                used.add(fingerprint); chosen.append(index)
                image.save(folder/f"{index:05d}.png")
                if len(chosen)==count:
                    break
            if len(chosen)!=count:
                raise ValueError("Cannot create the requested content-disjoint split")
            indices[f"{split}/{name}"]=chosen
    metadata={"source":"Official Zalando Fashion-MNIST via torchvision, verified archive MD5",
              "url":"https://github.com/zalandoresearch/fashion-mnist","seed":args.seed,
              "train_per_class":args.train_per_class,"val_per_class":args.val_per_class,
              "test_per_class":args.test_per_class,"indices":indices,"skipped_duplicate_content":skipped,
              "labels":"Original Fashion-MNIST labels, not pseudo labels",
              "split_policy":"content-disjoint train/val from original train, test from official test; no test metric used for tuning"}
    marker.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Prepared real Fashion-MNIST: {root}; skipped {skipped} duplicate images")


if __name__=="__main__":
    main()
