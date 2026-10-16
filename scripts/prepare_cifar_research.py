"""Public CIFAR-10 stratified benchmark: original labels, train-derived val, original test."""

import argparse
import json
from pathlib import Path
import random


def main():
    from torchvision.datasets import CIFAR10
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="runs/datasets/cifar10_research")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--train-per-class",type=int,default=100)
    parser.add_argument("--val-per-class",type=int,default=30)
    parser.add_argument("--test-per-class",type=int,default=50)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if (root/"PROVENANCE.json").exists():
        prior=json.loads((root/"PROVENANCE.json").read_text(encoding="utf-8"))
        if any(prior[k]!=getattr(args,k) for k in ("seed","train_per_class","val_per_class","test_per_class")):
            raise ValueError("Existing dataset has a different preparation protocol; use a new output folder")
        print(f"Already prepared: {root}")
        return
    if min(args.train_per_class,args.val_per_class,args.test_per_class)<1:
        raise ValueError("Per-class sample counts must be positive")
    train=CIFAR10(str(root/"raw"),train=True,download=True)
    test=CIFAR10(str(root/"raw"),train=False,download=True)
    rng=random.Random(args.seed)
    indices={}
    for label,name in enumerate(train.classes):
        candidates=[i for i,y in enumerate(train.targets) if y==label]
        rng.shuffle(candidates)
        train_ids=candidates[:args.train_per_class]
        val_ids=candidates[args.train_per_class:args.train_per_class+args.val_per_class]
        test_ids=[i for i,y in enumerate(test.targets) if y==label]
        rng.shuffle(test_ids); test_ids=test_ids[:args.test_per_class]
        if len(val_ids)!=args.val_per_class or len(test_ids)!=args.test_per_class:
            raise ValueError("Requested subset exceeds available class images")
        for split,dataset,ids in (("train",train,train_ids),("val",train,val_ids),("test",test,test_ids)):
            folder=root/split/name; folder.mkdir(parents=True,exist_ok=True)
            for index in ids:
                image,_=dataset[index]
                image.save(folder/f"{index:05d}.png")
            indices[f"{split}/{name}"]=ids
    metadata={"source":"torchvision.datasets.CIFAR10, archive downloaded and verified by torchvision",
              "url":"https://www.cs.toronto.edu/~kriz/cifar.html","seed":args.seed,
              "train_per_class":args.train_per_class,"val_per_class":args.val_per_class,
              "test_per_class":args.test_per_class,"indices":indices,
              "labels":"Original CIFAR-10 class IDs; no pseudo labels",
              "split_policy":"train/val disjoint samples from original training; test from original test"}
    (root/"PROVENANCE.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Prepared real CIFAR-10 subset: {root}")


if __name__=="__main__":
    main()
