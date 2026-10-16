"""Create tiny image fixtures in runs/ for task-routing integration checks."""

from pathlib import Path
import random
from PIL import Image,ImageDraw


def main():
    root=Path("runs/pipeline_fixtures")
    rng=random.Random(17)
    for split,n in (("train",8),("val",4)):
        for cls in (0,1):
            for i in range(n):
                im=Image.new("RGB",(64,64),(rng.randint(1,30),rng.randint(1,30),rng.randint(1,30)))
                draw=ImageDraw.Draw(im)
                color=(220,rng.randint(1,40),30) if cls==0 else (30,rng.randint(1,40),220)
                x=rng.randint(4,20);y=rng.randint(4,20)
                draw.rectangle((x,y,x+24,y+24),fill=color)
                path=root/"classification"/split/f"class_{cls}"/f"{i}.png"
                path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
                raw=root/"reconstruction"/split/f"{cls}_{i}.png"
                raw.parent.mkdir(parents=True,exist_ok=True);im.save(raw)
    print(root.resolve())


if __name__=="__main__":
    main()
