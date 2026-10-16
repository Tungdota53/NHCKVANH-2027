import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from yolo_tuner.adapters import discover,export_yolo,subset


def picture(path,index):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.new("RGB",(40,32),(index%255,(index*13)%255,(index*23)%255)).save(path)
    return path


class AdapterTests(unittest.TestCase):
    def test_corrupt_image_is_excluded_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(4):
                picture(root/f"{i}.png",i)
            corrupt=root/"broken.jpg";corrupt.write_bytes(b"not a JPEG")
            manifest=discover(root)
            self.assertTrue(corrupt.exists())
            self.assertEqual(sum(manifest["counts"].values()),4)
            self.assertTrue(any("unreadable" in w for w in manifest["warnings"]))

    def test_caption_json_deduplicates_and_preserves_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,indices in (("train",range(4)),("val",range(4,6)),("test",range(6,8))):
                images=[];anns=[]
                for i in indices:
                    picture(root/"images"/f"{i}.png",i)
                    images.append({"id":i,"file_name":f"{i}.png"})
                    anns.extend([{"image_id":i,"caption":f"Ảnh {i}"},{"image_id":i,"caption":f"Ảnh {i}"}])
                (root/f"captions_{split}.json").write_text(json.dumps({"images":images,"annotations":anns}),encoding="utf-8")
            manifest=discover(root)
            self.assertEqual(manifest["task"],"captioning")
            self.assertEqual(manifest["counts"],{"train":4,"val":2,"test":2})
            self.assertEqual(manifest["captions"]["train"],4)
            short=subset(manifest,2,1)
            self.assertFalse(short["splits"]["test"])
            self.assertEqual(short,subset(manifest,2,1))

    def test_yolo_layout_without_yaml_infers_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(6):
                split="train" if i<4 else "val"
                picture(root/"images"/split/f"{i}.png",i)
                labels=root/"labels"/split
                labels.mkdir(parents=True,exist_ok=True)
                (labels/f"{i}.txt").write_text("1 0.5 0.5 0.3 0.4\n")
            manifest=discover(root)
            self.assertEqual(manifest["task"],"detection")
            self.assertEqual(manifest["names"],["class_0","class_1"])
            out=export_yolo(manifest,root/"converted")
            self.assertTrue(out.exists())
            self.assertEqual(len(list((out.parent/"labels"/"train").glob("*.txt"))),4)

    def test_coco_in_annotations_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/"annotations";folder.mkdir()
            images=[];anns=[]
            for i in range(5):
                picture(root/"images"/f"{i}.png",i)
                images.append({"id":i,"file_name":f"{i}.png"})
                anns.append({"image_id":i,"category_id":7,"bbox":[4,3,12,8]})
            file=folder/"instances_train.json"
            file.write_text(json.dumps({"images":images,"annotations":anns,"categories":[{"id":7,"name":"object"}]}))
            manifest=discover(file)
            self.assertEqual(manifest["task"],"detection")
            self.assertEqual(manifest["names"],["object"])
            self.assertEqual(manifest["counts"]["val"],1)
            self.assertEqual(manifest["splits"]["train"][0]["boxes"][0][0],0)

    def test_class_folders_and_raw_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for cls in range(2):
                for i in range(4):
                    picture(root/f"class{cls}"/f"{i}.png",10*cls+i)
            manifest=discover(root)
            self.assertEqual(manifest["task"],"classification")
            self.assertEqual(len(manifest["splits"]["val"]),2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(5):
                picture(root/f"{i}.png",i)
            manifest=discover(root)
            self.assertEqual(manifest["task"],"reconstruction")
            self.assertTrue(manifest["validation_ready"])

    def test_same_content_cannot_leak_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            picture(root/"train"/"a.png",1)
            picture(root/"train"/"b.png",2)
            picture(root/"train"/"c.png",3)
            picture(root/"val"/"duplicate.png",1)
            manifest=discover(root)
            train={r["sha256"] for r in manifest["splits"]["train"]}
            val={r["sha256"] for r in manifest["splits"]["val"]}
            self.assertFalse(train & val)
            self.assertTrue(any("Duplicate" in w for w in manifest["warnings"]))

    def test_labelme_rectangles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(4):
                picture(root/f"{i}.png",i)
                (root/f"{i}.json").write_text(json.dumps({"imagePath":f"{i}.png","shapes":[
                    {"label":"ball","shape_type":"rectangle","points":[[2,4],[12,14]]}]}))
            manifest=discover(root)
            self.assertEqual(manifest["task"],"detection")
            self.assertEqual(manifest["names"],["ball"])

    def test_single_image_returns_planable_but_not_quality_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);picture(root/"a.png",1)
            manifest=discover(root)
            self.assertFalse(manifest["validation_ready"])


if __name__=="__main__":
    unittest.main()
