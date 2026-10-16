import unittest


class VisionModelTests(unittest.TestCase):
    def test_each_task_has_finite_loss_and_encoder_gradients_at_batch_one(self):
        import torch
        import torch.nn.functional as F
        from yolo_tuner.vision_models import build
        torch.set_num_threads(1)
        config={"width":16,"hidden":32}
        image=torch.rand(1,3,32,32)
        for task in ("captioning","classification","reconstruction"):
            model=build(task,config,vocab_size=12,classes=2)
            if task=="captioning":
                token_ids=torch.tensor([[2,4,5,3]])
                logits=model(image,token_ids[:,:-1])
                loss=F.cross_entropy(logits.reshape(-1,12),token_ids[:,1:].reshape(-1))
            elif task=="classification":
                loss=F.cross_entropy(model(image),torch.tensor([1]))
            else:
                loss=F.mse_loss(model(image),image)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()))


if __name__=="__main__":
    unittest.main()
