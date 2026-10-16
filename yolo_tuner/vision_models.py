"""Portable vision models, including image-conditioned captioning v2.

ResNet18 initialization is explicit and verified; inference loads the encoder
from the trained checkpoint without downloading weights. Legacy models remain.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        blocks = []
        previous = 3
        for channels in (width,width*2,width*3):
            blocks.extend([nn.Conv2d(previous,channels,3,stride=2,padding=1),
                           nn.GroupNorm(4,channels),nn.SiLU()])
            previous = channels
        self.layers = nn.Sequential(*blocks)
        self.channels = previous

    def forward(self,images):
        return self.layers(images)


class Captioner(nn.Module):
    def __init__(self,vocab_size,width=32,hidden=128):
        super().__init__()
        self.encoder = Encoder(width)
        self.embedding = nn.Embedding(vocab_size,hidden,padding_idx=0)
        self.project = nn.Linear(self.encoder.channels,hidden)
        self.decoder = nn.GRU(hidden,hidden,batch_first=True)
        self.output = nn.Linear(hidden,vocab_size)

    def forward(self,images,tokens):
        features = self.encoder(images).mean(dim=(2,3))
        initial = torch.tanh(self.project(features)).unsqueeze(0)
        decoded,_ = self.decoder(self.embedding(tokens),initial)
        return self.output(decoded)

    @torch.inference_mode()
    def generate(self, images, max_tokens=48):
        """Autoregressive greedy captions from images only, without reference tokens."""
        if max_tokens < 2:
            raise ValueError("max_tokens must include at least BOS and EOS")
        features = self.encoder(images).mean(dim=(2, 3))
        hidden = torch.tanh(self.project(features)).unsqueeze(0)
        current = torch.full((len(images), 1), 2, dtype=torch.long, device=images.device)
        finished = torch.zeros(len(images), dtype=torch.bool, device=images.device)
        generated = []
        for _ in range(max_tokens - 1):
            decoded, hidden = self.decoder(self.embedding(current), hidden)
            logits = self.output(decoded[:, -1]).clone()
            logits[:, 0] = -torch.inf  # PAD and BOS are not valid generated words.
            logits[:, 2] = -torch.inf
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.zeros_like(next_token), next_token)
            generated.append(next_token)
            finished |= next_token == 3
            current = next_token[:, None]
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1)


class Classifier(nn.Module):
    def __init__(self,classes,width=32):
        super().__init__()
        self.encoder = Encoder(width)
        self.output = nn.Linear(self.encoder.channels,classes)

    def forward(self,images):
        return self.output(self.encoder(images).mean(dim=(2,3)))


def spatial_encoder(config, load_pretrained=False):
    if config.get("encoder") == "resnet18":
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        if load_pretrained:
            from .caption_features import ensure_weights
            path, _ = ensure_weights(offline=True)
            backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        encoder = nn.Sequential(*list(backbone.children())[:-2])
        encoder.requires_grad_(False).eval()
        return encoder, 512
    encoder = Encoder(config.get("width", 32))
    return encoder, encoder.channels


class AttentionCaptioner(nn.Module):
    """Image context at every GRU step, plus spatial cross-attention.

    Legacy Captioner remains unchanged so existing checkpoints still load.
    """
    def __init__(self, vocab_size, config, load_pretrained=False):
        super().__init__()
        hidden = config["hidden"]
        self.frozen_encoder = config.get("encoder") == "resnet18"
        self.grid = config.get("feature_grid", 7)
        self.encoder, channels = spatial_encoder(config, load_pretrained)
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.project = nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.decoder = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.dropout = nn.Dropout(config.get("dropout", 0.3))
        self.output = nn.Linear(hidden * 2, vocab_size)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()  # Never update pretrained BatchNorm on cached/frozen features.
        return self

    def encode(self, images):
        features = self.encoder(images)
        features = F.adaptive_avg_pool2d(features, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        # Match the disk cache's numerical precision during independent inference.
        return features.to(torch.float16).to(torch.float32) if self.frozen_encoder else features

    def forward_features(self, features, token_ids):
        memory = self.project(features)
        context = memory.mean(dim=1)
        embeddings = self.dropout(self.embedding(token_ids))
        inputs = torch.cat((embeddings, context[:, None].expand(-1, embeddings.shape[1], -1)), dim=-1)
        states, _ = self.decoder(inputs, context.unsqueeze(0))
        attended, _ = self.attention(states, memory, memory, need_weights=False)
        return self.output(self.dropout(torch.cat((states, attended), dim=-1)))

    def forward(self, images, token_ids):
        return self.forward_features(self.encode(images), token_ids)

    @torch.inference_mode()
    def generate_features(self, features, max_tokens=48):
        if max_tokens < 2:
            raise ValueError("max_tokens must include BOS and EOS")
        memory = self.project(features)
        context = memory.mean(dim=1)
        hidden = context.unsqueeze(0)
        current = torch.full((len(features), 1), 2, dtype=torch.long, device=features.device)
        finished = torch.zeros(len(features), dtype=torch.bool, device=features.device)
        result = []
        for _ in range(max_tokens - 1):
            inputs = torch.cat((self.embedding(current), context[:, None]), dim=-1)
            states, hidden = self.decoder(inputs, hidden)
            attended, _ = self.attention(states, memory, memory, need_weights=False)
            logits = self.output(torch.cat((states, attended), dim=-1))[:, 0].clone()
            logits[:, 0] = logits[:, 2] = -torch.inf
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.zeros_like(token), token)
            result.append(token)
            finished |= token == 3
            current = token[:, None]
            if bool(finished.all()):
                break
        return torch.stack(result, dim=1)

    @torch.inference_mode()
    def generate(self, images, max_tokens=48):
        return self.generate_features(self.encode(images), max_tokens)


class Autoencoder(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.encoder = Encoder(width)
        self.decoder = nn.Sequential(nn.Conv2d(self.encoder.channels,width,3,padding=1),
                                     nn.SiLU(),nn.Conv2d(width,3,3,padding=1),nn.Sigmoid())

    def forward(self,images):
        features = F.adaptive_avg_pool2d(self.encoder(images),(4,4))
        decoded = self.decoder(features)
        return F.interpolate(decoded,size=images.shape[-2:],mode="bilinear",align_corners=False)


def build(task,config,vocab_size=0,classes=0,load_pretrained=False):
    if task == "captioning":
        if config.get("model") == "attn_gru_captioner":
            return AttentionCaptioner(vocab_size, config, load_pretrained)
        return Captioner(vocab_size,config["width"],config["hidden"])
    if task == "classification":
        return Classifier(classes,config["width"])
    if task == "reconstruction":
        return Autoencoder(config["width"])
    raise ValueError(f"Unsupported model task: {task}")
