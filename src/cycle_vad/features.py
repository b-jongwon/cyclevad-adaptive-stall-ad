"""Frozen visual towers, full-field preprocessing, AMP, and resumable caches.

No Qwen, text encoder, prompts, object vocabulary, or discrete phase generator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .data import fingerprint, frame_paths, load_cache, validate_cache


class VisualEncoder:
    def __init__(self, config, model=None, processor=None):
        self.cfg = dict(config)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.family = config.get("family", "dinov2")
        if model is None:
            from transformers import AutoImageProcessor, Dinov2Model, CLIPVisionModel
            cls = {"dinov2": Dinov2Model, "clip": CLIPVisionModel}[self.family]
            kwargs = {"revision": config.get("revision", "main"), "attn_implementation": "sdpa"}
            # output_hidden_states is used for multi-layer patch descriptors.
            model = cls.from_pretrained(config["model_id"], **kwargs)
            processor = AutoImageProcessor.from_pretrained(config["model_id"],
                                                           revision=config.get("revision", "main"), use_fast=False)
        self.model = model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.processor = processor
        self.size = int(config.get("image_size", 336))
        self.grid = int(config.get("patch_grid", 6))
        if self.size % self.model.config.patch_size or self.grid < 1:
            raise ValueError("Image size must be a multiple of the backbone patch size")
        self.layers = tuple(config.get("feature_layers", [-1, -3]))
        channels = self.model.config.hidden_size * len(self.layers)
        rng = np.random.default_rng(config.get("seed", 42))
        out_dim = int(config.get("patch_dim", 64))
        self.projection = torch.tensor(rng.standard_normal((channels, out_dim))/np.sqrt(out_dim),
                                       dtype=torch.float32, device=self.device)
        self.identity = {**self.cfg, "resolved_revision": getattr(model.config, "_commit_hash", None),
                         "extractor_version": 1, "preprocessing": "full_frame_letterbox",
                         "torch": torch.__version__}

    def preprocess(self, image):
        # Center cropping would discard much of IPAD's wide frame. Letterbox
        # preserves every device region. Padding is fixed, mean-colored.
        mean = getattr(self.processor, "image_mean", [.485, .456, .406])
        fill = tuple(int(round(v*255)) for v in mean)
        image = ImageOps.pad(image.convert("RGB"), (self.size, self.size),
                             method=Image.Resampling.BICUBIC, color=fill)
        return self.processor(images=image, return_tensors="pt", do_resize=False,
                              do_center_crop=False)["pixel_values"][0]

    @torch.inference_mode()
    def encode(self, images):
        pixels = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            kwargs = {"pixel_values": pixels, "output_hidden_states": True, "return_dict": True}
            if self.family == "clip":
                kwargs["interpolate_pos_encoding"] = True
            out = self.model(**kwargs)
            global_z = F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)
            side = self.size // self.model.config.patch_size
            layers = []
            for layer in self.layers:
                tokens = out.hidden_states[layer][:, 1:].float()
                if tokens.shape[1] != side*side:
                    raise ValueError("Unexpected token layout: use DINOv2 without register tokens or CLIP")
                spatial = F.normalize(tokens, dim=-1).transpose(1, 2).reshape(len(images), -1, side, side)
                spatial = F.adaptive_avg_pool2d(spatial, (self.grid, self.grid)).flatten(2).transpose(1, 2)
                layers.append(F.normalize(spatial, dim=-1))
        # Fixed seeded random projection reduces cache size, is data-independent,
        # and is identical for FIT/validation/calibration/test.
        patches = F.normalize(torch.cat(layers, dim=-1) @ self.projection, dim=-1)
        return global_z.cpu().numpy(), patches.cpu().numpy()


def _read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_sequence(encoder, row, cache_root, stride=2, batch_size=8, workers=4):
    if stride < 1 or batch_size < 1:
        raise ValueError("Positive stride and batch_size required")
    files = frame_paths(row["directory"])
    indices = np.arange(0, len(files), stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError(f"Sequence too short at requested stride: {row['id']}")
    # Stat metadata for all frames detects replacement/length changes without
    # rereading image contents from Drive. Content hashes are not claimed.
    source = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    signature = fingerprint({"source": source, "stride": stride, "encoder": encoder.identity})
    path = Path(cache_root) / (row["id"] + ".npz")
    if path.exists():
        cache = load_cache(path)
        if str(cache["signature"]) == signature:
            print(f"[cached] {row['id']}", flush=True)
            return path
    started = time.perf_counter()
    globals_, patches = [], []
    cursor = 0
    batch = batch_size
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while cursor < len(indices):
            selected = indices[cursor:cursor+batch]
            images = list(pool.map(_read_image, [files[i] for i in selected]))
            try:
                global_z, local_z = encoder.encode(images)
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise RuntimeError("GPU OOM at batch=1: reduce image_size or use dinov2-small")
                batch = max(1, batch//2)
                torch.cuda.empty_cache()
                print(f"[OOM retry] batch_size={batch}", flush=True)
                continue
            globals_.append(global_z.astype(np.float16))
            patches.append(local_z.astype(np.float16))
            cursor += len(selected)
            if cursor % (batch*50) == 0:
                print(f"[extract] {row['id']} {cursor}/{len(indices)}", flush=True)
    data = {"global": np.concatenate(globals_), "patches": np.concatenate(patches),
            "indices": indices, "frame_count": np.array(len(files)), "signature": np.array(signature),
            "encoder_identity": np.array(json.dumps(encoder.identity, sort_keys=True))}
    validate_cache(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **data)
    os.replace(temp, path)
    elapsed = time.perf_counter()-started
    print(f"[extracted] {row['id']}: {len(indices)} observations in {elapsed:.1f}s", flush=True)
    return path
