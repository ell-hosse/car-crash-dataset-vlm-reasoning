"""Real-time asynchronous captioner using a free, fast VLM (SmolVLM2-256M).

Runs in a background thread: grabs the most recent frame, generates a caption,
publishes it. The trajectory model always reads the *latest* caption without
ever blocking on generation - that is what keeps the main loop real-time.
"""
import threading
import time

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import REALTIME


class AsyncCaptioner:
    def __init__(self, device: torch.device, cfg=REALTIME):
        self.cfg = cfg
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.captioner_model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.captioner_model, torch_dtype=dtype).to(device).eval()

        self._latest_frame = None
        self._lock = threading.Lock()
        self.caption = "The ego vehicle is driving on a road."  # bootstrap
        self.caption_version = 0
        self.last_latency_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # -- main-thread API ----------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def submit_frame(self, bgr):
        with self._lock:
            self._latest_frame = bgr

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    # -- worker -------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": self.cfg.caption_prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(
                self.device, dtype=self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg.caption_max_new_tokens,
            do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip()

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.02)
                continue
            t0 = time.time()
            try:
                caption = self._generate(frame)
                if caption:
                    self.caption = caption
                    self.caption_version += 1
                self.last_latency_s = time.time() - t0
            except Exception as e:  # noqa: BLE001
                print(f"[captioner] error: {e}")
            # throttle to the configured interval
            wait = self.cfg.caption_interval_s - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)
