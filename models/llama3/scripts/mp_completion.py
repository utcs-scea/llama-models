# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.


from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
import os
import torch
import signal
import time

from termcolor import cprint

from models.datatypes import RawMediaItem
from models.llama3.generation import Llama3

from torch.multiprocessing import Queue, Process, Event, set_start_method

# Flag for signal
running = True

THIS_DIR = Path(__file__).parent


def handle_exit(signum, frame):
    global running
    print(f"Received signal {signum}, exiting...")
    running = False


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_vision(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    request_len: int = 10,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    queue=None,
    start_event=None,
    end_event=None,
):
    vision_only = True
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        vision_only=vision_only,
    )

    interleaved_contents = []
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        for i in range (request_len):
            interleaved_contents.append(
                    [
                    RawMediaItem(type="image", data=BytesIO(img)),
                    "If I had to write a haiku for this one",
                    ]
                )
    assert(len(interleaved_contents) == request_len)

    print("Waiting for start_event...")
    start_event.wait()

    total_vis_time = 0
    num_done = 0
    #for content in interleaved_contents:
    content = interleaved_contents[0]
    #for num_done in range(request_len):
    while True:
        batch = [content]

        vis_time, xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = generator.vision_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
        )
        if num_done != 0:
            total_vis_time += vis_time
        num_done += 1 
        #print(f"Req {num_done}, Cur vision latency {vis_time / 1000:.3f} sec")
        if end_event.is_set():
            break

    print(f"Average vision latency {total_vis_time / (num_done-1) / 1000:.3f} sec")


def run_text(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    request_len: int = 10,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    queue=None,
    start_event=None,
    end_event=None,
):
    vision_only = False
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
        vision_only=vision_only,
    )

    interleaved_contents = []
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        for i in range (request_len):
            interleaved_contents.append(
                [
                    RawMediaItem(type="image", data=BytesIO(img)),
                    "If I had to write a haiku for this one",
                ]
            )
    assert(len(interleaved_contents) == request_len)

    content = interleaved_contents[0]
    batch = [content]
    vis_time, xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = generator.vision_completion(
        batch,
        temperature=temperature,
        top_p=top_p,
    )
    interm_data = (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask)

    num_done = 0
    total_text_time = 0
    start_event.set()
    for i in range(request_len):
        text_time, results = generator.text_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            interm_data=interm_data,
        )
        if num_done != 0:
            total_text_time += text_time
        num_done += 1
        #print(f"Req {num_done}, Cur text lat {text_time / 1000:.3f} sec")

#        for token_result in results:
#            cprint(token_result.text, color="yellow", end="")
    end_event.set()
    print(f"Average text latency {total_text_time / (num_done-1) / 1000:.3f} sec")


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # Handle termination signal

    set_start_method('spawn', force=True)

    start_event = Event()
    end_event = Event()
    q = Queue()

    p1 = Process(target=run_vision,
                kwargs={'ckpt_dir': '/home/kimtaekl/.llama/checkpoints/Llama3.2-11B-Vision',
                        'queue': q,
                        'start_event': start_event,
                        'end_event': end_event})
    p2 = Process(target=run_text,
                kwargs={'ckpt_dir': '/home/kimtaekl/.llama/checkpoints/Llama3.2-11B-Vision',
                        'queue': q,
                        'start_event': start_event,
                        'end_event': end_event})
    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    main()
