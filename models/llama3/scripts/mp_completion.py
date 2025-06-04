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
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    queue=None,
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
    print(generator.args.vision_chunk_size)
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        interleaved_contents.append(
                [
                RawMediaItem(type="image", data=BytesIO(img)),
                "If I had to write a haiku for this one",
                ]
            )

    for content in interleaved_contents:
        cprint(f"{content}", end="")
        batch = [content]

        xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = generator.vision_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
        )
        interm_data = (xattn_caches, cross_attention_masks,
                       full_text_row_masked_out_mask)
        queue.put(interm_data)
        print(f"proc1: {queue.qsize()}")
        print("\n==================================\n")

    while True:
        time.sleep(1)
    print("Done")


def run_text(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    queue=None,
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

        interleaved_contents.append(
            [
                RawMediaItem(type="image", data=BytesIO(img)),
                "If I had to write a haiku for this one",
            ]
        )
    while True:
        if queue.qsize() == 1:
            break
    interm_data = queue.get()
    print(f"proc2: {queue.qsize()}")
    for content in interleaved_contents:
        cprint(f"{content}", end="")
        batch = [content]

        results = generator.text_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            interm_data=interm_data,
        )
        for token_result in results:
            cprint(token_result.text, color="yellow", end="")
        print("\n==================================\n")


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # Handle termination signal

    set_start_method('spawn', force=True)
    q = Queue()

    p1 = Process(target=run_vision,
                kwargs={'ckpt_dir': '/home/tkim/.llama/checkpoints/Llama3.2-11B-Vision',
                # kwargs={'ckpt_dir': '/home/tkim/.llama/checkpoints/Llama3.1-8B',
                         'queue': q})
    p2 = Process(target=run_text,
                kwargs={'ckpt_dir': '/home/tkim/.llama/checkpoints/Llama3.2-11B-Vision',
                         'queue': q})
    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    main()
