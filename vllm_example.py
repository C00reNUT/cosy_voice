#!/usr/bin/env python3
"""CosyVoice vLLM accelerated inference examples.

Provides 5-10x speedup over PyTorch baseline using vLLM backend.

Micromamba env: cosyvoice_vllm

Model paths:
    - CosyVoice v2: pretrained_models/CosyVoice2-0.5B
    - CosyVoice v3: pretrained_models/Fun-CosyVoice3-0.5B
    - Fine-tuned: /path/to/CosyVoice3-YourModel

Usage:
    python vllm_example.py

AutoModel flags:
    - load_vllm=True: Enable vLLM backend (5x speedup)
    - load_trt=True: Enable TensorRT for DiT decoder
    - load_jit=True: Enable TorchScript JIT
    - fp16=True/False: Half precision (True for v2, False for v3)

Batch processing for higher throughput (2x with 6 workers):
    from concurrent.futures import ThreadPoolExecutor

    def infer(text):
        for r in cosyvoice.inference_instruct2(text, ...):
            return r['tts_speech']

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(infer, sentences))

Performance (RTX 3090):
    - Sequential: RTF 0.192 (5.2x real-time)
    - Concurrent (6 workers): RTF 0.099 (10.1x real-time)
"""
import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_example():
    """ CosyVoice3 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
            continue


def main():
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
