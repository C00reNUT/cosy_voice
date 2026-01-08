#!/usr/bin/env python3
"""Czech CosyVoice3 evaluation benchmark - compare inference methods.

Tests both inference_cross_lingual and inference_instruct2 methods
using the Czech evaluation sentences from examples/czech.

Micromamba env: cosyvoice
"""
import time
import sys
import os

sys.path.insert(0, '.')
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

import logging
logging.basicConfig(level=logging.ERROR)

import torchaudio

# Register vLLM model
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel

# Import Czech eval sentences
from examples.czech.local.eval_sentences import get_eval_sentences

MODEL_DIR = '/mnt/8TB/AUDIO/TEXT_TO_SPEECH/MODELS/CosyVoice3-Czech-HobbitDeep'
# Use 7s hobbit reference (from 12s dataset) - optimal for vLLM/TRT
PROMPT_WAV = '/mnt/8TB/PYTHON_CODE/PROJECTS/audio_dataset_maker/Hobbit_deep_12s_Dataset/segments/hobbit_deep_voice_PART_ONE_var2_segment_0000.wav'
OUTPUT_DIR = './outputs/czech_eval_hobbit'

# Default instruction for instruct2 method
DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def format_prompt_text(text: str) -> str:
    """Format prompt text with endofprompt marker."""
    prompt_text = (text or '').strip()
    if not prompt_text:
        return '<|endofprompt|>'
    if '<|endofprompt|>' not in prompt_text:
        prompt_text = f'{prompt_text} <|endofprompt|>'
    return prompt_text


def run_benchmark(cosyvoice, sentences, method='cross_lingual', output_subdir='', output_base=None):
    """Run benchmark with specified inference method."""
    base = output_base or OUTPUT_DIR
    output_path = os.path.join(base, output_subdir) if output_subdir else base
    os.makedirs(output_path, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'Method: {method}')
    print(f'{"="*70}')

    results = []
    total_audio = 0
    total_time = 0

    for i, sentence in enumerate(sentences, 1):
        start = time.time()

        if method == 'cross_lingual':
            for r in cosyvoice.inference_cross_lingual(
                sentence,
                PROMPT_WAV,
                stream=False,
                speed=1.0
            ):
                audio = r['tts_speech']
        elif method == 'instruct2':
            prompt_text = format_prompt_text(DEFAULT_INSTRUCT)
            for r in cosyvoice.inference_instruct2(
                tts_text=sentence,
                instruct_text=prompt_text,
                prompt_wav=PROMPT_WAV,
                stream=False,
                speed=1.0,
                text_frontend=False  # Preserve Czech diacritics
            ):
                audio = r['tts_speech']
        else:
            raise ValueError(f"Unknown method: {method}")

        infer_time = time.time() - start
        audio_len = audio.shape[1] / cosyvoice.sample_rate
        rtf = infer_time / audio_len

        # Save to file
        filename = f'{method}_{i:02d}.wav'
        filepath = os.path.join(output_path, filename)
        torchaudio.save(filepath, audio.cpu(), cosyvoice.sample_rate)

        # Truncate for display
        display = sentence[:50] + '...' if len(sentence) > 50 else sentence
        print(f'{i:>2} | {audio_len:>5.1f}s | {infer_time:>5.2f}s | RTF {rtf:.3f} | {display}')

        results.append({
            'sentence': sentence,
            'audio_len': audio_len,
            'infer_time': infer_time,
            'rtf': rtf
        })
        total_audio += audio_len
        total_time += infer_time

    avg_rtf = total_time / total_audio
    print(f'{"-"*70}')
    print(f'TOTAL: {total_audio:.1f}s audio in {total_time:.1f}s | Avg RTF: {avg_rtf:.3f} ({1/avg_rtf:.1f}x real-time)')

    return {
        'method': method,
        'total_audio': total_audio,
        'total_time': total_time,
        'avg_rtf': avg_rtf,
        'results': results
    }


def main():
    """Run Czech evaluation benchmark."""
    import argparse
    parser = argparse.ArgumentParser(description='CosyVoice3 Czech Benchmark')
    parser.add_argument('--trt', action='store_true', help='Enable TensorRT')
    parser.add_argument('--vllm', action='store_true', help='Enable vLLM')
    parser.add_argument('--method', choices=['both', 'cross_lingual', 'instruct2'],
                        default='both', help='Inference method to test')
    args = parser.parse_args()

    # Determine backend name
    if args.trt and args.vllm:
        backend = 'TRT+vLLM'
    elif args.trt:
        backend = 'TensorRT'
    elif args.vllm:
        backend = 'vLLM'
    else:
        backend = 'PyTorch'

    print('='*70)
    print(f'CosyVoice3 Czech - Eval Sentences Benchmark [{backend}]')
    print('='*70)

    # Load eval sentences
    sentences = get_eval_sentences()
    print(f'\nLoaded {len(sentences)} Czech evaluation sentences')

    # Load model
    print(f'\nLoading model with {backend}...')
    start = time.time()
    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=args.trt,
        load_vllm=args.vllm,
        fp16=False
    )
    load_time = time.time() - start
    print(f'Model loaded in {load_time:.1f}s')

    # Update output dir with backend name
    output_base = f'{OUTPUT_DIR}_{backend.lower().replace("+", "_")}'

    # Warmup
    print('\nWarmup...')
    for r in cosyvoice.inference_cross_lingual('Test warmup.', PROMPT_WAV, stream=False):
        pass

    # Run benchmarks
    results = {}

    # Test methods based on args
    if args.method in ['both', 'cross_lingual']:
        results['cross_lingual'] = run_benchmark(
            cosyvoice, sentences, method='cross_lingual',
            output_subdir='cross_lingual', output_base=output_base
        )

    if args.method in ['both', 'instruct2']:
        results['instruct2'] = run_benchmark(
            cosyvoice, sentences, method='instruct2',
            output_subdir='instruct2', output_base=output_base
        )

    # Summary comparison
    print('\n' + '='*70)
    print(f'COMPARISON SUMMARY [{backend}]')
    print('='*70)
    print(f'{"Method":<15} | {"Total Audio":>12} | {"Total Time":>12} | {"Avg RTF":>8} | {"Speed":>10}')
    print('-'*70)
    for method, data in results.items():
        speed = f'{1/data["avg_rtf"]:.1f}x'
        print(f'{method:<15} | {data["total_audio"]:>10.1f}s | {data["total_time"]:>10.1f}s | {data["avg_rtf"]:>8.3f} | {speed:>10}')
    print('='*70)

    print(f'\nOutput files saved to: {output_base}/')


if __name__ == '__main__':
    main()
