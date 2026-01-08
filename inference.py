#!/usr/bin/env python3
"""CosyVoice inference CLI - Text-to-speech synthesis.

Micromamba env: cosyvoice (or cosyvoice_vllm for vLLM)

Usage:
    # Single sentence
    python inference.py --text "Hello world" --prompt-wav ref.wav --output out.wav

    # Multiple sentences from file (one per line)
    python inference.py --file sentences.txt --prompt-wav ref.wav --output-dir ./outputs

    # With vLLM acceleration and concurrent batch processing (2x throughput)
    python inference.py --file sentences.txt --prompt-wav ref.wav --vllm \
        --workers 6 --temperature 0.8 --output-dir ./outputs

Arguments:
    --text TEXT             Single sentence to synthesize
    --file FILE             Text file with sentences (one per line)
    --prompt-wav PATH       Reference audio (required, max 30s, recommended 3-10s)
    --output PATH           Output wav file (for single text, default: output.wav)
    --output-dir DIR        Output directory (for file input, default: ./outputs)
    --model-dir PATH        Model directory
    --method METHOD         instruct2 or cross_lingual (default: instruct2)
    --instruct TEXT         Instruction text (default: "You are a helpful assistant.")
    --vllm                  Enable vLLM backend (5x speedup)
    --trt                   Enable TensorRT
    --temperature FLOAT     LLM temperature (default: 1.0)
    --top-k INT             Top-K sampling (default: 25)
    --top-p FLOAT           Nucleus sampling (default: 1.0)
    --speed FLOAT           Speech speed multiplier (default: 1.0)
    --workers INT           Concurrent workers (default: 1, recommended: 6 for vLLM)

Performance (RTX 3090, vLLM):
    - Sequential (workers=1): RTF ~0.19 (5.2x real-time)
    - Concurrent (workers=6): RTF ~0.10 (10x real-time, 2x throughput)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

import logging
logging.basicConfig(level=logging.ERROR)

import torchaudio

# Default paths
DEFAULT_MODEL_DIR = '/mnt/8TB/AUDIO/TEXT_TO_SPEECH/MODELS/CosyVoice3-Czech-HobbitDeep'
DEFAULT_INSTRUCT = 'You are a helpful assistant.'


def format_instruct_text(text: str) -> str:
    """Format instruction text with endofprompt marker."""
    text = (text or '').strip()
    if not text:
        return '<|endofprompt|>'
    if '<|endofprompt|>' not in text:
        text = f'{text}<|endofprompt|>'
    return text


def load_sentences(file_path: str) -> list[str]:
    """Load sentences from text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def run_inference(cosyvoice, text: str, prompt_wav: str, method: str,
                  instruct: str, speed: float, sampling: int,
                  temperature: float, top_p: float):
    """Run single inference and return audio tensor."""
    if method == 'cross_lingual':
        for r in cosyvoice.inference_cross_lingual(
            text,
            prompt_wav,
            stream=False,
            speed=speed
        ):
            return r['tts_speech']
    elif method == 'instruct2':
        instruct_text = format_instruct_text(instruct)
        for r in cosyvoice.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_wav=prompt_wav,
            stream=False,
            speed=speed,
            text_frontend=False,
            sampling=sampling,
            temperature=temperature,
            top_p=top_p
        ):
            return r['tts_speech']
    else:
        raise ValueError(f'Unknown method: {method}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CosyVoice TTS Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single sentence with vLLM
  python inference.py --text "Hello world" --prompt-wav ref.wav --vllm

  # Multiple sentences from file
  python inference.py --file sentences.txt --prompt-wav ref.wav --output-dir ./out

  # With generation parameters
  python inference.py --text "Test" --prompt-wav ref.wav --temperature 0.8
'''
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Single sentence to synthesize')
    input_group.add_argument('--file', type=str, help='Text file with sentences (one per line)')

    # Reference audio (required)
    parser.add_argument('--prompt-wav', type=str, required=True,
                        help='Reference audio path (max 30s, recommended 3-10s)')

    # Output options
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output wav file (for single text, default: output.wav)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory (for file input, default: ./outputs)')

    # Model options
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Model directory path')
    parser.add_argument('--method', type=str, choices=['instruct2', 'cross_lingual'],
                        default='instruct2', help='Inference method (default: instruct2)')
    parser.add_argument('--instruct', type=str, default=DEFAULT_INSTRUCT,
                        help='Instruction text for instruct2 method')

    # Acceleration options
    parser.add_argument('--vllm', action='store_true', help='Enable vLLM backend (5x speedup)')
    parser.add_argument('--trt', action='store_true', help='Enable TensorRT')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='LLM temperature (default: 1.0)')
    parser.add_argument('--top-k', type=int, default=25,
                        help='Top-K sampling (default: 25)')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Nucleus sampling (default: 1.0)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed multiplier (default: 1.0)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent workers (default: 1, recommended: 6 for vLLM)')

    args = parser.parse_args()

    # Validate reference audio exists
    if not os.path.exists(args.prompt_wav):
        print(f'Error: Reference audio not found: {args.prompt_wav}')
        sys.exit(1)

    # Register vLLM model if needed
    if args.vllm:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel

    # Determine backend name
    if args.trt and args.vllm:
        backend = 'TRT+vLLM'
    elif args.trt:
        backend = 'TensorRT'
    elif args.vllm:
        backend = 'vLLM'
    else:
        backend = 'PyTorch'

    print('=' * 60)
    print(f'CosyVoice Inference [{backend}]')
    print('=' * 60)

    # Load sentences
    if args.text:
        sentences = [args.text]
        print(f'\nText: {args.text[:60]}{"..." if len(args.text) > 60 else ""}')
    else:
        sentences = load_sentences(args.file)
        print(f'\nLoaded {len(sentences)} sentences from {args.file}')

    # Load model
    print(f'\nLoading model from {args.model_dir}...')
    start = time.time()
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.trt,
        load_vllm=args.vllm,
        fp16=False
    )
    print(f'Model loaded in {time.time() - start:.1f}s')

    # Warmup
    print('\nWarmup...')
    for r in cosyvoice.inference_cross_lingual('Test.', args.prompt_wav, stream=False):
        pass

    # Run inference
    print(f'\nMethod: {args.method}')
    print(f'Reference: {args.prompt_wav}')
    if args.method == 'instruct2' and (args.temperature != 1.0 or args.top_k != 25 or args.top_p != 1.0):
        print(f'Params: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}')
    if args.workers > 1:
        print(f'Workers: {args.workers} (concurrent)')
    print('-' * 60)

    # Prepare output directory
    if not args.text:
        os.makedirs(args.output_dir, exist_ok=True)

    total_audio = 0
    total_time = 0
    start_all = time.time()

    if args.workers > 1 and len(sentences) > 1:
        # Concurrent processing
        def infer_one(idx_sentence):
            idx, sentence = idx_sentence
            t0 = time.time()
            audio = run_inference(
                cosyvoice, sentence, args.prompt_wav, args.method,
                args.instruct, args.speed, args.top_k, args.temperature, args.top_p
            )
            return idx, sentence, audio, time.time() - t0

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(infer_one, (i, s)): i
                       for i, s in enumerate(sentences)}
            for future in as_completed(futures):
                idx, sentence, audio, infer_time = future.result()
                audio_len = audio.shape[1] / cosyvoice.sample_rate
                results.append((idx, sentence, audio, audio_len, infer_time))

        # Save in order and report
        results.sort(key=lambda x: x[0])
        for idx, sentence, audio, audio_len, infer_time in results:
            if args.text:
                output_path = args.output
            else:
                output_path = os.path.join(args.output_dir, f'{idx+1:03d}.wav')
            torchaudio.save(output_path, audio.cpu(), cosyvoice.sample_rate)
            display = sentence[:50] + '...' if len(sentence) > 50 else sentence
            rtf = infer_time / audio_len
            print(f'{idx+1:>3} | {audio_len:>5.1f}s | {infer_time:>5.2f}s | RTF {rtf:.3f} | {display}')
            print(f'    -> {output_path}')
            total_audio += audio_len

        total_time = time.time() - start_all
    else:
        # Sequential processing
        for i, sentence in enumerate(sentences):
            start = time.time()

            audio = run_inference(
                cosyvoice, sentence, args.prompt_wav, args.method,
                args.instruct, args.speed, args.top_k, args.temperature, args.top_p
            )

            infer_time = time.time() - start
            audio_len = audio.shape[1] / cosyvoice.sample_rate
            rtf = infer_time / audio_len

            # Determine output path
            if args.text:
                output_path = args.output
            else:
                output_path = os.path.join(args.output_dir, f'{i+1:03d}.wav')

            # Save audio
            torchaudio.save(output_path, audio.cpu(), cosyvoice.sample_rate)

            # Display progress
            display = sentence[:50] + '...' if len(sentence) > 50 else sentence
            print(f'{i+1:>3} | {audio_len:>5.1f}s | {infer_time:>5.2f}s | RTF {rtf:.3f} | {display}')
            print(f'    -> {output_path}')

            total_audio += audio_len
            total_time += infer_time

    # Summary
    print('-' * 60)
    avg_rtf = total_time / total_audio if total_audio > 0 else 0
    print(f'Total: {total_audio:.1f}s audio in {total_time:.1f}s | RTF {avg_rtf:.3f} ({1/avg_rtf:.1f}x real-time)')


if __name__ == '__main__':
    main()
