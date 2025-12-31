#!/usr/bin/env python3
"""Filter Czech dataset to remove samples containing written-out number words.

This script filters out samples from a CSV dataset that contain Czech number words
(e.g., 'jedna', 'dva', 'osmnáct', 'sto', 'tisíc'). This ensures the training data
doesn't contain samples that could confuse the TTS model with number vocabulary.

Micromamba env: cosyvoice

Usage:
    python filter_number_words.py \
        --input_csv /path/to/dataset.csv \
        --output_csv /path/to/filtered_dataset.csv \
        --delimiter '|'
"""
import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Czech number words to filter out
# Includes cardinals, ordinals fragments, and common number-related words
CZECH_NUMBER_WORDS = [
    # Basic numbers 1-10
    'jedna', 'jeden', 'jedné', 'jednoho', 'jedním', 'jednou',
    'dva', 'dvě', 'dvou', 'dvěma',
    'tři', 'třech', 'třem', 'třemi',
    'čtyři', 'čtyřech', 'čtyřem', 'čtyřmi',
    'pět', 'pěti',
    'šest', 'šesti',
    'sedm', 'sedmi',
    'osm', 'osmi',
    'devět', 'devíti',
    'deset', 'deseti', 'desíti',
    # 11-19
    'jedenáct', 'jedenácti',
    'dvanáct', 'dvanácti',
    'třináct', 'třinácti',
    'čtrnáct', 'čtrnácti',
    'patnáct', 'patnácti',
    'šestnáct', 'šestnácti',
    'sedmnáct', 'sedmnácti',
    'osmnáct', 'osmnácti',
    'devatenáct', 'devatenácti',
    # Tens
    'dvacet', 'dvaceti',
    'třicet', 'třiceti',
    'čtyřicet', 'čtyřiceti',
    'padesát', 'padesáti',
    'šedesát', 'šedesáti',
    'sedmdesát', 'sedmdesáti',
    'osmdesát', 'osmdesáti',
    'devadesát', 'devadesáti',
    # Hundreds and larger
    'sto', 'sta', 'set', 'stě', 'stem',
    'dvěstě', 'třista', 'čtyřista', 'pětset', 'šestset', 'sedmset', 'osmset', 'devětset',
    'tisíc', 'tisíce', 'tisících', 'tisíci',
    'milion', 'milionu', 'miliony', 'milionů',
    'miliarda', 'miliardy', 'miliard',
]

# Compile pattern for whole-word matching (case-insensitive)
NUMBER_WORD_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(word) for word in CZECH_NUMBER_WORDS) + r')\b',
    re.IGNORECASE
)


def contains_number_words(text: str) -> bool:
    """Check if text contains Czech number words.

    Args:
        text: Input text to check.

    Returns:
        True if text contains any Czech number words.
    """
    return bool(NUMBER_WORD_PATTERN.search(text))


def filter_csv(input_path: str, output_path: str, delimiter: str = '|',
               text_column: int = 1) -> dict:
    """Filter CSV file to remove rows with number words in text column.

    Args:
        input_path: Path to input CSV file.
        output_path: Path to output filtered CSV file.
        delimiter: CSV field delimiter.
        text_column: Index of text column (0-based).

    Returns:
        Statistics dict with counts.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'kept': 0,
        'filtered': 0,
        'filtered_samples': []
    }

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            # Check for header
            if line_num == 0 and 'audio_file' in line.lower():
                f_out.write(line + '\n')
                continue

            stats['total'] += 1
            parts = line.split(delimiter)

            if len(parts) <= text_column:
                # Invalid line, keep it
                f_out.write(line + '\n')
                stats['kept'] += 1
                continue

            text = parts[text_column]

            if contains_number_words(text):
                stats['filtered'] += 1
                if len(stats['filtered_samples']) < 10:
                    # Store first 10 filtered samples for logging
                    stats['filtered_samples'].append(text[:100])
            else:
                f_out.write(line + '\n')
                stats['kept'] += 1

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Filter Czech dataset to remove samples with number words'
    )
    parser.add_argument(
        '--input_csv', required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output_csv', required=True,
        help='Path to output filtered CSV file'
    )
    parser.add_argument(
        '--delimiter', default='|',
        help='CSV field delimiter (default: |)'
    )
    parser.add_argument(
        '--text_column', type=int, default=1,
        help='Index of text column (0-based, default: 1)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Czech Dataset Number Words Filter")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_csv}")
    logger.info(f"Output: {args.output_csv}")
    logger.info(f"Delimiter: '{args.delimiter}'")
    logger.info(f"Text column: {args.text_column}")
    logger.info("=" * 60)

    stats = filter_csv(
        args.input_csv,
        args.output_csv,
        args.delimiter,
        args.text_column
    )

    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  Total samples:    {stats['total']}")
    logger.info(f"  Kept samples:     {stats['kept']}")
    logger.info(f"  Filtered samples: {stats['filtered']}")
    logger.info(f"  Keep rate:        {100 * stats['kept'] / max(1, stats['total']):.2f}%")
    logger.info("=" * 60)

    if stats['filtered_samples']:
        logger.info("Sample filtered texts (first 10):")
        for i, sample in enumerate(stats['filtered_samples'], 1):
            logger.info(f"  {i}. {sample}...")

    logger.info(f"Filtered dataset saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
