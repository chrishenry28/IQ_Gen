#!/usr/bin/env python3
"""
fhss_hopset_gen.py - Generate FHSS hop frequency lists.

Creates a list of hop frequencies centered around baseband (0 Hz),
either evenly spaced by a given spacing, or evenly distributed in a
span with a specific number of channels.

Output is a text file containing a comma-separated frequency list,
e.g.:

-9.5e6,-9.0e6,-8.5e6,-8.0e6,...

Typical use case: generate a hop set for FHSS CW in a 20 MHz-wide band
(fs = 20e6, Nyquist ±10 MHz).

Examples
--------
# 1) 20 MHz span (±10 MHz), spacing 0.5 MHz, excluding DC:
python fhss_hopset_gen.py --span 10e6 --spacing 0.5e6 \
    --exclude-zero -o hopset_20M_500k.txt

# 2) Same as above, but shuffle order (random hop order):
python fhss_hopset_gen.py --span 10e6 --spacing 0.5e6 \
    --exclude-zero --shuffle --seed 123 \
    -o hopset_20M_500k_shuffled.txt

# 3) Use 40 total channels in ±9.5 MHz, automatically spaced:
python fhss_hopset_gen.py --span 9.5e6 --num 40 \
    --exclude-zero -o hopset_19M_40ch.txt
"""

import argparse
import math
import random
from typing import List


def format_freq(freq_hz: float, style: str = "e6") -> str:
    """
    Format frequency as string.

    style = 'e6'  -> f/1e6 with 'e6' suffix (e.g. -9.5e6)
          = 'hz'  -> plain Hz (e.g. -9500000.0)
    """
    if style == "hz":
        return f"{freq_hz:.6f}"
    elif style == "e6":
        return f"{freq_hz / 1e6:.6g}e6"
    else:
        raise ValueError(f"Unknown format style: {style}")


def gen_hopset_with_spacing(span: float,
                            spacing: float,
                            include_zero: bool) -> List[float]:
    """
    Generate hop frequencies with a fixed spacing.

    span     : maximum |freq| in Hz (half-span), e.g. 10e6 for ±10 MHz
    spacing  : spacing between hop channels (Hz), e.g. 0.5e6
    include_zero: whether to include 0 Hz in the hop set
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    # Number of channels on one side
    n_side = int(span // spacing)
    if n_side < 1:
        raise ValueError("span is too small for the given spacing.")

    positives = [k * spacing for k in range(1, n_side + 1)]
    negatives = [-f for f in positives]

    freqs = sorted(negatives + positives)
    if include_zero:
        freqs.insert(len(negatives), 0.0)

    return freqs


def gen_hopset_with_num(span: float,
                        num: int,
                        include_zero: bool) -> List[float]:
    """
    Generate hop frequencies with a given total number of channels,
    distributed between -span and +span.

    span     : maximum |freq| in Hz (half-span), e.g. 10e6 for ±10 MHz
    num      : total number of frequencies (including negative and positive)
    include_zero: whether to include 0 Hz in the hop set

    Note: This uses a linspace-like approach, then optionally removes or
    inserts 0 Hz to match include_zero.
    """
    if num < 1:
        raise ValueError("num must be >= 1")

    # If we want to include zero, it's easiest to create a symmetric
    # set with num points that spans [-span, +span].
    if include_zero:
        # evenly spaced from -span to +span, including endpoints
        step = (2 * span) / (num - 1) if num > 1 else 0
        freqs = [-span + i * step for i in range(num)]
        # Ensure exact 0 in the list if num is odd
        if num % 2 == 1:
            mid_idx = num // 2
            freqs[mid_idx] = 0.0
        return freqs

    # Excluding zero: create num+1 points, drop the one closest to zero
    full_num = num + 1
    step = (2 * span) / (full_num - 1) if full_num > 1 else 0
    freqs_full = [-span + i * step for i in range(full_num)]

    # Drop the value closest to 0
    idx_zero = min(range(full_num), key=lambda i: abs(freqs_full[i]))
    freqs = [f for i, f in enumerate(freqs_full) if i != idx_zero]
    return freqs


def main():
    parser = argparse.ArgumentParser(
        description="Generate FHSS hop frequency lists centered around baseband."
    )
    parser.add_argument("--span", type=float, required=True,
                        help="Maximum |f| in Hz (half-span). e.g., 10e6 for ±10 MHz.")
    parser.add_argument("--spacing", type=float, default=None,
                        help="Channel spacing in Hz. If provided, overrides --num.")
    parser.add_argument("--num", type=int, default=40,
                        help="Total number of hop frequencies (used if --spacing not given).")
    parser.add_argument("--include-zero", action="store_true",
                        help="Include 0 Hz as a hop frequency.")
    parser.add_argument("--exclude-zero", action="store_true",
                        help="Exclude 0 Hz (default).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle (randomize) hop order.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling (optional).")
    parser.add_argument("--fmt", type=str, default="e6",
                        choices=["e6", "hz"],
                        help="Output format: 'e6' (e.g. -9.5e6) or 'hz' (plain Hz).")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="Output text file (comma-separated list).")

    args = parser.parse_args()

    if args.include_zero and args.exclude_zero:
        raise ValueError("Cannot use both --include-zero and --exclude-zero.")
    include_zero = args.include_zero and not args.exclude_zero

    span = args.span
    if span <= 0:
        raise ValueError("--span must be > 0")

    # Generate frequencies
    if args.spacing is not None:
        freqs = gen_hopset_with_spacing(span, args.spacing, include_zero)
    else:
        freqs = gen_hopset_with_num(span, args.num, include_zero)

    # Shuffle if requested
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(freqs)

    # Format frequencies
    freq_strs = [format_freq(f, style=args.fmt) for f in freqs]

    # Write to file as a single comma-separated line
    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write(",".join(freq_strs) + "\n")

    print(f"Wrote {len(freqs)} hop frequencies to '{args.outfile}'.")
    print("Example preview:")
    preview = ",".join(freq_strs[:10])
    print("  " + preview + ("..." if len(freqs) > 10 else ""))


if __name__ == "__main__":
    main()
