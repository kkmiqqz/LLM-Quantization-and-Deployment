#!/usr/bin/env python3
"""Estimate static Huffman coding lengths for existing GGUF Q8_0 integers.

CPU-only length probe: no bitstream is written and no decoding is timed.
Only numpy and gguf are needed; use your existing working environment.
One Huffman codebook per tensor, not per 32-value quantization block.
Q8_0 FP16 scales are kept byte-for-byte (2 bytes per 32 integers).

Examples:
  python q8_huffman_probe.py /path/qwen-q8.gguf --csv q8_huffman_report.csv
  python q8_huffman_probe.py /path/qwen-q8.gguf --all --csv q8_huffman_all.csv
  python q8_huffman_probe.py /path/qwen-q8.gguf --stream-weights 4096 --csv streams.csv
  python q8_huffman_probe.py --self-test

Reporting model, NOT an implemented compressed file format:
  payload = sum(count[s] * Huffman_length[s]), byte-padded per stream;
  codebook = 256 bytes (canonical code lengths, one uint8 per byte symbol);
  header = 32 bytes per tensor (four uint64 values for sizes/counts);
  scales = original uncompressed scale bytes;
  optional independent streams = 16 bytes/stream for offset/bit-count/length.
Global container directories, extra alignment and checksums are excluded.
No length-limited GPU codebook or fused computation is implemented.

References:
  https://theory.stanford.edu/~blynn/pr/symbol.html
  https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py
"""
from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

import numpy as np

CODEBOOK_BYTES = 256
HEADER_BYTES = 32
STREAM_INDEX_BYTES = 16
HISTOGRAM_BLOCKS = 32768


def huffman_lengths(counts):
    """Binary Huffman lengths for up to 256 symbols; singleton uses 1 bit."""
    counts = np.asarray(counts)
    if counts.shape != (256,) or np.any(counts < 0):
        raise ValueError("Expected 256 nonnegative symbol counts")
    heap = [(int(c), s, s) for s, c in enumerate(counts) if c > 0]
    lengths = np.zeros(256, dtype=np.uint16)
    if not heap:
        return lengths
    if len(heap) == 1:
        lengths[heap[0][2]] = 1
        return lengths
    heapq.heapify(heap)
    serial = 256  # Unique tie-breaker; Python never compares different node types.
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], serial, (a[2], b[2])))
        serial += 1
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = depth
        else:
            stack.append((node[0], depth + 1))
            stack.append((node[1], depth + 1))
    return lengths


def summarize_counts(counts):
    n = sum(int(c) for c in counts)
    if n <= 0:
        raise ValueError("Empty tensor")
    lengths = huffman_lengths(counts)
    bits = sum(int(c) * int(l) for c, l in zip(counts, lengths))
    prob = np.asarray(counts[counts > 0], dtype=np.float64) / n
    entropy = float(-(prob * np.log2(prob)).sum())
    mean_length = bits / n
    if not entropy - 1e-10 <= mean_length <= 8 + 1e-10:
        raise RuntimeError("Invalid Huffman average length")
    sorted_counts = sorted((int(c) for c in counts), reverse=True)
    candidates = [(b + 8 * (1 - sum(sorted_counts[:(1 << b) - 1]) / n), b)
                  for b in (4, 5, 6, 7)]
    fixed_bpw, fixed_width = min(candidates)
    return n, lengths, bits, entropy, mean_length, fixed_bpw, fixed_width


def tensor_blocks(tensor):
    # GGUFReader maps the file read-only; process small slices, not a model copy.
    data = np.asarray(tensor.data)
    if data.dtype != np.uint8 or not data.flags.c_contiguous:
        raise ValueError(f"Unexpected Q8_0 storage for {tensor.name}")
    if data.nbytes != tensor.n_bytes or data.nbytes % 34:
        raise ValueError(f"Unexpected Q8_0 byte count for {tensor.name}")
    blocks = data.reshape(-1, 34)
    if len(blocks) * 32 != tensor.n_elements:
        raise ValueError(f"Q8_0 element/block mismatch for {tensor.name}")
    return blocks


def histogram(blocks):
    counts = np.zeros(256, dtype=np.int64)
    for start in range(0, len(blocks), HISTOGRAM_BLOCKS):
        integers = blocks[start:start + HISTOGRAM_BLOCKS, 2:].reshape(-1)
        counts += np.bincount(integers, minlength=256)
    return counts


def profile(blocks, name, stream_weights=0):
    counts = histogram(blocks)
    n, lengths, bits, entropy, mean_length, fixed_bpw, fixed_width = summarize_counts(counts)
    streams = 1
    payload_bytes = (bits + 7) // 8
    index_bytes = 0
    if stream_weights:
        block_step = stream_weights // 32
        streams = (len(blocks) + block_step - 1) // block_step
        payload_bytes = 0
        # Shared tensor codebook; independent starts add padding/index overhead.
        for start in range(0, len(blocks), block_step):
            q = blocks[start:start + block_step, 2:].reshape(-1)
            stream_bits = int(lengths[q].sum(dtype=np.uint64))
            payload_bytes += (stream_bits + 7) // 8
        index_bytes = streams * STREAM_INDEX_BYTES
    scales = len(blocks) * 2
    original = len(blocks) * 34
    estimated = payload_bytes + scales + CODEBOOK_BYTES + HEADER_BYTES + index_bytes
    used_lengths = lengths[counts > 0]
    return {
        "tensor": name,
        "weights": n,
        "used_symbols": int((counts > 0).sum()),
        "entropy_bpw": entropy,
        "huffman_bpw": mean_length,
        "huffman_gap_bpw": mean_length - entropy,
        "min_code_length": int(used_lengths.min()),
        "max_code_length": int(used_lengths.max()),
        "best_fixed_escape_width": fixed_width,
        "best_fixed_escape_bpw": fixed_bpw,
        "original_integer_bytes": n,
        "original_scale_bytes": scales,
        "original_q8_bytes": original,
        "payload_bits": bits,
        "payload_bytes": payload_bytes,
        "padding_bits": payload_bytes * 8 - bits,
        "codebook_bytes": CODEBOOK_BYTES,
        "assumed_header_bytes": HEADER_BYTES,
        "stream_weights": stream_weights,
        "streams": streams,
        "assumed_stream_index_bytes": index_bytes,
        "estimated_q8_bytes": estimated,
        "estimated_q8_bpw": estimated * 8 / n,
        "integer_payload_saving_percent": 100 * (1 - payload_bytes / n),
        "estimated_q8_saving_percent": 100 * (1 - estimated / original),
    }


def run(args):
    from gguf import GGUFReader, GGMLQuantizationType
    model = Path(args.model).expanduser()
    if not model.is_file():
        raise FileNotFoundError(f"Model does not exist: {model}")
    reader = GGUFReader(str(model), mode="r")
    all_q8 = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.Q8_0]
    if args.tensor:
        tensors = [t for t in all_q8 if t.name == args.tensor]
    elif args.all:
        tensors = all_q8
    else:
        tensors = [t for t in all_q8 if t.name.startswith("blk.0.")]
        if not tensors:
            raise ValueError("No blk.0 Q8_0 tensors. Use --tensor NAME or --all.")
    if not tensors:
        raise ValueError("No matching Q8_0 tensor")
    output = Path(args.csv).expanduser()
    if output.resolve() == model.resolve():
        raise ValueError("Output must not be the source model")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Selected {len(tensors)} / {len(all_q8)} Q8_0 tensors; model opened read-only.")
    print("Codebook: 256 B/tensor; assumed header: 32 B/tensor; scales unchanged.")
    if args.stream_weights:
        print(f"Independent streams: up to {args.stream_weights} weights; assumed index: 16 B/stream.")
    else:
        print("One continuous stream per tensor; GPU checkpoints/indexes NOT included.")
    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = None
        for tensor in tensors:
            row = profile(tensor_blocks(tensor), tensor.name, args.stream_weights)
            results.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"{tensor.name}: H={row['entropy_bpw']:.4f}, "
                  f"Huffman={row['huffman_bpw']:.4f}, "
                  f"fixed+escape={row['best_fixed_escape_bpw']:.4f}, "
                  f"Q8 saving(est.)={row['estimated_q8_saving_percent']:.2f}%, "
                  f"max_code={row['max_code_length']}", flush=True)
    n = sum(r["weights"] for r in results)
    original = sum(r["original_q8_bytes"] for r in results)
    estimated = sum(r["estimated_q8_bytes"] for r in results)
    entropy = sum(r["entropy_bpw"] * r["weights"] for r in results) / n
    mean_length = sum(r["payload_bits"] for r in results) / n
    print("\nSUMMARY (selected Q8_0 tensors only; not a compressed GGUF file)")
    print(f"Weights={n:,}; weighted H={entropy:.4f}; Huffman={mean_length:.4f} bits/integer")
    print(f"Original Q8_0 bytes={original:,}; modeled bytes={estimated:,}")
    print(f"Aggregate modeled Q8 saving={100 * (1-estimated/original):.2f}%")
    print(f"Tensors with positive modeled saving={sum(r['estimated_q8_bytes'] < r['original_q8_bytes'] for r in results)}/{len(results)}")
    print("CSV:", output.resolve())
    print("Exact weighted code lengths; modeled metadata. No bitstream, decode, GPU timing or model-size claim.")


def self_test():
    def make(values):
        counts = np.zeros(256, dtype=np.int64)
        counts[:len(values)] = values
        return counts

    classic = make([5, 9, 12, 13, 16, 45])
    _, lengths, bits, _, _, _, _ = summarize_counts(classic)
    assert list(lengths[:6]) == [4, 4, 3, 3, 3, 1] and bits == 224
    assert summarize_counts(make([4, 2, 1, 1]))[2] == 14
    assert summarize_counts(make([10]))[2] == 10  # Explicit singleton convention.
    assert not huffman_lengths(make([])).any()
    assert summarize_counts(make([1] * 256))[2] == 2048
    # Verify prefix-tree capacity exactly, not with floating point Kraft sums.
    rng = np.random.default_rng(24)
    for _ in range(50):
        counts = rng.integers(0, 1000, size=256, dtype=np.int64)
        _, lengths, _, h, mean, _, _ = summarize_counts(counts)
        active = [int(l) for l in lengths[counts > 0]]
        maximum = max(active)
        assert sum(1 << (maximum - l) for l in active) == 1 << maximum
        assert h - 1e-10 <= mean < h + 1 and mean <= 8
    # Scale bytes must never enter the symbol histogram.
    blocks = np.full((64, 34), 255, dtype=np.uint8)
    blocks[:, 2:] = 0
    assert int(histogram(blocks)[0]) == 2048 and int(histogram(blocks)[255]) == 0
    row = profile(blocks, "zero")
    assert row["payload_bits"] == 2048
    assert row["original_q8_bytes"] == 64 * 34
    assert row["estimated_q8_bytes"] == 256 + 128 + 256 + 32
    chunked = profile(blocks, "zero", 256)
    assert chunked["streams"] == 8 and chunked["assumed_stream_index_bytes"] == 128
    assert chunked["payload_bits"] == row["payload_bits"]
    # Byte padding on tiny streams (256 symbols is not assumed).
    mixed = blocks.copy()
    mixed[:, 2] = 1
    a = profile(mixed, "mixed")
    b = profile(mixed, "mixed", 32)
    assert b["payload_bytes"] >= a["payload_bytes"] and 0 <= b["padding_bits"] < 8 * b["streams"]
    print("PASS: known Huffman examples, uniform/singleton cases, entropy and Kraft checks, Q8_0 scale separation and stream accounting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="Path to existing Q8 GGUF")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="Analyze every Q8_0 tensor")
    scope.add_argument("--tensor", help="Analyze exactly one named Q8_0 tensor")
    parser.add_argument("--csv", default="q8_huffman_report.csv", help="New CSV output; never overwrites")
    parser.add_argument("--stream-weights", type=int, default=0,
                        help="Optional independent stream size, a positive multiple of 32; 0=one tensor stream")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model:
        parser.error("Provide a model path, or use --self-test")
    if args.stream_weights < 0 or args.stream_weights % 32:
        parser.error("--stream-weights must be 0 or a positive multiple of 32")
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
