#!/usr/bin/env python3
"""Q8_0 lossless coding: inspect, analyze, encode/decode one GGUF tensor.

CPU reference prototype; numpy + gguf required. Never changes the source model.
The .q8lut container stores ONE tensor, not a loadable GGUF model. The fixed-width
code stream uses code 0 for an exception, with original INT8 bytes in a side stream.
Q8_0 scale bytes are preserved exactly. This is not a CUDA or fused GEMM kernel.
Examples:
  python q8_codec_probe.py inspect /path/qwen-q8.gguf
  python q8_codec_probe.py analyze /path/qwen-q8.gguf --csv q8_report.csv
  python q8_codec_probe.py roundtrip /path/qwen-q8.gguf --out-dir results
  python q8_codec_probe.py selftest
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

MAGIC = b"Q8LUT01\n"
WIDTHS = (4, 5, 6, 7)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_q8(raw: bytes):
    if len(raw) % 34:
        raise ValueError("Q8_0 bytes must be a multiple of 34")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    return blocks[:, :2].copy().reshape(-1), blocks[:, 2:].copy().reshape(-1)


def counts_of(raw: bytes):
    scales, q = split_q8(raw)
    return scales, q, np.bincount(q, minlength=256)


def ordered_symbols(counts):
    order = np.argsort(-counts, kind="stable")
    return order[counts[order] > 0]


def pack_codes(codes, width):
    bits = (codes[:, None].astype(np.uint16) >> np.arange(width)) & 1
    return np.packbits(bits.astype(np.uint8).reshape(-1), bitorder="little").tobytes()


def unpack_codes(data: bytes, width: int, count: int):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = bits[:count * width].reshape(count, width).astype(np.uint16)
    return (bits * (1 << np.arange(width))).sum(axis=1).astype(np.uint8)


def encode(raw: bytes, width: int, name: str = "tensor") -> bytes:
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}")
    scales, q, counts = counts_of(raw)
    lut = ordered_symbols(counts)[:(1 << width) - 1].astype(np.uint8)
    mapping = np.zeros(256, dtype=np.uint8)
    mapping[lut] = np.arange(1, len(lut) + 1, dtype=np.uint8)
    codes = mapping[q]
    exceptions = q[codes == 0].tobytes()
    packed = pack_codes(codes, width)
    header = {
        "version": 1, "layout": "cpu-reference-v1", "tensor": name,
        "qtype": "Q8_0", "width": width, "n": len(q),
        "scale_bytes": len(scales), "lut_bytes": len(lut),
        "code_bytes": len(packed), "exception_bytes": len(exceptions),
        "original_bytes": len(raw), "original_sha256": digest(raw),
    }
    meta = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    return MAGIC + struct.pack("<I", len(meta)) + meta + scales.tobytes() + lut.tobytes() + packed + exceptions


def decode(blob: bytes):
    if blob[:8] != MAGIC or len(blob) < 12:
        raise ValueError("Not a Q8LUT01 tensor container")
    length = struct.unpack_from("<I", blob, 8)[0]
    header = json.loads(blob[12:12 + length])
    if header["version"] != 1 or header["qtype"] != "Q8_0":
        raise ValueError("Unsupported format")
    n, width = header["n"], header["width"]
    if width not in WIDTHS or n <= 0 or n % 32:
        raise ValueError("Invalid dimensions / code width")
    if header["scale_bytes"] != n // 32 * 2 or header["original_bytes"] != n // 32 * 34:
        raise ValueError("Invalid Q8_0 scale/raw lengths")
    if header["code_bytes"] != (n * width + 7) // 8:
        raise ValueError("Invalid fixed-width stream length")
    if not 1 <= header["lut_bytes"] <= (1 << width) - 1:
        raise ValueError("Invalid dictionary size")
    pos = 12 + length
    streams = []
    for key in ("scale_bytes", "lut_bytes", "code_bytes", "exception_bytes"):
        size = header[key]
        if size < 0 or pos + size > len(blob):
            raise ValueError("Truncated container")
        streams.append(blob[pos:pos + size])
        pos += size
    if pos != len(blob):
        raise ValueError("Unexpected trailing bytes")
    scales, lut, packed, exceptions = streams
    codes = unpack_codes(packed, width, n)
    if int(codes.max()) > len(lut):
        raise ValueError("Invalid dictionary index")
    escape = codes == 0
    if int(escape.sum()) != len(exceptions):
        raise ValueError("Invalid exception count")
    table = np.zeros(256, dtype=np.uint8)
    table[1:len(lut) + 1] = np.frombuffer(lut, dtype=np.uint8)
    q = table[codes]
    q[escape] = np.frombuffer(exceptions, dtype=np.uint8)
    restored = np.empty((n // 32, 34), dtype=np.uint8)
    restored[:, :2] = np.frombuffer(scales, dtype=np.uint8).reshape(-1, 2)
    restored[:, 2:] = q.reshape(-1, 32)
    raw = restored.tobytes()
    if digest(raw) != header["original_sha256"]:
        raise ValueError("SHA256 mismatch: decoded bytes are not the original tensor")
    return raw, header


def load_reader(path):
    from gguf import GGUFReader
    return GGUFReader(str(path), mode="r")


def q8_tensors(reader, name=None):
    from gguf import GGMLQuantizationType
    tensors = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.Q8_0]
    if name:
        tensors = [t for t in tensors if t.name == name]
    if not tensors:
        raise ValueError("No matching Q8_0 tensor. Run inspect and check the actual type/name.")
    return tensors


def raw_tensor(tensor):
    raw = np.ascontiguousarray(tensor.data).view(np.uint8).tobytes()
    if len(raw) != tensor.n_bytes or len(raw) % 34:
        raise ValueError("Unexpected Q8_0 data layout; do not dequantize and requantize")
    return raw


def inspect_model(args):
    reader = load_reader(args.model)
    print("Tensor types:", dict(Counter(t.tensor_type.name for t in reader.tensors)))
    for t in reader.tensors[:args.limit]:
        print(f"{t.name:50} {t.tensor_type.name:8} {tuple(t.shape)}  {t.n_bytes:,} bytes")
    candidates = q8_tensors(reader)
    chosen = next((t for t in candidates if t.name.startswith("blk.0.")), candidates[0])
    print("Suggested first tensor:", chosen.name)


def analyze_model(args):
    reader = load_reader(args.model)
    tensors = q8_tensors(reader, args.tensor)
    if not args.all and not args.tensor:
        tensors = [t for t in tensors if t.name.startswith("blk.0.")] or tensors[:1]
    rows = []
    for t in tensors:
        raw = raw_tensor(t)
        _, q, counts = counts_of(raw)
        prob = counts[counts > 0] / len(q)
        row = {"tensor": t.name, "weights": len(q), "original_q8_tensor_bytes": len(raw),
               "entropy_bits_per_integer": float(-(prob * np.log2(prob)).sum())}
        order = ordered_symbols(counts)
        for b in WIDTHS:
            hits = int(counts[order[:(1 << b) - 1]].sum())
            coverage = hits / len(q)
            row[f"top{(1 << b) - 1}_coverage"] = coverage
            row[f"b{b}_integer_payload_bpw"] = b + 8 * (1 - coverage)
        rows.append(row)
        best = min(WIDTHS, key=lambda b: row[f"b{b}_integer_payload_bpw"])
        print(f"{t.name}: H={row['entropy_bits_per_integer']:.3f}; "
              f"best b={best}, payload={row[f'b{best}_integer_payload_bpw']:.3f} bits/integer")
    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", output.resolve())
    print("Payload estimates exclude scales, LUT, container header, alignment and GPU block indexes.")
    print("Use roundtrip for actual single-tensor container size. These are NOT measured GPU latencies.")


def roundtrip_model(args):
    reader = load_reader(args.model)
    tensors = q8_tensors(reader, args.tensor)
    t = next((t for t in tensors if t.name.startswith("blk.0.")), tensors[0])
    raw = raw_tensor(t)
    trials = []
    for b in WIDTHS:
        start = time.perf_counter()
        blob = encode(raw, b, t.name)
        elapsed = time.perf_counter() - start
        trials.append((len(blob), b, elapsed))
        print(f"b={b}: {len(blob):,} bytes, saving={1-len(blob)/len(raw):.2%}, CPU encode={elapsed:.3f}s")
    _, b, _ = min(trials)
    blob = encode(raw, b, t.name)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = t.name.replace("/", "_")
    coded = output / f"{stem}.q8lut"
    restored_file = output / f"{stem}.restored.q8raw"
    if coded.exists() or restored_file.exists():
        raise FileExistsError("Outputs already exist; choose a new --out-dir")
    with coded.open("xb") as f:
        f.write(blob)
    start = time.perf_counter()
    restored, _ = decode(coded.read_bytes())
    decode_time = time.perf_counter() - start
    if restored != raw:
        raise RuntimeError("Roundtrip failed")
    with restored_file.open("xb") as f:
        f.write(restored)
    print("Tensor:", t.name)
    print("SHA256 / all scale and INT8 bytes: PASS", digest(raw))
    print(f"Selected b={b}; actual single-tensor saving={1-len(blob)/len(raw):.2%}")
    print(f"CPU file-read + decode + integrity check: {decode_time:.3f}s")
    print("Encoded:", coded.resolve())
    print("Restored raw tensor:", restored_file.resolve())
    print("These files are ONE tensor; neither is a model llama.cpp can load.")
    if len(blob) >= len(raw):
        print("No compression benefit: a future full-model container should retain this tensor raw.")


def selftest(_args):
    rng = np.random.default_rng(17)
    cases = [np.zeros(32, np.uint8), np.arange(256, dtype=np.uint8),
             rng.integers(0, 256, size=544, dtype=np.uint8),
             rng.choice(np.array([0, 1, 255, 2], np.uint8), size=4096)]
    passed = 0
    for q in cases:
        scales = rng.integers(0, 256, size=(len(q) // 32, 2), dtype=np.uint8)
        raw = np.concatenate([scales, q.reshape(-1, 32)], axis=1).tobytes()
        for b in WIDTHS:
            blob = encode(raw, b)
            restored, _ = decode(blob)
            assert restored == raw
            passed += 1
            broken = bytearray(blob)
            broken[-1] ^= 1
            try:
                decode(bytes(broken))
            except (ValueError, IndexError):
                pass
            else:
                raise AssertionError("Corruption was not detected")
    print(f"PASS: {passed} byte-exact roundtrips plus corruption checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect"); p.add_argument("model"); p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=inspect_model)
    p = sub.add_parser("analyze"); p.add_argument("model"); p.add_argument("--tensor")
    p.add_argument("--all", action="store_true"); p.add_argument("--csv", default="q8_report.csv")
    p.set_defaults(func=analyze_model)
    p = sub.add_parser("roundtrip"); p.add_argument("model"); p.add_argument("--tensor")
    p.add_argument("--out-dir", default="q8_roundtrip"); p.set_defaults(func=roundtrip_model)
    p = sub.add_parser("selftest"); p.set_defaults(func=selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
