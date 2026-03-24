import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from functools import lru_cache
from typing import Iterable, Iterator

import numpy as np
import regex as re

from cs336_basics.train_bpe import PAT

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TINYSTORIES_TRAIN = DEFAULT_DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_TINYSTORIES_VALID = DEFAULT_DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_OWT_TRAIN = DEFAULT_DATA_DIR / "owt_train.txt"
DEFAULT_OWT_VALID = DEFAULT_DATA_DIR / "owt_valid.txt"
DEFAULT_TINYSTORIES_VOCAB = DEFAULT_DATA_DIR / "bpe_vocab.json"
DEFAULT_TINYSTORIES_MERGES = DEFAULT_DATA_DIR / "bpe_merges.json"
DEFAULT_OWT_VOCAB = DEFAULT_DATA_DIR / "owt_bpe_vocab.json"
DEFAULT_OWT_MERGES = DEFAULT_DATA_DIR / "owt_bpe_merges.json"


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens: list[str] | None = None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        self._pretoken_re = re.compile(PAT)
        self._special_tokens_set = frozenset(self.special_tokens)
        if self.special_tokens:
            special_tokens_pattern = "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_split_re = re.compile(f"({special_tokens_pattern})")
        else:
            self._special_split_re = None

    @classmethod
    def _deserialize_token_bytes(cls, token_str: str, token_id: int | None = None) -> bytes:
        # Our own trained tokenizers should preserve the base byte vocabulary exactly.
        if token_id is not None and token_id < 256:
            return bytes([token_id])

        # Lossless latin-1 decoding is preferred for our serialized tokenizers.
        try:
            return token_str.encode("latin-1")
        except UnicodeEncodeError:
            # GPT-2 style byte-to-unicode strings need UTF-8 encoding instead.
            return token_str.encode("utf-8")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        vocab = {
            int(token_id): cls._deserialize_token_bytes(token_str, int(token_id))
            for token_id, token_str in raw_vocab.items()
        }
        merges = [
            (cls._deserialize_token_bytes(left), cls._deserialize_token_bytes(right))
            for left, right in raw_merges
        ]

        if special_tokens:
            existing = set(vocab.values())
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing:
                    vocab[len(vocab)] = token_bytes
                    existing.add(token_bytes)

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[bytes]:
        pretokens: list[bytes] = []

        if self._special_split_re is None:
            segments = [text]
        else:
            segments = self._special_split_re.split(text)

        for piece in segments:
            if not piece:
                continue

            if piece in self._special_tokens_set:
                pretokens.append(piece.encode("utf-8"))
                continue

            for match in self._pretoken_re.finditer(piece):
                pretokens.append(match.group(0).encode("utf-8"))

        return pretokens

    @lru_cache(maxsize=1_000_000)
    def _encode_pre_token_cached(self, pretoken_bytes: bytes) -> tuple[int, ...]:
        direct_token_id = self.bytes_to_id.get(pretoken_bytes)
        if direct_token_id is not None:
            return (direct_token_id,)

        current_tokens = [pretoken_bytes[i:i+1] for i in range(len(pretoken_bytes))]

        while len(current_tokens) > 1:
            best_index = None
            best_rank = None

            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break

            merged_token = current_tokens[best_index] + current_tokens[best_index + 1]
            current_tokens[best_index : best_index + 2] = [merged_token]

        return tuple(self.bytes_to_id[token] for token in current_tokens)

    def _encode_pre_token(self, pretoken_bytes: bytes) -> list[int]:
        return list(self._encode_pre_token_cached(pretoken_bytes))

    def encode(self, text: str) -> list[int]:
        result_ids: list[int] = []
        seq = self._pre_tokenize(text)

        for pretoken_bytes in seq:
            result_ids.extend(self._encode_pre_token(pretoken_bytes))

        return result_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[token_id] for token_id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")


def iter_documents(
    input_path: str | Path,
    delimiter: str = DEFAULT_SPECIAL_TOKENS[0],
    chunk_chars: int = 1 << 20,
) -> Iterator[str]:
    remainder = ""

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if chunk == "":
                tail = remainder.strip()
                if tail:
                    yield tail
                return

            pieces = (remainder + chunk).split(delimiter)
            remainder = pieces.pop()

            for piece in pieces:
                document = piece.strip()
                if document:
                    yield document


def sample_documents(
    input_path: str | Path,
    num_documents: int,
    seed: int,
    delimiter: str = DEFAULT_SPECIAL_TOKENS[0],
) -> list[str]:
    rng = random.Random(seed)
    samples: list[str] = []

    for seen, document in enumerate(iter_documents(input_path, delimiter=delimiter), start=1):
        if len(samples) < num_documents:
            samples.append(document)
            continue

        sample_index = rng.randrange(seen)
        if sample_index < num_documents:
            samples[sample_index] = document

    return samples


def take_documents_by_bytes(
    input_path: str | Path,
    target_bytes: int,
    delimiter: str = DEFAULT_SPECIAL_TOKENS[0],
) -> list[str]:
    documents: list[str] = []
    bytes_so_far = 0

    for document in iter_documents(input_path, delimiter=delimiter):
        documents.append(document)
        bytes_so_far += len(document.encode("utf-8"))
        if bytes_so_far >= target_bytes:
            break

    return documents


def count_text_bytes(texts: Iterable[str]) -> int:
    return sum(len(text.encode("utf-8")) for text in texts)


def compression_ratio_bytes_per_token(tokenizer: BPETokenizer, documents: Iterable[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))

    if total_tokens == 0:
        return math.inf, total_bytes, total_tokens

    return total_bytes / total_tokens, total_bytes, total_tokens


def benchmark_tokenizer_throughput(
    tokenizer: BPETokenizer,
    documents: Iterable[str],
) -> tuple[float, int, int, float]:
    total_bytes = 0
    total_tokens = 0
    started_at = time.perf_counter()

    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    return total_bytes / elapsed_seconds, total_bytes, total_tokens, elapsed_seconds


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def build_default_tokenizer(kind: str) -> BPETokenizer:
    if kind == "tinystories":
        return BPETokenizer.from_files(
            DEFAULT_TINYSTORIES_VOCAB,
            DEFAULT_TINYSTORIES_MERGES,
            DEFAULT_SPECIAL_TOKENS,
        )
    if kind == "owt":
        return BPETokenizer.from_files(
            DEFAULT_OWT_VOCAB,
            DEFAULT_OWT_MERGES,
            DEFAULT_SPECIAL_TOKENS,
        )
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def encode_file_to_raw_uint16(
    tokenizer: BPETokenizer,
    input_path: str | Path,
    output_path: str | Path,
    progress_every_bytes: int = 1 << 30,
) -> tuple[int, int]:
    output_path = Path(output_path)
    total_tokens = 0
    total_bytes = 0
    next_progress_mark = progress_every_bytes

    with open(input_path, "r", encoding="utf-8") as src, open(output_path, "wb") as dst:
        for line in src:
            encoded_line = line.encode("utf-8")
            token_ids = tokenizer.encode(line)
            np.asarray(token_ids, dtype=np.uint16).tofile(dst)
            total_tokens += len(token_ids)
            total_bytes += len(encoded_line)

            if total_bytes >= next_progress_mark:
                print(
                    f"[encode] {Path(input_path).name}: processed {total_bytes / (1 << 30):.2f} GiB, "
                    f"wrote {total_tokens:,} tokens",
                    file=sys.stderr,
                )
                next_progress_mark += progress_every_bytes

    return total_tokens, total_bytes


def count_file_tokens(tokenizer: BPETokenizer, input_path: str | Path) -> tuple[int, int]:
    total_tokens = 0
    total_bytes = 0

    with open(input_path, "r", encoding="utf-8") as src:
        for line in src:
            total_bytes += len(line.encode("utf-8"))
            total_tokens += len(tokenizer.encode(line))

    return total_tokens, total_bytes


def encode_file_to_npy_uint16(
    tokenizer: BPETokenizer,
    input_path: str | Path,
    output_path: str | Path,
    progress_every_bytes: int = 1 << 30,
) -> tuple[int, int]:
    total_tokens, total_bytes = count_file_tokens(tokenizer, input_path)
    memmap = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    written = 0
    processed_bytes = 0
    next_progress_mark = progress_every_bytes

    with open(input_path, "r", encoding="utf-8") as src:
        for line in src:
            token_ids = tokenizer.encode(line)
            chunk = np.asarray(token_ids, dtype=np.uint16)
            memmap[written : written + len(chunk)] = chunk
            written += len(chunk)
            processed_bytes += len(line.encode("utf-8"))

            if processed_bytes >= next_progress_mark:
                print(
                    f"[encode] {Path(input_path).name}: processed {processed_bytes / (1 << 30):.2f} GiB, "
                    f"wrote {written:,}/{total_tokens:,} tokens",
                    file=sys.stderr,
                )
                next_progress_mark += progress_every_bytes

    memmap.flush()
    return total_tokens, total_bytes


def run_experiments(num_documents: int, seed: int, benchmark_bytes: int) -> None:
    tiny_tokenizer = build_default_tokenizer("tinystories")
    owt_tokenizer = build_default_tokenizer("owt")

    tiny_docs = sample_documents(DEFAULT_TINYSTORIES_TRAIN, num_documents=num_documents, seed=seed)
    owt_docs = sample_documents(DEFAULT_OWT_TRAIN, num_documents=num_documents, seed=seed)

    tiny_ratio, tiny_bytes, tiny_tokens = compression_ratio_bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio, owt_bytes, owt_tokens = compression_ratio_bytes_per_token(owt_tokenizer, owt_docs)
    tiny_on_owt_ratio, _, tiny_on_owt_tokens = compression_ratio_bytes_per_token(tiny_tokenizer, owt_docs)

    benchmark_tiny_docs = take_documents_by_bytes(DEFAULT_TINYSTORIES_TRAIN, target_bytes=benchmark_bytes)
    benchmark_owt_docs = take_documents_by_bytes(DEFAULT_OWT_TRAIN, target_bytes=benchmark_bytes)
    tiny_bps, _, _, tiny_elapsed = benchmark_tokenizer_throughput(tiny_tokenizer, benchmark_tiny_docs)
    owt_bps, _, _, owt_elapsed = benchmark_tokenizer_throughput(owt_tokenizer, benchmark_owt_docs)

    pile_bytes = 825 * 10**9
    tiny_pile_seconds = pile_bytes / tiny_bps
    owt_pile_seconds = pile_bytes / owt_bps

    results = {
        "seed": seed,
        "num_documents": num_documents,
        "benchmark_bytes": benchmark_bytes,
        "tinystories": {
            "compression_bytes_per_token": tiny_ratio,
            "sample_bytes": tiny_bytes,
            "sample_tokens": tiny_tokens,
            "throughput_bytes_per_second": tiny_bps,
            "benchmark_elapsed_seconds": tiny_elapsed,
            "pile_825gb_seconds": tiny_pile_seconds,
            "pile_825gb_human": format_duration(tiny_pile_seconds),
        },
        "owt": {
            "compression_bytes_per_token": owt_ratio,
            "sample_bytes": owt_bytes,
            "sample_tokens": owt_tokens,
            "throughput_bytes_per_second": owt_bps,
            "benchmark_elapsed_seconds": owt_elapsed,
            "pile_825gb_seconds": owt_pile_seconds,
            "pile_825gb_human": format_duration(owt_pile_seconds),
        },
        "owt_with_tinystories_tokenizer": {
            "compression_bytes_per_token": tiny_on_owt_ratio,
            "sample_bytes": owt_bytes,
            "sample_tokens": tiny_on_owt_tokens,
        },
    }
    print(json.dumps(results, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BPE tokenizer utilities and assignment experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiments_parser = subparsers.add_parser("experiments", help="Run tokenizer experiments for assignment part (a)-(c).")
    experiments_parser.add_argument("--num-documents", type=int, default=10)
    experiments_parser.add_argument("--seed", type=int, default=42)
    experiments_parser.add_argument("--benchmark-bytes", type=int, default=8 * (1 << 20))

    encode_parser = subparsers.add_parser("encode-dataset", help="Encode a corpus to uint16 token ids.")
    encode_parser.add_argument("--tokenizer", choices=["tinystories", "owt"], required=True)
    encode_parser.add_argument("--input", type=Path, required=True)
    encode_parser.add_argument("--output", type=Path, required=True)
    encode_parser.add_argument(
        "--format",
        choices=["raw", "npy"],
        default="raw",
        help="`raw` writes a flat uint16 binary file; `npy` writes a NumPy .npy array.",
    )
    encode_parser.add_argument("--progress-every-bytes", type=int, default=1 << 30)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "experiments":
        run_experiments(
            num_documents=args.num_documents,
            seed=args.seed,
            benchmark_bytes=args.benchmark_bytes,
        )
        return

    if args.command == "encode-dataset":
        tokenizer = build_default_tokenizer(args.tokenizer)
        if args.format == "raw":
            total_tokens, total_bytes = encode_file_to_raw_uint16(
                tokenizer,
                input_path=args.input,
                output_path=args.output,
                progress_every_bytes=args.progress_every_bytes,
            )
        else:
            total_tokens, total_bytes = encode_file_to_npy_uint16(
                tokenizer,
                input_path=args.input,
                output_path=args.output,
                progress_every_bytes=args.progress_every_bytes,
            )
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "format": args.format,
                    "total_bytes": total_bytes,
                    "total_tokens": total_tokens,
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    main()
