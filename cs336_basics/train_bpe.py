import os
import regex as re

from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Iterable
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_worker_pretoken_re: re.Pattern
_worker_special_tokens: list[str]
_worker_special_split_re: re.Pattern

def _init_worker(pattern: str, special_tokens: list[str]):
    global _worker_pretoken_re, _worker_special_tokens, _worker_special_split_re
    _worker_pretoken_re = re.compile(pattern)
    _worker_special_tokens = special_tokens
    _worker_special_split_re = re.compile("|".join(re.escape(token) for token in special_tokens))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _build_file_chunk_tasks(
    input_path: str | os.PathLike,
    num_workers: int,
    split_special_token: bytes,
) -> list[tuple[str | os.PathLike, int, int]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_special_token)
    return [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

def _iter_trainable_segments(text: str) -> Iterable[str]:
    for piece in _worker_special_split_re.split(text):
        if not piece:
            continue
        if piece in _worker_special_tokens:
            continue
        yield piece

def _count_pretokens_in_text(text: str) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for segment in _iter_trainable_segments(text):
        for match in _worker_pretoken_re.finditer(segment):
            pretoken_bytes = match.group(0).encode("utf-8")
            counts[pretoken_bytes] += 1
    return counts
        
def _count_pretokens_in_chunk(task: tuple[str | os.PathLike, int, int]) -> Counter[bytes]:
    input_path, start, end = task
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens_in_text(chunk)

def _count_adjacent_pairs(sequences: list[list[int]], frequencies: list[int]) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:#注意统计的是token id而不是bytes
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_positions: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for i, (seq, freq) in enumerate(zip(sequences, frequencies)):
        if len(seq) < 2:
            continue
        left = seq[0]
        for j in range(len(seq) - 1):
            left = seq[j]
            right = seq[j + 1]
            pair_counts[(left, right)] += freq
            if (left, right) not in pair_positions:
                pair_positions[(left, right)] = []
            pair_positions[(left, right)].append((i, j))
    
    return pair_counts, pair_positions

def _select_best_pair(pair_counts: Counter[tuple[int, int]], token_bytes: list[bytes]) -> tuple[int, int]:
    return max(
        pair_counts, 
        key=lambda pair: (
            pair_counts[pair],
            token_bytes[pair[0]],
            token_bytes[pair[1]],
        ),
    )

def _remove_sequence_pair(seq_num: int, seq: list[int], frequency: int, pair_counts: Counter[tuple[int, int]], pair_positions: dict[tuple[int, int], list[tuple[int, int]]]):
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i + 1])
        pair_counts[pair] -= frequency
        if pair_counts[pair] == 0:
            del pair_counts[pair]
        pair_positions[pair].remove((seq_num, i))
        if not pair_positions[pair]:
            del pair_positions[pair]

def _add_sequence_pair(seq_num: int, merged_seq: list[int], frenquency: int, pair_counts: Counter[tuple[int, int]], pair_positions: dict[tuple[int, int], list[tuple[int, int]]]):
    for i in range(len(merged_seq) - 1):
        pair = (merged_seq[i], merged_seq[i + 1])
        pair_counts[pair] += frenquency
        pair_positions.setdefault(pair, []).append((seq_num, i))

def _merge_pair_in_sequence(
    seq_num: int, 
    seq: list[int], 
    frequency: int,
    left_id: int, 
    right_id: int, 
    merged_id: int, 
    pair_counts: Counter[tuple[int, int]], 
    pair_positions: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[int]:
    merged_seq: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left_id and seq[i + 1] == right_id:
            merged_seq.append(merged_id)
            i += 2
        else:
            merged_seq.append(seq[i])
            i += 1
    
    _remove_sequence_pair(seq_num, seq, frequency, pair_counts, pair_positions)
    _add_sequence_pair(seq_num, merged_seq, frequency, pair_counts, pair_positions)

    return merged_seq


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    base_vocab_size = 256 + len(special_tokens)
    num_workers = max(1, cpu_count() - 1)

    tasks = _build_file_chunk_tasks(input_path, num_workers, split_special_token=special_tokens[0].encode("utf-8"))
    pretoken_counts: Counter[bytes] = Counter()

    with Pool(num_workers, initializer=_init_worker, initargs=(PAT, special_tokens)) as pool:
        for local_counts in pool.imap_unordered(_count_pretokens_in_chunk, tasks, chunksize=1):
            pretoken_counts.update(local_counts)

    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    token_bytes: list[bytes] = [bytes([token_id]) for token_id in range(256)]

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
        token_bytes.append(token.encode("utf-8"))

    sequences: list[list[int]] = []
    frequiencies: list[int] = []

    for pretoken_bytes, count in pretoken_counts.items():
        sequences.append(list(pretoken_bytes))
        frequiencies.append(count)

    max_merges = vocab_size - base_vocab_size
    merges: list[tuple[bytes, bytes]] = []

    pair_counts, pair_positions = _count_adjacent_pairs(sequences, frequiencies)

    for _ in range(max_merges):
        left_id, right_id = _select_best_pair(pair_counts, token_bytes)

        merged_token_bytes = token_bytes[left_id] + token_bytes[right_id]
        merged_id = len(token_bytes)

        token_bytes.append(merged_token_bytes)
        vocab[merged_id] = merged_token_bytes
        merges.append((token_bytes[left_id], token_bytes[right_id]))

        affected_seq_ids = sorted({seq_id for seq_id, _ in pair_positions.get((left_id, right_id), [])})

        for seq_id in affected_seq_ids:#注意对同一个seq只能处理1次，否则会出现重复合并的情况，比如一个pair在一个seq中出现了两次，那么第一次合并后这个pair就不存在了，如果第二次还处理这个seq就会出问题，所以只能处理一次
            sequences[seq_id] = _merge_pair_in_sequence(seq_id, sequences[seq_id], frequiencies[seq_id], left_id, right_id, merged_id, pair_counts, pair_positions)
        
    return vocab, merges