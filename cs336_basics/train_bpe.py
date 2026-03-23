import heapq
import json
import os
from collections import Counter
from multiprocessing import cpu_count, get_all_start_methods, get_context
from typing import BinaryIO, Iterable

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_MIN_CHUNK_BYTES = 1 << 20
HEAP_REBUILD_FACTOR = 4
PAIR_SHIFT = 32
PAIR_MASK = (1 << PAIR_SHIFT) - 1

_POOL_CONTEXT = get_context("fork") if "fork" in get_all_start_methods() else get_context()

_worker_pretoken_re: re.Pattern[str]
_worker_special_tokens: frozenset[str]
_worker_special_split_re: re.Pattern[str] | None
type DescBytesKey = tuple[int, ...]
type PairHeapItem = tuple[int, DescBytesKey, DescBytesKey, int]


def _decode_pair(pair: int) -> tuple[int, int]:
    return pair >> PAIR_SHIFT, pair & PAIR_MASK


def _descending_bytes_key(token: bytes) -> DescBytesKey:
    return tuple(255 - byte for byte in token) + (256,)


def _init_worker(pattern: str, special_tokens: list[str]) -> None:
    global _worker_pretoken_re, _worker_special_tokens, _worker_special_split_re
    _worker_pretoken_re = re.compile(pattern)
    _worker_special_tokens = frozenset(special_tokens)
    if special_tokens:
        special_tokens_pattern = "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
        _worker_special_split_re = re.compile(f"({special_tokens_pattern})")
    else:
        _worker_special_split_re = None


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

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        boundary_position = chunk_boundaries[boundary_index]
        file.seek(boundary_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = boundary_position + found_at
                break
            boundary_position += mini_chunk_size

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
    if _worker_special_split_re is None:
        yield text
        return

    for piece in _worker_special_split_re.split(text):
        if not piece or piece in _worker_special_tokens:
            continue
        yield piece


def _count_pretokens_in_text(text: str) -> dict[bytes, int]:
    counts: dict[bytes, int] = {}
    finditer = _worker_pretoken_re.finditer

    for segment in _iter_trainable_segments(text):
        for match in finditer(segment):
            pretoken_bytes = match.group(0).encode("utf-8")
            counts[pretoken_bytes] = counts.get(pretoken_bytes, 0) + 1

    return counts


def _count_pretokens_in_chunk(task: tuple[str | os.PathLike, int, int]) -> dict[bytes, int]:
    input_path, start, end = task
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens_in_text(chunk)


def _resolve_num_workers(input_path: str | os.PathLike, requested_num_workers: object) -> int:
    if requested_num_workers is not None:
        return max(1, int(requested_num_workers))

    file_size = os.path.getsize(input_path)
    available_workers = max(1, cpu_count())
    size_limited_workers = max(1, file_size // DEFAULT_MIN_CHUNK_BYTES)
    return min(available_workers, size_limited_workers)


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int,
    split_boundary: bytes,
) -> Counter[bytes]:
    pretoken_counts: Counter[bytes] = Counter()

    if num_workers == 1:
        _init_worker(PAT, special_tokens)
        with open(input_path, "rb") as f:
            pretoken_counts.update(_count_pretokens_in_text(f.read().decode("utf-8", errors="ignore")))
        return pretoken_counts

    tasks = _build_file_chunk_tasks(input_path, num_workers, split_special_token=split_boundary)
    if len(tasks) == 1:
        _init_worker(PAT, special_tokens)
        pretoken_counts.update(_count_pretokens_in_chunk(tasks[0]))
        return pretoken_counts

    chunk_size = max(1, len(tasks) // (num_workers * 4))
    pool = _POOL_CONTEXT.Pool(num_workers, initializer=_init_worker, initargs=(PAT, special_tokens))
    try:
        for local_counts in pool.imap_unordered(_count_pretokens_in_chunk, tasks, chunksize=chunk_size):
            pretoken_counts.update(local_counts)
    except Exception:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    return pretoken_counts


def _build_merge_state(
    pretoken_counts: Counter[bytes],
) -> tuple[list[int], list[int], list[int], list[int], dict[int, int], dict[int, set[int]]]:
    node_token: list[int] = []
    node_next: list[int] = []
    node_prev: list[int] = []
    node_frequency: list[int] = []
    pair_counts: dict[int, int] = {}
    pair_occurrences: dict[int, set[int]] = {}

    for pretoken_bytes, frequency in pretoken_counts.items():
        prev_node = -1
        for token in pretoken_bytes:
            node_id = len(node_token)
            node_token.append(token)
            node_next.append(-1)
            node_prev.append(prev_node)
            node_frequency.append(frequency)

            if prev_node != -1:
                node_next[prev_node] = node_id
                pair = (node_token[prev_node] << PAIR_SHIFT) | token
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
                left_nodes = pair_occurrences.get(pair)
                if left_nodes is None:
                    pair_occurrences[pair] = {prev_node}
                else:
                    left_nodes.add(prev_node)

            prev_node = node_id

    return node_token, node_next, node_prev, node_frequency, pair_counts, pair_occurrences


def _build_pair_heap(
    pair_counts: dict[int, int],
    token_order_keys: list[DescBytesKey],
) -> list[PairHeapItem]:
    pair_heap = [
        (-count, token_order_keys[pair >> PAIR_SHIFT], token_order_keys[pair & PAIR_MASK], pair)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(pair_heap)
    return pair_heap


def _push_pair_heap_item(
    pair_heap: list[PairHeapItem],
    pair: int,
    count: int,
    token_order_keys: list[DescBytesKey],
) -> None:
    heapq.heappush(pair_heap, (-count, token_order_keys[pair >> PAIR_SHIFT], token_order_keys[pair & PAIR_MASK], pair))


def _select_best_pair_from_heap(
    pair_counts: dict[int, int],
    pair_heap: list[PairHeapItem],
) -> int:
    while pair_heap:
        neg_count, _, _, pair = pair_heap[0]
        if pair_counts.get(pair) == -neg_count:
            return pair
        heapq.heappop(pair_heap)

    raise ValueError("Pair heap became empty while pair counts were still present.")


def _merge_pair_occurrences(
    pair: int,
    merged_id: int,
    node_token: list[int],
    node_next: list[int],
    node_prev: list[int],
    node_frequency: list[int],
    pair_counts: dict[int, int],
    pair_occurrences: dict[int, set[int]],
    pair_heap: list[PairHeapItem],
    token_order_keys: list[DescBytesKey],
) -> None:
    left_id, right_id = _decode_pair(pair)
    occurrence_nodes = sorted(pair_occurrences.get(pair, ()))
    pair_counts_get = pair_counts.get
    pair_counts_pop = pair_counts.pop
    pair_occurrences_get = pair_occurrences.get
    pair_occurrences_pop = pair_occurrences.pop
    heap_push = heapq.heappush
    token_order_keys_local = token_order_keys

    for left_node in occurrence_nodes:
        if node_token[left_node] != left_id:
            continue

        right_node = node_next[left_node]
        if right_node == -1 or node_token[right_node] != right_id:
            continue

        frequency = node_frequency[left_node]
        prev_node = node_prev[left_node]
        next_node = node_next[right_node]

        if prev_node != -1:
            prev_pair = (node_token[prev_node] << PAIR_SHIFT) | left_id
            new_count = pair_counts_get(prev_pair, 0) - frequency
            if new_count:
                pair_counts[prev_pair] = new_count
                heap_push(
                    pair_heap,
                    (-new_count, token_order_keys_local[prev_pair >> PAIR_SHIFT], token_order_keys_local[prev_pair & PAIR_MASK], prev_pair),
                )
            else:
                pair_counts_pop(prev_pair, None)

            left_nodes = pair_occurrences_get(prev_pair)
            if left_nodes is not None:
                left_nodes.discard(prev_node)
                if not left_nodes:
                    pair_occurrences_pop(prev_pair, None)

        new_count = pair_counts_get(pair, 0) - frequency
        if new_count:
            pair_counts[pair] = new_count
            heap_push(
                pair_heap,
                (-new_count, token_order_keys_local[pair >> PAIR_SHIFT], token_order_keys_local[pair & PAIR_MASK], pair),
            )
        else:
            pair_counts_pop(pair, None)

        left_nodes = pair_occurrences_get(pair)
        if left_nodes is not None:
            left_nodes.discard(left_node)
            if not left_nodes:
                pair_occurrences_pop(pair, None)

        if next_node != -1:
            right_pair = (right_id << PAIR_SHIFT) | node_token[next_node]
            new_count = pair_counts_get(right_pair, 0) - frequency
            if new_count:
                pair_counts[right_pair] = new_count
                heap_push(
                    pair_heap,
                    (-new_count, token_order_keys_local[right_pair >> PAIR_SHIFT], token_order_keys_local[right_pair & PAIR_MASK], right_pair),
                )
            else:
                pair_counts_pop(right_pair, None)

            left_nodes = pair_occurrences_get(right_pair)
            if left_nodes is not None:
                left_nodes.discard(right_node)
                if not left_nodes:
                    pair_occurrences_pop(right_pair, None)

        node_token[left_node] = merged_id
        node_next[left_node] = next_node
        if next_node != -1:
            node_prev[next_node] = left_node

        node_token[right_node] = -1
        node_prev[right_node] = -1
        node_next[right_node] = -1

        if prev_node != -1:
            new_prev_pair = (node_token[prev_node] << PAIR_SHIFT) | merged_id
            new_count = pair_counts_get(new_prev_pair, 0) + frequency
            pair_counts[new_prev_pair] = new_count
            heap_push(
                pair_heap,
                (
                    -new_count,
                    token_order_keys_local[new_prev_pair >> PAIR_SHIFT],
                    token_order_keys_local[new_prev_pair & PAIR_MASK],
                    new_prev_pair,
                ),
            )
            left_nodes = pair_occurrences_get(new_prev_pair)
            if left_nodes is None:
                pair_occurrences[new_prev_pair] = {prev_node}
            else:
                left_nodes.add(prev_node)

        if next_node != -1:
            new_next_pair = (merged_id << PAIR_SHIFT) | node_token[next_node]
            new_count = pair_counts_get(new_next_pair, 0) + frequency
            pair_counts[new_next_pair] = new_count
            heap_push(
                pair_heap,
                (
                    -new_count,
                    token_order_keys_local[new_next_pair >> PAIR_SHIFT],
                    token_order_keys_local[new_next_pair & PAIR_MASK],
                    new_next_pair,
                ),
            )
            left_nodes = pair_occurrences_get(new_next_pair)
            if left_nodes is None:
                pair_occurrences[new_next_pair] = {left_node}
            else:
                left_nodes.add(left_node)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    base_vocab_size = 256 + len(special_tokens)
    num_workers = _resolve_num_workers(input_path, kwargs.get("num_workers"))
    verbose = bool(kwargs.get("verbose", False))

    split_boundary = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
    pretoken_counts = _count_pretokens(input_path, special_tokens, num_workers, split_boundary)

    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    token_bytes: list[bytes] = [bytes([token_id]) for token_id in range(256)]
    token_order_keys: list[DescBytesKey] = [_descending_bytes_key(token) for token in token_bytes]

    for token in special_tokens:
        token_as_bytes = token.encode("utf-8")
        vocab[len(vocab)] = token_as_bytes
        token_bytes.append(token_as_bytes)
        token_order_keys.append(_descending_bytes_key(token_as_bytes))

    max_merges = vocab_size - base_vocab_size
    merges: list[tuple[bytes, bytes]] = []

    node_token, node_next, node_prev, node_frequency, pair_counts, pair_occurrences = _build_merge_state(pretoken_counts)
    pair_heap = _build_pair_heap(pair_counts, token_order_keys)

    for merge_index in range(max_merges):
        if verbose and merge_index % 100 == 0:
            print(f"Merge {merge_index}/{max_merges} - vocab size {len(vocab)}")
        if not pair_counts:
            break
        if len(pair_heap) > len(pair_counts) * HEAP_REBUILD_FACTOR:
            pair_heap = _build_pair_heap(pair_counts, token_order_keys)

        best_pair = _select_best_pair_from_heap(pair_counts, pair_heap)
        left_id, right_id = _decode_pair(best_pair)
        merged_token_bytes = token_bytes[left_id] + token_bytes[right_id]
        merged_id = len(token_bytes)

        token_bytes.append(merged_token_bytes)
        token_order_keys.append(_descending_bytes_key(merged_token_bytes))
        vocab[merged_id] = merged_token_bytes
        merges.append((token_bytes[left_id], token_bytes[right_id]))

        _merge_pair_occurrences(
            best_pair,
            merged_id,
            node_token,
            node_next,
            node_prev,
            node_frequency,
            pair_counts,
            pair_occurrences,
            pair_heap,
            token_order_keys,
        )

    return vocab, merges

def main() -> None:
    vocab, merges = run_train_bpe(
        "/Users/ogyco/PythonProject/cs336/assignment1-basics/data/owt_valid.txt",
        32000,
        ["<|endoftext|>"],
    )
    with open("/Users/ogyco/PythonProject/cs336/assignment1-basics/data/owt_bpe_vocab.json", "w") as f:
        json.dump(
            {token_id: token_bytes.decode("utf-8", errors="ignore") for token_id, token_bytes in vocab.items()},
            f,
            indent=2,
        )
    with open("/Users/ogyco/PythonProject/cs336/assignment1-basics/data/owt_bpe_merges.json", "w") as f:
        json.dump(
            [(left.decode("utf-8", errors="ignore"), right.decode("utf-8", errors="ignore")) for left, right in merges],
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
