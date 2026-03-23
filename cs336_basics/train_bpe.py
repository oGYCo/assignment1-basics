import heapq
import json
import os
from collections import Counter
from multiprocessing import cpu_count, get_all_start_methods, get_context
from typing import BinaryIO, Callable, Iterable, SupportsIndex, SupportsInt, cast

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_MIN_CHUNK_BYTES = 1 << 20
PRETOKEN_TASKS_PER_WORKER = 4
HEAP_REBUILD_FACTOR = 5
PAIR_SHIFT = 32
PAIR_MASK = (1 << PAIR_SHIFT) - 1

_POOL_CONTEXT = get_context("fork") if "fork" in get_all_start_methods() else get_context()

_worker_pretoken_re: re.Pattern[str]
_worker_special_tokens: frozenset[str]
_worker_special_split_re: re.Pattern[str] | None
_worker_single_special_token: str | None
type PairHeapItem = tuple[int, int]
type ConvertibleToInt = str | bytes | bytearray | SupportsInt | SupportsIndex
type MergePairOccurrences = Callable[
    [int, int, list[int], list[int], list[int], list[int], dict[int, int], dict[int, list[int]], list[PairHeapItem]],
    None,
]

try:
    from ._train_bpe_cython import merge_pair_occurrences as _merge_pair_occurrences_cython
except ImportError:
    _merge_pair_occurrences_cython = None


def _decode_pair(pair: int) -> tuple[int, int]:
    return pair >> PAIR_SHIFT, pair & PAIR_MASK


def _init_worker(pattern: str, special_tokens: list[str]) -> None:
    global _worker_pretoken_re, _worker_single_special_token, _worker_special_tokens, _worker_special_split_re
    _worker_pretoken_re = re.compile(pattern)
    _worker_special_tokens = frozenset(special_tokens)
    _worker_single_special_token = special_tokens[0] if len(special_tokens) == 1 else None
    if len(special_tokens) > 1:
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
    if _worker_single_special_token is not None:
        token = _worker_single_special_token
        token_length = len(token)
        segment_start = 0

        while True:
            token_start = text.find(token, segment_start)
            if token_start == -1:
                if segment_start < len(text):
                    yield text[segment_start:]
                return
            if token_start > segment_start:
                yield text[segment_start:token_start]
            segment_start = token_start + token_length

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


def _resolve_num_workers(
    input_path: str | os.PathLike,
    requested_num_workers: ConvertibleToInt | None,
) -> int:
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

    tasks = _build_file_chunk_tasks(
        input_path,
        num_workers * PRETOKEN_TASKS_PER_WORKER,
        split_special_token=split_boundary,
    )
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
) -> tuple[list[int], list[int], list[int], list[int], dict[int, int], dict[int, list[int]]]:
    node_token: list[int] = []
    node_next: list[int] = []
    node_prev: list[int] = []
    node_frequency: list[int] = []
    pair_counts: dict[int, int] = {}
    pair_occurrences: dict[int, list[int]] = {}

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
                    pair_occurrences[pair] = [prev_node]
                else:
                    left_nodes.append(prev_node)

            prev_node = node_id

    return node_token, node_next, node_prev, node_frequency, pair_counts, pair_occurrences
def _build_pair_heap(
    pair_counts: dict[int, int],
) -> list[PairHeapItem]:
    pair_heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)
    return pair_heap


def _select_best_pair_from_heap(
    pair_counts: dict[int, int],
    pair_heap: list[PairHeapItem],
    token_bytes: list[bytes],
) -> int:
    top_count = 0
    best_pair = -1
    best_left = b""
    best_right = b""
    tied_pairs: list[int] = []
    pair_counts_get = pair_counts.get
    heap_pop = heapq.heappop
    heap_push = heapq.heappush
    token_bytes_local = token_bytes

    while pair_heap:
        neg_count, pair = heap_pop(pair_heap)
        count = pair_counts_get(pair, 0)
        if count != -neg_count:
            continue
        if best_pair == -1:
            top_count = count
            best_pair = pair
            best_left = token_bytes_local[pair >> PAIR_SHIFT]
            best_right = token_bytes_local[pair & PAIR_MASK]
            continue
        if count != top_count:
            heap_push(pair_heap, (neg_count, pair))
            break
        left = token_bytes_local[pair >> PAIR_SHIFT]
        right = token_bytes_local[pair & PAIR_MASK]
        if left > best_left or (left == best_left and right > best_right):
            tied_pairs.append(best_pair)
            best_pair = pair
            best_left = left
            best_right = right
        elif pair != best_pair:
            tied_pairs.append(pair)

    if best_pair == -1:
        raise ValueError("Pair heap became empty while pair counts were still present.")

    for pair in tied_pairs:
        heap_push(pair_heap, (-top_count, pair))

    return best_pair


def _merge_pair_occurrences_python(
    pair: int,
    merged_id: int,
    node_token: list[int],
    node_next: list[int],
    node_prev: list[int],
    node_frequency: list[int],
    pair_counts: dict[int, int],
    pair_occurrences: dict[int, list[int]],
    pair_heap: list[PairHeapItem],
) -> None:
    left_id, right_id = _decode_pair(pair)
    occurrence_nodes = pair_occurrences.pop(pair, [])
    occurrence_nodes.sort()
    count_deltas: dict[int, int] = {}
    new_occurrences: dict[int, list[int]] = {}
    count_deltas_get = count_deltas.get
    new_occurrences_get = new_occurrences.get
    pair_counts_get = pair_counts.get
    pair_counts_pop = pair_counts.pop
    heap_push = heapq.heappush
    pair_counts_pop(pair, None)

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
            count_deltas[prev_pair] = count_deltas_get(prev_pair, 0) - frequency

        if next_node != -1:
            right_pair = (right_id << PAIR_SHIFT) | node_token[next_node]
            count_deltas[right_pair] = count_deltas_get(right_pair, 0) - frequency

        node_token[left_node] = merged_id
        node_next[left_node] = next_node
        if next_node != -1:
            node_prev[next_node] = left_node

        node_token[right_node] = -1
        node_prev[right_node] = -1
        node_next[right_node] = -1

        if prev_node != -1:
            new_prev_pair = (node_token[prev_node] << PAIR_SHIFT) | merged_id
            count_deltas[new_prev_pair] = count_deltas_get(new_prev_pair, 0) + frequency
            left_nodes = new_occurrences_get(new_prev_pair)
            if left_nodes is None:
                new_occurrences[new_prev_pair] = [prev_node]
            else:
                left_nodes.append(prev_node)

        if next_node != -1:
            new_next_pair = (merged_id << PAIR_SHIFT) | node_token[next_node]
            count_deltas[new_next_pair] = count_deltas_get(new_next_pair, 0) + frequency
            left_nodes = new_occurrences_get(new_next_pair)
            if left_nodes is None:
                new_occurrences[new_next_pair] = [left_node]
            else:
                left_nodes.append(left_node)

    for updated_pair, delta in count_deltas.items():
        if not delta:
            continue
        old_count = pair_counts_get(updated_pair, 0)
        new_count = old_count + delta
        if new_count == old_count:
            continue
        if new_count:
            pair_counts[updated_pair] = new_count
            heap_push(pair_heap, (-new_count, updated_pair))
        else:
            pair_counts_pop(updated_pair, None)

    for updated_pair, appended_nodes in new_occurrences.items():
        left_nodes = pair_occurrences.get(updated_pair)
        if left_nodes is None:
            pair_occurrences[updated_pair] = appended_nodes
        else:
            left_nodes.extend(appended_nodes)


def _resolve_merge_pair_occurrences(use_cython: bool) -> MergePairOccurrences:
    if use_cython and _merge_pair_occurrences_cython is not None:
        return cast(MergePairOccurrences, _merge_pair_occurrences_cython)
    return _merge_pair_occurrences_python


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    base_vocab_size = 256 + len(special_tokens)
    num_workers = _resolve_num_workers(input_path, cast(ConvertibleToInt | None, kwargs.get("num_workers")))
    merge_pair_occurrences = _resolve_merge_pair_occurrences(bool(kwargs.get("use_cython", True)))
    verbose = bool(kwargs.get("verbose", False))

    split_boundary = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
    pretoken_counts = _count_pretokens(input_path, special_tokens, num_workers, split_boundary)

    token_bytes: list[bytes] = [bytes([token_id]) for token_id in range(256)]

    for token in special_tokens:
        token_bytes.append(token.encode("utf-8"))

    max_merges = vocab_size - base_vocab_size
    merges: list[tuple[bytes, bytes]] = []

    node_token, node_next, node_prev, node_frequency, pair_counts, pair_occurrences = _build_merge_state(pretoken_counts)
    pair_heap = _build_pair_heap(pair_counts)

    for merge_index in range(max_merges):
        if verbose and merge_index % 100 == 0:
            print(f"Merge {merge_index}/{max_merges} - vocab size {len(token_bytes)}")
        if not pair_counts:
            break
        if len(pair_heap) > len(pair_counts) * HEAP_REBUILD_FACTOR:
            pair_heap = _build_pair_heap(pair_counts)

        best_pair = _select_best_pair_from_heap(pair_counts, pair_heap, token_bytes)
        left_id, right_id = _decode_pair(best_pair)
        merged_token_bytes = token_bytes[left_id] + token_bytes[right_id]
        merged_id = len(token_bytes)

        token_bytes.append(merged_token_bytes)
        merges.append((token_bytes[left_id], token_bytes[right_id]))

        merge_pair_occurrences(
            best_pair,
            merged_id,
            node_token,
            node_next,
            node_prev,
            node_frequency,
            pair_counts,
            pair_occurrences,
            pair_heap,
        )

    vocab: dict[int, bytes] = {token_id: token for token_id, token in enumerate(token_bytes)}
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
