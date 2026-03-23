def merge_pair_occurrences(
    pair: int,
    merged_id: int,
    node_token: list[int],
    node_next: list[int],
    node_prev: list[int],
    node_frequency: list[int],
    pair_counts: dict[int, int],
    pair_occurrences: dict[int, list[int]],
    pair_heap: list[tuple[int, int]],
) -> None: ...
