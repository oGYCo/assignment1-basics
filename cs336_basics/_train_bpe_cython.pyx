# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, infer_types=True

import heapq

cdef inline unsigned long long _encode_pair(int left_id, int right_id):
    return ((<unsigned long long>left_id) << 32) | <unsigned int>right_id


def merge_pair_occurrences(
    unsigned long long pair,
    int merged_id,
    list node_token,
    list node_next,
    list node_prev,
    list node_frequency,
    dict pair_counts,
    dict pair_occurrences,
    list pair_heap,
):
    cdef int left_id = <int>(pair >> 32)
    cdef int right_id = <int>(pair & 0xFFFFFFFF)
    cdef list occurrence_nodes = pair_occurrences.pop(pair, [])
    cdef dict count_deltas = {}
    cdef dict new_occurrences = {}
    cdef Py_ssize_t index
    cdef int left_node
    cdef int right_node
    cdef int prev_node
    cdef int next_node
    cdef int frequency
    cdef int token_id
    cdef int new_count
    cdef unsigned long long prev_pair
    cdef unsigned long long right_pair
    cdef unsigned long long new_prev_pair
    cdef unsigned long long new_next_pair
    cdef object appended_nodes
    cdef object left_nodes
    cdef object updated_pair
    cdef object delta
    cdef int old_count

    occurrence_nodes.sort()
    pair_counts.pop(pair, None)

    for index in range(len(occurrence_nodes)):
        left_node = <int>occurrence_nodes[index]
        if <int>node_token[left_node] != left_id:
            continue

        right_node = <int>node_next[left_node]
        if right_node == -1 or <int>node_token[right_node] != right_id:
            continue

        frequency = <int>node_frequency[left_node]
        prev_node = <int>node_prev[left_node]
        next_node = <int>node_next[right_node]

        if prev_node != -1:
            token_id = <int>node_token[prev_node]
            prev_pair = _encode_pair(token_id, left_id)
            count_deltas[prev_pair] = <int>count_deltas.get(prev_pair, 0) - frequency

        if next_node != -1:
            token_id = <int>node_token[next_node]
            right_pair = _encode_pair(right_id, token_id)
            count_deltas[right_pair] = <int>count_deltas.get(right_pair, 0) - frequency

        node_token[left_node] = merged_id
        node_next[left_node] = next_node
        if next_node != -1:
            node_prev[next_node] = left_node

        node_token[right_node] = -1
        node_prev[right_node] = -1
        node_next[right_node] = -1

        if prev_node != -1:
            token_id = <int>node_token[prev_node]
            new_prev_pair = _encode_pair(token_id, merged_id)
            count_deltas[new_prev_pair] = <int>count_deltas.get(new_prev_pair, 0) + frequency
            left_nodes = new_occurrences.get(new_prev_pair)
            if left_nodes is None:
                new_occurrences[new_prev_pair] = [prev_node]
            else:
                left_nodes.append(prev_node)

        if next_node != -1:
            token_id = <int>node_token[next_node]
            new_next_pair = _encode_pair(merged_id, token_id)
            count_deltas[new_next_pair] = <int>count_deltas.get(new_next_pair, 0) + frequency
            left_nodes = new_occurrences.get(new_next_pair)
            if left_nodes is None:
                new_occurrences[new_next_pair] = [left_node]
            else:
                left_nodes.append(left_node)

    for updated_pair, delta in count_deltas.items():
        if not delta:
            continue
        old_count = <int>pair_counts.get(updated_pair, 0)
        new_count = old_count + <int>delta
        if new_count == old_count:
            continue
        if new_count:
            pair_counts[updated_pair] = new_count
            heapq.heappush(pair_heap, (-new_count, updated_pair))
        else:
            pair_counts.pop(updated_pair, None)

    for updated_pair, appended_nodes in new_occurrences.items():
        left_nodes = pair_occurrences.get(updated_pair)
        if left_nodes is None:
            pair_occurrences[updated_pair] = appended_nodes
        else:
            left_nodes.extend(appended_nodes)
