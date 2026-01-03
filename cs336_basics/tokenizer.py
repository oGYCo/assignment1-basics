import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _init_vocab(vocab: dict, special_token:list):
    special_token_encoded = [s.encode('UTF-8') for s in special_token]
    idx = 0
    for code in special_token_encoded:
        vocab[idx] = code
        idx += 1
    
    for i in range(256):
        init_str = bytes([i])
        if init_str not in vocab.values():
            vocab[idx] = init_str
            idx += 1
    return vocab

def pre_tokenization(s: str, special_token: list[str]) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # ① 没有 special 也要按正则切
    if not special_token:
        return re.findall(PAT, s)

    # ③ 长→短排序，防止短的抢先匹配
    toks = sorted(special_token, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"({union})", s)

    out = []
    st = set(special_token)
    for part in parts:
        if not part:
            continue
        # ② special 只作为边界，完全跳过
        if part in st:
            continue
        out.extend(re.findall(PAT, part))
    return out

def word_2_byte(word: str) -> tuple[bytes, ...]:
    word_decoded = list(word.encode('UTF-8'))
    #split the bytes
    word_byte = [bytes([b]) for b in word_decoded]
    return tuple(word_byte)

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #initialize the vocabulary
    vocab = _init_vocab({}, special_tokens)
    #pretokenization
    cnt_pretokens = Counter()
    with open(input_path, 'r', encoding='UTF-8') as f:
        text = f.read()
    chunked_text = pre_tokenization(text, special_tokens)
    for word in chunked_text:
        cnt_pretokens[word_2_byte(word)] += 1
    #merge
    merge_rule = []
    while len(vocab) < vocab_size:
        pair_cnt = Counter()
        for pretoken, cnt in cnt_pretokens.items():
            #pretoken e.g. (b'a', b'b', b'x0f8')
            for i in range(len(pretoken)-1):
                pair = (pretoken[i], pretoken[i+1])
                pair_cnt[pair] += cnt
        if not pair_cnt:
            break
        max_cnt = max(pair_cnt.values())
        candidate = [p for p,cnt in pair_cnt.items() if cnt == max_cnt]
        merge_pair = max(pair_cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merge_rule.append(merge_pair)
        n = len(vocab)
        new_token = merge_pair[0] + merge_pair[1]
        vocab[n] = new_token
        #now we can apply the merge to tokens
        change = []
        for pretoken, cnt in cnt_pretokens.items():
            start_idx = [i for i in range(len(pretoken)-1) if pretoken[i:i+2] == merge_pair]
            if start_idx:
                i = 0
                new_pre_token = []
                while i < len(pretoken):
                    if i in start_idx:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(pretoken[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                change.append([new_pre_token, pretoken, cnt])
        if not change:
            break
        for new_t, old_t, cnt in change:
            cnt_pretokens[new_t] += cnt
            cnt_pretokens[old_t] -= cnt
            if cnt_pretokens[old_t] <= 0:
                del cnt_pretokens[old_t]

    return vocab, merge_rule