from collections.abc import Iterable
import regex as re
from functools import reduce
import json


class BPETokenizer:
    
    def build_special_tokens_pattern(self, tokens_list):
        """Sorts special tokens by length and creates a regex pattern."""
        # 1. Escape special regex characters in each token
        escaped_tokens = [re.escape(token) for token in tokens_list]
        # 2. Sort by length in descending order
        # This is crucial to match '<|endoftext|><|endoftext|>' before '<|endoftext|>'
        sorted_tokens = sorted(escaped_tokens, key=len, reverse=True)
        # 3. Join with '|' to create the final OR pattern
        return "|".join(sorted_tokens)
    
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
        # self, vocab: dict[str, int], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        # self.token2idx = vocab  # idx to token
        self.token2idx = {token: idx for idx, token in vocab.items()}
        # self.token2idx = {token.encode('utf-8'): idx for token, idx in vocab.items()}
        self.idx2token = vocab
        # self.idx2token = {idx: token.encode('utf-8') for token, idx in vocab.items()}  # idx to token
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        if self.special_tokens != []:
            self.PAT = (
                f"(?:{'|'.join(map(re.escape, self.special_tokens))})|"
                # + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                 r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+| ?[^\s\p{L}\p{N}]+?"
            )  # pretokenization rule
        else:
            self.PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+| ?[^\s\p{L}\p{N}]+?"
            # self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            # vocab = dict(tuple(map(eval, line.strip().split("\t"))) for line in f if line.strip())
            vocab = json.load(f)
            vocab_new = {idx: token.replace('\u0120', ' ').encode('utf-8') for token, idx in vocab.items()}
            
        
        merges = []  
        with open(merges_filepath, encoding="utf-8") as f:
            # merges = [tuple(map(eval, line.strip().split(" "))) for line in f]
            for line in f:
                # output = tuple(
                    # map(lambda c: b" " if c == '\u0120' else c.encode(),
                    # line.strip().split(" "))
                    # )
                output = tuple(map(lambda x: x.replace('\u0120', ' ').encode('utf-8'), line.strip('\n').split(' ')))
                merges.append(output)

        return cls(vocab=vocab_new, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        # into list[tuple[bytes]]
        special_tokens = {}  # record special tokens's positioin -> val
        pretokens = []
        for idx, r in enumerate(re.finditer(self.PAT, text)):
            token = r.group()
            if token in self.special_tokens:
                special_tokens[idx] = token
                pretokens.append(None)
            else:
                pretokens.append([bytes([c]) for c in token.encode()])

        # merging: [(t, h, e), ...] -> [(t, h), ...,  (th, e)] (merges)
        merged_list = []
        for idx, pretoken in enumerate(pretokens):
            if pretoken is None:
                merged_list.append(special_tokens[idx].encode())
                continue
            
            for idx, merge in enumerate(self.merges):
                mleft, mright = merge                
                for i in range(len(pretoken) - 1):
                    if pretoken[i] == mleft and pretoken[i+1] == mright:
                        pretoken[i] = mleft + mright
                        pretoken.pop(i + 1)
                        break
                    
            merged_list.extend(pretoken)
        

        return [self.token2idx[token] for token in merged_list]

    def encode_old(self, text: str) -> list[int]:
        # into list[tuple[bytes]]
        special_tokens = {}  # record special tokens's positioin -> val
        pretokens = []
        for idx, r in enumerate(re.finditer(self.PAT, text)):
            token = r.group()
            if token in self.special_tokens:
                special_tokens[idx] = token
                pretokens.append(None)
            else:
                pretokens.append(tuple(bytes([c]) for c in token.encode()))

        # merging: [(t, h, e), ...] -> [(t, h), ...,  (th, e)] (merges)
        merged_list = []
        for idx, pretoken in enumerate(pretokens):
            if pretoken is None:
                merged_list.append(special_tokens[idx].encode())
                continue
            pleft, pright = 0, 1
            while pleft < len(pretoken) - 1:
                merged = pretoken[pleft]
                for pright in range(pleft + 1, len(pretoken)):
                    for merge in self.merges:
                        segment = b"".join(pretoken[pleft + len(merged) : pright + 1])
                        if (merged, segment) == merge:
                            merged += segment
                            break
                pleft += len(merged)
                merged_list.append(merged)
            if pleft == len(pretoken) - 1:  # last element
                merged_list.append(pretoken[-1])

        return [self.token2idx[token] for token in merged_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, ids: list[int]) -> str:
        return reduce(lambda x, y: x + y, [self.idx2token[idx] for idx in ids], b"").decode(errors="ignore")


if __name__ == "__main__":
    
    output_dir = "output/TinyStoriesV2-GPT4-train_10000/"
    tokenizer = BPETokenizer.from_files(
        # vocab_filepath= output_dir + "vocab.json",
        # merges_filepath= output_dir + "merges.txt",
        vocab_filepath= "tests/fixtures/gpt2_vocab.json",
        merges_filepath= "tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    # tokenizer.encode("    ")
    print("Type anything (Ctrl-C to exit):")
    try:
        while True:
            text = input("> ")
            embed = tokenizer.encode(text)
            print(embed)
            print("decoded back:")
            print(tokenizer.decode(embed))
    except KeyboardInterrupt:
        print("\nExiting.")
