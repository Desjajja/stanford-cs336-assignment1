from collections.abc import Iterable
import regex as re
from functools import reduce
import json


class BPETokenizer:
    
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.token2idx = {token: idx for idx, token in vocab.items()}
        self.idx2token = vocab
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens else set()
        
        # Add special tokens to vocabulary if they're not already there
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            if special_token_bytes not in self.token2idx:
                # Add special token to vocabulary with next available index
                next_idx = max(self.idx2token.keys()) + 1 if self.idx2token else 0
                self.token2idx[special_token_bytes] = next_idx
                self.idx2token[next_idx] = special_token_bytes
        
        # Build regex pattern
        if self.special_tokens:
            # Sort special tokens by length (descending) to match longer tokens first
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_special)
            self.PAT = (
                f"(?:{special_pattern})|"
                + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )
        else:
            self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
            vocab_new = {idx: token.replace('\u0120', ' ').encode('utf-8') for token, idx in vocab.items()}
        
        # Ensure all possible bytes (0-255) are in vocabulary
        max_idx = max(vocab_new.keys()) if vocab_new else -1
        for byte_val in range(256):
            byte_token = bytes([byte_val])
            if byte_token not in {token for token in vocab_new.values()}:
                max_idx += 1
                vocab_new[max_idx] = byte_token
        
        merges = []  
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(' ')
                    if len(parts) == 2:  # Ensure we have exactly 2 parts
                        output = tuple(part.replace('\u0120', ' ').encode('utf-8') for part in parts)
                        merges.append(output)

        return cls(vocab=vocab_new, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using BPE algorithm."""
        if not text:
            return []
        
        # Step 1: Pre-tokenization - split text into chunks
        chunks = []
        for match in re.finditer(self.PAT, text):
            token = match.group()
            if token in self.special_tokens:
                # Special tokens are kept as-is
                chunks.append(('special', token))
            else:
                # Regular tokens are split into bytes
                chunks.append(('regular', [bytes([b]) for b in token.encode('utf-8')]))
        
        # Step 2: Apply BPE merges to regular tokens
        result_tokens = []
        for chunk_type, chunk_data in chunks:
            if chunk_type == 'special':
                # Special tokens are encoded directly
                special_token_bytes = chunk_data.encode('utf-8')
                result_tokens.append(special_token_bytes)
            else:
                # Apply BPE merges to regular tokens
                tokens = chunk_data.copy()  # Work on a copy
                
                # Apply each merge in order
                for merge_left, merge_right in self.merges:
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == merge_left and tokens[i + 1] == merge_right:
                            # Merge the two tokens
                            merged_token = merge_left + merge_right
                            tokens[i] = merged_token
                            tokens.pop(i + 1)
                            # Don't increment i, check the same position again
                        else:
                            i += 1
                
                result_tokens.extend(tokens)
        
        # Step 3: Convert tokens to IDs
        token_ids = []
        for token in result_tokens:
            if token in self.token2idx:
                token_ids.append(self.token2idx[token])
            else:
                # Handle unknown tokens - this shouldn't happen with proper vocabulary
                raise ValueError(f"Token not found in vocabulary: {token}")
        
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Encode an iterable of strings."""
        for string in iterable:
            yield from self.encode(string)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not ids:
            return ""
        
        try:
            tokens = [self.idx2token[idx] for idx in ids]
            return b"".join(tokens).decode('utf-8', errors='replace')
        except KeyError as e:
            raise ValueError(f"Token ID not found in vocabulary: {e}")

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.idx2token)
    
    def get_vocab(self) -> dict[bytes, int]:
        """Return the token to index mapping."""
        return self.token2idx.copy()


if __name__ == "__main__":
    output_dir = "output/TinyStoriesV2-GPT4-train_10000/"
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="tests/fixtures/gpt2_vocab.json",
        merges_filepath="tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    
    print("Type anything (Ctrl-C to exit):")
    try:
        while True:
            text = input("> ")
            embed = tokenizer.encode(text)
            print(f"Encoded: {embed}")
            print(f"Decoded: {repr(tokenizer.decode(embed))}")
    except KeyboardInterrupt:
        print("\nExiting.")