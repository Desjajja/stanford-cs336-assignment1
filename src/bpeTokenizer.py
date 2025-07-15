from collections.abc import Iterable


class BPETokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merge: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        ...
        
    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merge_filepath,
        special_tokens=None
    ):
        with open(vocab_filepath, encoding='utf-8') as f:
            vocab = dict(tuple(map(eval, line.strip().split('\t'))) for line in f if line.strip())
            
        with open(merge_filepath, encoding='utf-8') as f:
            merge = [tuple(map(eval, line.strip().split(' '))) for line in f if line.strip()]
            
        return cls(
            vocab=vocab,
            merge=merge,
            special_tokens=special_tokens
        )
        
    def encode(self, text: str) -> list[int]:
        ...
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        ...
        
    def decode(self, ids: list[int]) -> str:
        ...