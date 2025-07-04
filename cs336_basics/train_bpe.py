import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
from pretokenization_on import update_pretokens

SPECIAL_TOKENS = [
    "<|endoftext|>"
]

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
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


## Usage
num_processes = 8  # Number of processes to use for parallelization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # pretokenization rule
pretokens = defaultdict(int)
with open("/Users/desjajja/Projects/standord-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, SPECIAL_TOKENS[0].encode("utf-8")) # By convention the first element is end-of-text token

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        if chunk in SPECIAL_TOKENS:
            continue
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        word_count = Counter(r.group().strip() for r in re.finditer(PAT, chunk) if r.group().strip())
        pretokens = {tuple(bytes([c]) for c in word.encode()): count for word, count in word_count.items()}
        # for word, count in word_count.items():
        #     # word_bytes = word.encode("utf-8")
        #     if len(word) > 1:
        #         for lhs, rhs in zip(word[:-1], word[1:]):
        #             pretokens[(lhs, rhs)] += count

# init merges


def get_pair_count(pretokens):
    pair_count = defaultdict(int)
    for token, count in pretokens.items():
        if len(token) > 1:
            for lhs, rhs in zip(token[:-1], token[1:]):
                pair_count[(lhs, rhs)] += count
    return pair_count

def find_max_pair(merges):
    return max(merges.items(), key=lambda item: (item[1], item[0]))

# MAX_VOCAB = 60
vocab = [token.encode() for token in SPECIAL_TOKENS]
vocab.extend([bytes([i]) for i in range(256)])
merges = []
MAX_MERGES = 100
while len(merges) < MAX_MERGES:
    pair_count = get_pair_count(pretokens=pretokens)
    max_pair, count = find_max_pair(pair_count)
    merges.append(max_pair)
    new_token = max_pair[0] + max_pair[1]
    vocab.append(new_token)
    pretokens = update_pretokens(pretokens, new_token)
    # del merges

vocab = {
    idx: byte
    for idx, byte in enumerate(vocab)
}
print(vocab)
print(merges)