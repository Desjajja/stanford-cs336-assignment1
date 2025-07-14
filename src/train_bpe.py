import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
import concurrent.futures
import copy
import time
from tqdm import tqdm



class BPETokenizer:
    SPECIAL_TOKENS = ["<|endoftext|>"]

    PAT = (
        f"(?:{'|'.join(map(re.escape, SPECIAL_TOKENS))})|"
        + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )  # pretokenization rule

    pretokens = defaultdict(int)
    merges = []

    def __init__(self, num_processes: int = 8):
        # MAX_VOCAB = 60
        self.vocab = [token.encode() for token in self.SPECIAL_TOKENS]
        self.vocab.extend([bytes([i]) for i in range(256)])
        self.num_processes = num_processes  # Number of processes to use for parallelization
        print()

    def _pretokenize(self, data_path: str):
        with open(data_path, "rb") as fp:
            boundaries = self._find_chunk_boundaries(
                fp, self.num_processes, self.SPECIAL_TOKENS[0].encode("utf-8")
            )  # By convention the first element is end-of-text token
        pbar = tqdm(total=len(boundaries) - 1, desc="Pretokenizing...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self._init_pretokens, data_path, start, end)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
                for byte, count in result.items():
                    self.pretokens[byte] += count
                pbar.update(1)

            except Exception as e:
                print(e)
        pbar.close()
        
    def _update_pretokens(self, new_merge: bytes):
        pretokens_copy = copy.deepcopy(self.pretokens)
        for token, count in self.pretokens.items():
            flag_modified = False
            if len(token) < 2:
                continue
            pleft = 0
            token_copy = list()
            while pleft < len(token) - 1:
                pright = pleft + 1
                if token[pleft] + token[pright] == new_merge:
                    token_copy.append(new_merge)
                    # pleft += 1
                    flag_modified = True
                    pleft += 2
                else:
                    token_copy.append(token[pleft])
                    pleft += 1
            if pleft == len(token) - 1:
                token_copy.append(token[-1])
            if flag_modified:
                del pretokens_copy[token]
                pretokens_copy[tuple(token_copy)] = count
                del token
            else:
                del token_copy
        return pretokens_copy

    def _find_chunk_boundaries(self, fp: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()
        fp.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        max_len_special = max(map(len, self.SPECIAL_TOKENS))

        for bi in range(0, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            fp.seek(initial_position)  # Start at boundary guess

            while True:
                mini_chunk = fp.read(mini_chunk_size + max_len_special - 1)  # Read a mini chunk
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi + 1] = file_size
                    break
                found_at, spec_token_len = min(
                    [(mini_chunk.rfind(token.encode()), len(token)) for token in self.SPECIAL_TOKENS],
                    key=lambda pair: pair[0],
                )

                if found_at != -1 and found_at < (mini_chunk_size - 1):
                    # chunk_boundaries[bi] = initial_position + found_at
                    if bi + 1 <= len(chunk_boundaries) - 1:
                        chunk_boundaries[bi + 1] = max(
                            chunk_boundaries[bi + 1], initial_position + found_at + spec_token_len
                        )  # let's assume that there no such thing as <spec><spec>...<spec>
                    break
                initial_position += mini_chunk_size
        return sorted(set(chunk_boundaries))

    def _init_pretokens(self, data_path, start: int, end: int) -> dict[tuple[bytes, ...], int] | None:
        with open(data_path, "rb") as fp:
            fp.seek(start)
            chunk = fp.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
        word_count = Counter(r.group().strip() for r in re.finditer(self.PAT, chunk) if r.group().strip())
        return {
            tuple(bytes([c]) for c in word.encode()): count
            for word, count in word_count.items()
            if word not in self.SPECIAL_TOKENS
        }

    def get_pair_count(self):
        pair_count = defaultdict(int)
        for token, count in self.pretokens.items():
            if len(token) > 1:
                for lhs, rhs in zip(token[:-1], token[1:]):
                    pair_count[(lhs, rhs)] += count
        return pair_count

    def find_max_pair(self, pair_count):
        return max(pair_count.items(), key=lambda item: (item[1], item[0]))

    def train(self, data_path: str, max_merges: int):
        ts = time.time()
        self._pretokenize(data_path)
        te = time.time()
        print(f"Pretokenization took {te - ts:.2f} seconds")

        ts = time.time()
        pbar = tqdm(total=max_merges, desc="Training...")
        while len(self.merges) < max_merges:
            pair_count = self.get_pair_count()
            if len(pair_count) == 0:  # exhausted all pairs to merge
                break
            max_pair, _ = self.find_max_pair(pair_count)
            self.merges.append(max_pair)
            new_token = max_pair[0] + max_pair[1]
            self.vocab.append(new_token)
            self.pretokens = self._update_pretokens(new_token)
            pbar.update(1)
        pbar.close()
        te = time.time()
        print(f"Done training in {te - ts:.2f} seconds")

        vocab = {idx: byte for idx, byte in enumerate(self.vocab)}
        return vocab, self.merges


if __name__ == "__main__":
    profile = True
    data_path = "./data/TinyStoriesV2-GPT4-train.txt"
    max_merges = 10_000
    def main():
        tokenizer = BPETokenizer(num_processes=8)
        vocab, merges = tokenizer.train(data_path, max_merges)
        print(vocab)
        print(merges)
    if profile:
        import cProfile
        import pstats
        
        cProfile.run('main()', "./profile/profile.txt")
        p = pstats.Stats("./profile/profile.txt")
        p.sort_stats("cumulative").print_stats(50)
        p.print_callees(20)
        # Save readable stats to a text file
        with open("./profile/multiprocess_train_10_000.txt", "w") as f:
            p.stream = f
            p.print_stats()
    else:
        
        main()
