import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
import concurrent.futures
import copy
import time
from tqdm import tqdm



class BPETrainer:
    SPECIAL_TOKENS = ["<|endoftext|>"]

    PAT = (
        f"(?:{'|'.join(map(re.escape, SPECIAL_TOKENS))})|"
        + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )  # pretokenization rule

    pretokens = defaultdict(int)
    merges = []
    pair_count = defaultdict(int)

    def __init__(self, num_processes: int = 8):
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
                # print(result)
                for byte, count in result.items():
                    self.pretokens[byte] += count
                pbar.update(1)

            except Exception as e:
                print(e)
        pbar.close()
        
    def _update_pretokens(self, new_merge: bytes):
        # pretokens_copy = copy.deepcopy(self.pretokens)
        changed_pretokens = dict() # orgin pretoken -> updated pretoken
        assert self.pretokens is not None
        for token, count in self.pretokens.items():
            flag_modified = False
            if len(token) < 2:
                continue
            pleft = 0
            token_copy = list()
            previous_token = None
            merged_token_lhs = None
            merged_token_rhs = None
            while pleft < len(token) - 1:
                pright = pleft + 1
                if token[pleft] + token[pright] == new_merge:
                    merged_token_lhs = token[pleft]
                    merged_token_rhs = token[pright]
                    
                    token_copy.append(new_merge)
                    if previous_token is None:
                        pass
                    elif previous_token == new_merge:
                        self.pair_count[(merged_token_rhs, merged_token_lhs)] -= count
                        self.pair_count[(new_merge, new_merge)] += count
                    elif previous_token:
                        self.pair_count[(previous_token, merged_token_lhs)] -= count
                        self.pair_count[(previous_token, new_merge)] += count

                    previous_token = new_merge
                    flag_modified = True
                    pleft += 2
                else:
                    if previous_token == new_merge:
                        self.pair_count[(merged_token_rhs, token[pleft])] -= count
                    token_copy.append(token[pleft])
                    previous_token = token[pleft]
                    pleft += 1
                
            if pleft == len(token) - 1:
                token_copy.append(token[-1])
                if previous_token == new_merge:
                    self.pair_count[(merged_token_rhs, token[pleft])] -= count
            if flag_modified:
                changed_pretokens[token] = tuple(token_copy)
   
        for ori_token, new_token in changed_pretokens.items():
            count = self.pretokens[ori_token]
            del self.pretokens[ori_token]
            self.pretokens[new_token] = count

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
        word_count = Counter(r.group() for r in re.finditer(self.PAT, chunk))
        # word_count = Counter(r.group().strip() for r in re.finditer(self.PAT, chunk) if r.group().strip())
        return {
            tuple(bytes([c]) for c in word.encode()): count
            for word, count in word_count.items()
            if word not in self.SPECIAL_TOKENS
        }

    def _init_pair_count(self):
        pair_count = defaultdict(int)
        assert self.pretokens is not None
        for token, count in self.pretokens.items():
            if len(token) > 1:
                for lhs, rhs in zip(token[:-1], token[1:]):
                    pair_count[(lhs, rhs)] += count
        self.pair_count = pair_count

    def pop_max_pair(self):
        pair, count = max(self.pair_count.items(), key=lambda item: (item[1], item[0]))
        del self.pair_count[pair]
        return pair, count



    def train(self, data_path: str, max_merges: int):
        ts = time.time()
        self._pretokenize(data_path)
        self._init_pair_count()
        te = time.time()
        print(f"Pretokenization took {te - ts:.2f} seconds")

        ts = time.time()
        pbar = tqdm(total=max_merges, desc="Training...")
        previous_pair = None
        previous_count = None
        while len(self.merges) < max_merges and len(self.pair_count) > 0:
            max_pair, count = self.pop_max_pair()
            if (previous_pair, previous_count) == (max_pair, count):
                break # no more new merges
            self.merges.append(max_pair)
            new_token = max_pair[0] + max_pair[1]
            self.vocab.append(new_token)
            self._update_pretokens(new_token)
            pbar.update(1)
        pbar.close()
        te = time.time()
        print(f"Done training in {te - ts:.2f} seconds")

        vocab = {idx: byte for idx, byte in enumerate(self.vocab)}
        return vocab, self.merges
    
    


if __name__ == "__main__":
    import os
    profile = False
    data_path = "./data/TinyStoriesV2-GPT4-train.txt"
    data_type = data_path.split('/')[-1].split('.')[0]
    max_merges = 10_000
    num_processes = 8
    def main():
        tokenizer = BPETrainer(num_processes)
        vocab, merges = tokenizer.train(data_path, max_merges)
        os.makedirs(f"./output/{data_type}_{max_merges}", exist_ok=True)
        with open(f"./output/{data_type}_{max_merges}/merges.txt", "w") as f_merges:
            for merge in merges:
                f_merges.write(f"{merge[0]}\t{merge[1]}\n")
        with open(f"./output/{data_type}_{max_merges}/vocab.txt", "w") as f_vocab:
            for idx, token in vocab.items():
                f_vocab.write(f"{idx}\t{token}\n")
    if profile:
        import cProfile
        import pstats
        
        cProfile.run('main()', "./output/profile.txt")
        p = pstats.Stats("./output/profile.txt")
        p.sort_stats("cumulative").print_stats(50)
        p.print_callees(20)
        # Save readable stats to a text file
        with open(f"./output/cached_train-{max_merges}.txt", "w") as f:
            p.stream = f
            p.print_stats()
        if os.path.exists("./output/profile.txt"):
            os.remove("./output/profile.txt")
    else:
        
        main()
