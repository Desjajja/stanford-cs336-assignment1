import copy

def update_pretokens(pretokens: dict[tuple[bytes], int], new_merge: bytes):
    pretokens_copy = copy.deepcopy(pretokens)
    for token, count in pretokens.items():
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

pret = {(b'l', b'o', b'w'): 4, (b'l', b'o', b'w', b'e', b'r'): 1, (b'w', b'i', b'd', b'e', b's', b't'): 3, (b'n', b'e', b'w', b'e', b's', b't'): 5}

# print(update_pretokens(pret, b'st'))