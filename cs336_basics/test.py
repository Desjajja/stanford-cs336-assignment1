import copy

def update_pretokens(pretokens, new_merge):
    pretokens_copy = copy.deepcopy(pretokens)
    for token, count in pretokens.items():
        flag_modified = False
        if len(token) < 2: 
            continue
        pleft = 0
        token_copy = list(token)
        while pleft < len(token_copy) - 1:
            pright = pleft + 1
            if token_copy[pleft] + token_copy[pright] == new_merge:
                token_copy[pleft] = new_merge
                token_copy.pop(pright)
                # pleft += 1
                flag_modified = True
            pleft += 1
        if flag_modified:
            del pretokens_copy[token]
            pretokens_copy[tuple(token_copy)] = count
    del pretokens
    return pretokens_copy

pret = {('l', 'o', 'w'): 4, ('l', 'o', 'w', 'e', 'r'): 1, ('w', 'i', 'd', 'e', 's', 't'): 3, ('n', 'e', 'w', 'e', 's', 't'): 5}

update_pretokens(pret, 'st')