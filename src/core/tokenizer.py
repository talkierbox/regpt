"""Tokenizer""" 

# Character level tokenizer
# Each token is itself a number
class Tokenizer:
    def __init__(self, vocab: str):
        self.alphabet = sorted(list(set(vocab)))

        self._char_to_token = {a: i for i, a in enumerate(self.alphabet)}
        self._token_to_char = {i: a for i, a in enumerate(self.alphabet)}

    def encode(self, st: str) -> list[int]:
        return [self._char_to_token[s] for s in st]
    
    def decode(self, token: list[int]) -> str:
        return ''.join(self._token_to_char[t] for t in token)    


if __name__ == "__main__":
    tokenizer = Tokenizer("abcdefghijklmnopqrstuvwxyz")
    print(tokenizer.encode("hello"))
    print(tokenizer.decode([1, 2, 3, 4, 5]))