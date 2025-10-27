import re
from collections import OrderedDict

class SmilesTokenizer:
    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

    def __init__(self, special_tokens=None):
        self.regex = re.compile(self.SMI_REGEX_PATTERN)
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<CLS>", "<MASK>"]
        self.vocab = OrderedDict()
        self.inv_vocab = OrderedDict()
        self._finalized = False

    def tokenize(self, sequence):
        import pdb; pdb.set_trace()
        tokens = []
        for part in sequence.strip().split():
            if part in self.special_tokens:
                tokens.append(part)
            else:
                tokens.extend(self.regex.findall(part))
        return tokens

    def build_vocab(self, sequences):
        """Build vocab from a list of raw strings (SMILES sequences)."""
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1

        for seq in sequences:
            for token in self.tokenize(seq):
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._finalized = True
        self.vocab_size = len(self.vocab)

    def encode(self, sequence):
        if not self._finalized:
            raise RuntimeError("Tokenizer vocab not built. Call build_vocab() first.")
        return [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in self.tokenize(sequence)]

    def decode(self, indices):
        if not self._finalized:
            raise RuntimeError("Tokenizer vocab not built. Call build_vocab() first.")
        return [self.inv_vocab.get(i, "<UNK>") for i in indices]

    def save_vocab(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, path):
        import json
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        self._finalized = True
        self.vocab_size = len(self.vocab)
