import math
from collections import defaultdict

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from marginal_surprisal.lattice import build_lattice
from marginal_surprisal.vocab import has_dummy_prefix, start_token_id, token_surface_bytes


class MarginalScorer:
    """Surprisal from a causal LM, marginalized over all tokenizations of the string.

    log P(s) is computed as a sum over every segmentation of the UTF-8 bytes of `s` into
    vocabulary tokens (a byte-lattice DP), rather than scoring only the canonical tokenization.

    Approximation: the next-token distribution at byte position j is conditioned on the
    *canonical* tokenization of s[:j], not on the particular path the DP took to reach j.
    Exact path-conditioning is exponential; this is the usual tractable approximation.

    Build one scorer per model and reuse it: the vocabulary scan happens once, in __init__.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | None = None,
        start_id: int | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.start_id = start_id if start_id is not None else start_token_id(tokenizer)
        self.dummy_prefix = has_dummy_prefix(tokenizer)

        output_embeddings = model.get_output_embeddings()
        n_logits = output_embeddings.weight.shape[0] if output_embeddings is not None else None

        # Several ids can share a surface; their probabilities are summed.
        surface_to_ids: dict[bytes, list[int]] = defaultdict(list)
        for tid, surface in enumerate(token_surface_bytes(tokenizer)):
            if surface is not None and (n_logits is None or tid < n_logits):
                surface_to_ids[surface].append(tid)
        self.surface_to_ids = {
            surf: torch.tensor(ids, dtype=torch.long) for surf, ids in surface_to_ids.items()
        }
        self.surfaces = set(self.surface_to_ids)
        self.max_token_len = max(len(surf) for surf in self.surfaces)
        # Used to condition on a prefix that ends inside a multi-byte character.
        self.single_byte_ids = {
            surf[0]: int(ids[0]) for surf, ids in self.surface_to_ids.items() if len(surf) == 1
        }

    def surprisal(self, context: str, continuations: list[str], sep: str = " ") -> list[float]:
        """Surprisal (in nats) of each continuation given the context and earlier continuations.

        Scores the string `context + sep + continuations[0] + sep + continuations[1] + ...`
        and returns -log P(continuation_k | everything before it) for each k.
        """
        text = context
        bounds = [len(self._internal(context))]
        for word in continuations:
            text += sep + word
            bounds.append(len(self._internal(text)))

        prefix_lp = self.prefix_log_probs(text)
        for b in bounds:
            if prefix_lp[b] == -math.inf:
                raise ValueError(f"No tokenization covers {text!r} up to byte {b}.")
        return [prefix_lp[a] - prefix_lp[b] for a, b in zip(bounds, bounds[1:])]

    def log_prob(self, text: str) -> float:
        """Marginal log P(text), summed over all tokenizations."""
        return self.prefix_log_probs(text)[-1]

    def prefix_log_probs(self, text: str) -> list[float]:
        """Return lp where lp[i] = marginal log-probability of all tokenizations ending at byte i.

        Positions are byte offsets into `self._internal(text)`.
        """
        s = self._internal(text)
        n = len(s)
        lattice = build_lattice(s, self.surfaces, self.max_token_len)

        lp = [-math.inf] * (n + 1)
        lp[0] = 0.0
        for j in range(n):
            if lp[j] == -math.inf or not lattice[j]:
                continue
            next_lp = self._next_token_log_probs(s[:j])
            for i, tok in lattice[j]:
                tok_lp = torch.logsumexp(next_lp[self.surface_to_ids[tok]], dim=0).item()
                lp[i] = _log_add_exp(lp[i], lp[j] + tok_lp)
        return lp

    def _internal(self, text: str) -> bytes:
        """The bytes the model actually sees: SentencePiece dummy-prefix tokenizers add a space."""
        return (" " + text if self.dummy_prefix else text).encode()

    def _next_token_log_probs(self, prefix: bytes) -> torch.Tensor:
        """log P(next token | canonical tokenization of prefix), over the full vocabulary."""
        # A prefix can end inside a multi-byte character: tokenize the complete characters
        # canonically and append the dangling bytes as single-byte tokens.
        k = len(prefix)
        while True:
            try:
                complete = prefix[:k].decode()
                break
            except UnicodeDecodeError:
                k -= 1
        if self.dummy_prefix:
            # The tokenizer adds the leading space itself.
            complete = complete[1:]
        ids = [self.start_id] + self.tokenizer.encode(complete, add_special_tokens=False)
        ids += [self.single_byte_ids[b] for b in prefix[k:]]

        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1]
        return torch.log_softmax(logits.float(), dim=-1).cpu()


def _log_add_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))
