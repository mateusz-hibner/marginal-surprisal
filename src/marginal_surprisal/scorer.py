"""Character-level surprisal from a token-level causal LM, following

    Giulianelli, Malagutti, Gastaldi, DuSell, Vieira & Cotterell (2024),
    "On the Proper Treatment of Tokenization in Psycholinguistics", and
    Vieira et al. (2024), "From Language Models over Tokens to Language Models over Characters".

The character-level prefix probability of a string sigma is a sum over its *prefix cover*:

    ->p_Sigma(sigma) = sum_{delta in C(sigma)} ->p_Delta(delta)
    C(sigma) = { d_1 ... d_m : kappa(d_1 ... d_{m-1}) < sigma <= kappa(d_1 ... d_m) }

i.e. every token sequence whose decoding starts with sigma, where only the last token may run past
the end of sigma. Each sequence is scored with the LM conditioned on its *own* token history.
Surprisal of a region is -log p(region | preceding text), built from next-character conditionals.

The cover grows exponentially with the string length, so -- as in the paper (beam size 5) -- it is
pruned with Vieira et al.'s `prune_top_K_buckets` after every character: members of the cover are
bucketed by their token sequence without its last token if that token runs past the current
position, or by the whole sequence if it ends exactly there; the K most probable buckets are kept.
`beam_size=None` disables pruning and gives the exact quantities.

"Characters" are bytes of the UTF-8 encoding, as in the paper (Sigma is "typically bytes"). Region
boundaries always fall on character boundaries, where byte- and character-level prefix
probabilities coincide.
"""

import bisect
import heapq
import math
from dataclasses import dataclass

import torch

from marginal_surprisal.vocab import has_dummy_prefix, start_token_id, token_surface_bytes


@dataclass
class _Closed:
    """A cover member whose decoding ends exactly at the current position."""

    tokens: tuple[int, ...]
    logp: float  # log ->p_Delta(tokens)


@dataclass
class _Open:
    """A bucket: all cover members `history + (t,)` whose last token t runs past the current position.

    kappa(history) == s[:start]; the tokens t are those whose surface starts with s[start:pos].
    """

    history: tuple[int, ...]
    start: int
    logp: float  # log ->p_Delta(history)
    next_lp: torch.Tensor  # log p(. | history), in sorted-surface order
    mass: float  # log of the bucket's total probability at the current position


class MarginalScorer:
    """Character-level surprisal from a causal LM, marginalized over tokenizations.

    Build one scorer per model and reuse it: the vocabulary scan happens once, in __init__.

    Args:
        model: a causal LM returning `.logits` of shape (batch, seq, vocab).
        tokenizer: its Hugging Face tokenizer. Optional if `surfaces`, `start_id` and `eos_id`
            are given explicitly.
        device: where to run the model (default: the model's device).
        start_id: the token every sequence is conditioned on first (default: whatever the tokenizer
            prepends, else bos, else eos).
        beam_size: number of buckets K kept by the pruning heuristic (paper: 5). None = exact.
        eos_id: end-of-string token, used by `log_prob` (default: tokenizer.eos_token_id).
        surfaces: surfaces[token_id] = the bytes that token stands for, or None for tokens that
            do not produce text (special tokens). Default: derived from the tokenizer.
        dummy_prefix: whether the tokenizer adds a leading space (SentencePiece). Default: detected.
        batch_size: max sequences per forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer=None,
        device: torch.device | None = None,
        start_id: int | None = None,
        *,
        beam_size: int | None = 5,
        eos_id: int | None = None,
        surfaces: list[bytes | None] | None = None,
        dummy_prefix: bool | None = None,
        batch_size: int = 32,
    ):
        if tokenizer is None and (surfaces is None or start_id is None):
            raise ValueError("Pass a tokenizer, or both `surfaces` and `start_id`.")
        if beam_size is not None and beam_size < 1:
            raise ValueError("beam_size must be a positive integer or None.")

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.start_id = start_id if start_id is not None else start_token_id(tokenizer)
        self.eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_token_id", None)
        if dummy_prefix is None:
            dummy_prefix = has_dummy_prefix(tokenizer) if tokenizer is not None else False
        self.dummy_prefix = dummy_prefix
        self.beam_size = beam_size
        self.batch_size = batch_size

        if surfaces is None:
            surfaces = token_surface_bytes(tokenizer)
        output_embeddings = model.get_output_embeddings()
        n_logits = output_embeddings.weight.shape[0] if output_embeddings is not None else None

        # Vocabulary sorted by surface: the tokens whose surface starts with r form one contiguous
        # range, found by binary search; duplicates (several ids, one surface) sit side by side.
        pairs = sorted(
            (surf, tid)
            for tid, surf in enumerate(surfaces)
            if surf and (n_logits is None or tid < n_logits)
        )
        if not pairs:
            raise ValueError("No token has a surface string.")
        self._surfaces = [surf for surf, _ in pairs]
        self._ids = torch.tensor([tid for _, tid in pairs], dtype=torch.long)

    # ----------------------------------------------------------------------------- public API

    def surprisal(self, context: str, continuations: list[str], sep: str = " ") -> list[float]:
        """Surprisal (in nats) of each region given the context and all earlier regions.

        Scores `context + sep + continuations[0] + sep + continuations[1] + ...`. Each region is
        `sep + continuation` (leading whitespace, as is conventional). Returns
        -log p(region_k | everything before it) for each k.
        """
        text = context
        bounds = [len(self._internal(context))]
        for word in continuations:
            text += sep + word
            bounds.append(len(self._internal(text)))
        lp = self.prefix_log_probs(text)
        return [lp[a] - lp[b] for a, b in zip(bounds, bounds[1:])]

    def prefix_log_probs(self, text: str) -> list[float]:
        """lp[i] = log ->p_Sigma(s[:i]) for the bytes s the model sees (see `_internal`).

        Exact when beam_size is None. With pruning, it is the chain-rule product of the pruned
        model's next-character conditionals (Vieira et al.'s `next_character_probability`).
        """
        cond, _ = self._run(self._internal(text), with_eos=False)
        lp = [0.0]
        for c in cond:
            lp.append(lp[-1] + c)
        return lp

    def byte_log_probs(self, text: str) -> list[float]:
        """log p(s[i] | s[:i]) for every byte of the string the model sees."""
        return self._run(self._internal(text), with_eos=False)[0]

    def log_prob(self, text: str) -> float:
        """log p_Sigma(text): probability of the complete string, i.e. followed by EOS."""
        if self.eos_id is None:
            raise ValueError("No EOS token id; pass eos_id.")
        cond, eos = self._run(self._internal(text), with_eos=True)
        return sum(cond) + eos

    # ----------------------------------------------------------------------------- internals

    def _internal(self, text: str) -> bytes:
        """The bytes the model actually sees: SentencePiece dummy-prefix tokenizers add a space."""
        return (" " + text if self.dummy_prefix else text).encode()

    def _run(self, s: bytes, with_eos: bool) -> tuple[list[float], float | None]:
        """Walk the string byte by byte, maintaining the (pruned) prefix cover.

        Returns the next-byte conditionals log p(s[b] | s[:b]) and, if requested,
        log p(EOS | s).
        """
        closed = [_Closed((), 0.0)]  # C(empty string) = {empty sequence}
        opens: list[_Open] = []
        log_z = 0.0  # log mass of the current (pruned) cover
        cond: list[float] = []

        for b in range(len(s)):
            new_closed: list[_Closed] = []
            new_open: list[_Open] = []

            # Cover members whose last token already runs past b: keep those that match s[b].
            for o in opens:
                self._extend(o.history, o.start, o.logp, o.next_lp, s[o.start : b + 1],
                             new_closed, new_open)

            # Cover members that end exactly at b: extend them by one more token.
            if closed:
                next_lps, _ = self._next_log_probs([c.tokens for c in closed])
                for c, next_lp in zip(closed, next_lps):
                    self._extend(c.tokens, b, c.logp, next_lp, s[b : b + 1], new_closed, new_open)

            masses = [c.logp for c in new_closed] + [o.mass for o in new_open]
            if not masses:
                if self.beam_size is None:
                    raise ValueError(f"No token sequence covers {s[: b + 1]!r}.")
                raise ValueError(
                    f"The beam (beam_size={self.beam_size}) kept no token sequence that can "
                    f"continue to {s[: b + 1]!r}: the pruned model gives it probability 0. "
                    "Increase beam_size, or pass beam_size=None for the exact computation."
                )
            cond.append(_logsumexp(masses) - log_z)

            closed, opens = self._prune(new_closed, new_open)
            log_z = _logsumexp([c.logp for c in closed] + [o.mass for o in opens])

        eos = None
        if with_eos:
            if closed:
                _, eos_lps = self._next_log_probs([c.tokens for c in closed], want_eos=True)
                eos = _logsumexp([c.logp + e for c, e in zip(closed, eos_lps)]) - log_z
            else:
                eos = -math.inf
        return cond, eos

    def _extend(self, history, start, logp, next_lp, r, new_closed, new_open):
        """Split the tokens t following `history` whose surface starts with r = s[start:pos].

        Tokens with surface == r complete a cover member ending exactly at pos; tokens with a
        longer surface stay together in one bucket keyed by `history`.
        """
        lo, mid, hi = self._range(r)
        for k in range(lo, mid):
            lp = next_lp[k].item()
            if lp > -math.inf:
                new_closed.append(_Closed(history + (int(self._ids[k]),), logp + lp))
        if mid < hi:
            m = torch.logsumexp(next_lp[mid:hi], dim=0).item()
            if m > -math.inf:
                new_open.append(_Open(history, start, logp, next_lp, logp + m))

    def _prune(self, closed, opens):
        """Vieira et al.'s prune_top_K_buckets: keep the K most probable buckets."""
        if self.beam_size is None or len(closed) + len(opens) <= self.beam_size:
            return closed, opens
        buckets = [(c.logp, 0, i) for i, c in enumerate(closed)]
        buckets += [(o.mass, 1, i) for i, o in enumerate(opens)]
        top = heapq.nlargest(self.beam_size, buckets)
        return (
            [closed[i] for _, kind, i in top if kind == 0],
            [opens[i] for _, kind, i in top if kind == 1],
        )

    def _range(self, r: bytes) -> tuple[int, int, int]:
        """Sorted-vocabulary indices: [lo, mid) have surface == r, [mid, hi) extend r."""
        lo = bisect.bisect_left(self._surfaces, r)
        mid = bisect.bisect_right(self._surfaces, r, lo=lo)
        succ = _successor(r)
        hi = len(self._surfaces) if succ is None else bisect.bisect_left(self._surfaces, succ, lo=mid)
        return lo, mid, hi

    def _next_log_probs(self, seqs, want_eos=False):
        """log p(. | start, *seq) for each seq, in sorted-surface order (and log p(EOS | ...))."""
        sorted_lps: list[torch.Tensor] = []
        eos_lps: list[float] = []
        for i in range(0, len(seqs), self.batch_size):
            chunk = seqs[i : i + self.batch_size]
            lengths = [len(seq) + 1 for seq in chunk]
            width = max(lengths)
            ids = torch.full((len(chunk), width), self.start_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, seq in enumerate(chunk):
                ids[row, : lengths[row]] = torch.tensor((self.start_id, *seq), dtype=torch.long)
                mask[row, : lengths[row]] = 1
            # Right padding: with causal attention the real positions never see the padding.
            with torch.no_grad():
                logits = self.model(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
                ).logits
            last = logits[torch.arange(len(chunk)), torch.tensor(lengths) - 1]
            lp = torch.log_softmax(last.double(), dim=-1).cpu()
            if want_eos:
                eos_lps.extend(lp[:, self.eos_id].tolist())
            else:
                sorted_lps.extend(lp[:, self._ids])
        return sorted_lps, eos_lps


def _successor(r: bytes) -> bytes | None:
    """Smallest byte string greater than every string that starts with r (None if none is)."""
    r = r.rstrip(b"\xff")
    if not r:
        return None
    return r[:-1] + bytes([r[-1] + 1])


def _logsumexp(xs: list[float]) -> float:
    xs = [x for x in xs if x > -math.inf]
    if not xs:
        return -math.inf
    hi = max(xs)
    return hi + math.log(sum(math.exp(x - hi) for x in xs))
