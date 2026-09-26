"""Offline checks of MarginalScorer against the definitions in Giulianelli et al. (2024) and the
pruned algorithm of Vieira et al. (2024), on a toy token-level LM (no downloads needed).

The references below are written independently of the scorer, straight from the definitions:
  * brute-force prefix cover:  ->p(sigma) = sum_{delta in C(sigma)} ->p_Delta(delta)
  * brute-force string prob:   p(sigma)   = sum_{kappa(delta) = sigma} ->p_Delta(delta) p(EOS|delta)
  * Vieira et al.'s enum_cover + prune_top_K_buckets + next_character_probability, recursively.
"""

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from marginal_surprisal import MarginalScorer

# Toy vocabulary over bytes. It has: tokens that straddle a word boundary (b"a b"), leading- and
# trailing-space tokens, a multi-byte character split across tokens (e-acute = C3 A9), a token
# ending mid-character (b"a\xc3"), two ids with the same surface (b"ab"), and special tokens with
# no surface (EOS, PAD).
E = "é".encode()
SURFACES = [
    b"a", b"b", b" ", b"ab", b"ba", b"aa", b" a", b" b", b" ab", b" ba", b" aba", b"bab",
    b"a b", b"b ", b"ab", E, E[:1], E[1:], b" " + E, b"a" + E[:1], E + b"b",
    None,  # EOS
    None,  # PAD (special, never produces text)
]
EOS = len(SURFACES) - 2
PAD = len(SURFACES) - 1
START = EOS


class ToyLM(torch.nn.Module):
    """History-dependent LM (GRU over the whole token history), HF-style interface."""

    def __init__(self, seed: int, temp: float = 2.0, pad_logit: float | None = None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        v, h = len(SURFACES), 16
        self.emb = torch.nn.Embedding(v, h)
        self.rnn = torch.nn.GRU(h, h, batch_first=True)
        self.out = torch.nn.Linear(h, v)
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g, dtype=torch.float64) * temp / math.sqrt(h)
        self.pad_logit = pad_logit
        self.double()

    def get_output_embeddings(self):
        return self.out

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = self.out(self.rnn(self.emb(input_ids))[0])
        if self.pad_logit is not None:
            logits[..., PAD] = self.pad_logit
        return SimpleNamespace(logits=logits)


def make(model, beam_size=None):
    return MarginalScorer(
        model, surfaces=SURFACES, start_id=START, eos_id=EOS, beam_size=beam_size,
        device=torch.device("cpu"),
    )


class Reference:
    """Independent implementation of the paper's quantities (true per-path histories)."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def next_lp(self, hist):
        if hist not in self.cache:
            with torch.no_grad():
                logits = self.model(torch.tensor([[START, *hist]])).logits[0, -1]
            self.cache[hist] = torch.log_softmax(logits.double(), -1).tolist()
        return self.cache[hist]

    def tokens(self):
        return [(t, s) for t, s in enumerate(SURFACES) if s]

    def prefix_prob(self, sigma: bytes) -> float:
        """sum over C(sigma) of ->p_Delta(delta), by depth-first enumeration."""
        if not sigma:
            return 1.0
        total = 0.0

        def dfs(hist, dec, p):
            nonlocal total
            lp = self.next_lp(hist)
            for t, s in self.tokens():
                d2 = dec + s
                if d2.startswith(sigma):  # kappa(hist) < sigma <= kappa(hist + t)
                    total += p * math.exp(lp[t])
                elif sigma.startswith(d2):
                    dfs(hist + (t,), d2, p * math.exp(lp[t]))

        dfs((), b"", 1.0)
        return total

    def string_prob(self, sigma: bytes) -> float:
        """sum over kappa(delta) == sigma of ->p_Delta(delta) * p(EOS | delta)."""
        total = 0.0

        def dfs(hist, dec, p):
            nonlocal total
            lp = self.next_lp(hist)
            if dec == sigma:
                total += p * math.exp(lp[EOS])
                return
            for t, s in self.tokens():
                d2 = dec + s
                if sigma.startswith(d2):
                    dfs(hist + (t,), d2, p * math.exp(lp[t]))

        dfs((), b"", 1.0)
        return total

    # --- Vieira et al. (2024), Algorithm: enum_cover with prune_top_K_buckets -------------

    def enum_cover(self, sigma: bytes, K):
        """Returns the pruned cover of sigma as a list of (logp, decoded, tokens)."""
        if not sigma:
            return [(0.0, b"", ())]
        return self._prune(sigma, self._extend(sigma, self.enum_cover(sigma[:-1], K)), K)

    def _extend(self, sigma, prev):
        n = len(sigma)
        out = []
        for lp, dec, toks in prev:
            if len(dec) >= n:  # last token already runs past sigma[:-1]
                if dec[n - 1] == sigma[n - 1]:
                    out.append((lp, dec, toks))
            else:  # dec == sigma[:-1]: extend with every token
                nl = self.next_lp(toks)
                for t, s in self.tokens():
                    d2 = dec + s
                    if d2[n - 1] == sigma[n - 1]:
                        out.append((lp + nl[t], d2, toks + (t,)))
        return out

    @staticmethod
    def _prune(sigma, out, K):
        if K is None:
            return out
        buckets = defaultdict(list)
        for item in out:
            _, dec, toks = item
            buckets[toks[:-1] if len(dec) > len(sigma) else toks].append(item)
        mass = lambda items: torch.logsumexp(torch.tensor([i[0] for i in items], dtype=torch.float64), 0).item()
        top = sorted(buckets.values(), key=mass, reverse=True)[:K]
        return [item for bucket in top for item in bucket]

    def next_byte_log_prob(self, sigma: bytes, c: int, K) -> float:
        """Vieira et al.'s next_character_probability(sigma)[c], normalised by the pruned cover."""
        cover = self.enum_cover(sigma, K)
        z = torch.logsumexp(torch.tensor([i[0] for i in cover], dtype=torch.float64), 0).item()
        num = self._extend(sigma + bytes([c]), cover)
        if not num:
            return -math.inf
        return torch.logsumexp(torch.tensor([i[0] for i in num], dtype=torch.float64), 0).item() - z


TEXTS = ["ab ba ab", "a bé ba", "é ab éb", "ba b a"]
SEEDS = [0, 1, 2]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("text", TEXTS)
def test_exact_prefix_probs_match_prefix_cover(seed, text):
    model = ToyLM(seed)
    ref = Reference(model)
    lp = make(model).prefix_log_probs(text)
    s = text.encode()
    for i in range(len(s) + 1):  # every byte position, including inside a character
        assert lp[i] == pytest.approx(math.log(ref.prefix_prob(s[:i])), abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_surprisal_is_ratio_of_prefix_probs(seed):
    model = ToyLM(seed)
    ref = Reference(model)
    context, words = "a b", ["ab", "é", "ba"]
    got = make(model).surprisal(context, words)
    text, prev = context, context
    for g, w in zip(got, words):
        text = text + " " + w
        want = math.log(ref.prefix_prob(prev.encode())) - math.log(ref.prefix_prob(text.encode()))
        assert g == pytest.approx(want, abs=1e-9)
        prev = text


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("text", TEXTS)
def test_exact_log_prob_is_string_prob_with_eos(seed, text):
    model = ToyLM(seed)
    ref = Reference(model)
    assert make(model).log_prob(text) == pytest.approx(
        math.log(ref.string_prob(text.encode())), abs=1e-9
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_next_character_distribution_is_normalised(seed):
    """sum_c p(c | sigma) + p(EOS | sigma) = 1 when EOS is the only special token."""
    model = ToyLM(seed, pad_logit=-math.inf)
    scorer = make(model)
    alphabet = sorted({byte for s in SURFACES if s for byte in s})
    for context in [b"", b"a", b"ab ", E[:1], b"b " + E]:  # E[:1]: inside a character
        total = 0.0
        for byte in alphabet:
            try:
                total += math.exp(scorer._run(context + bytes([byte]), with_eos=False)[0][-1])
            except ValueError:  # no token sequence covers it: probability 0
                pass
        eos = scorer._run(context, with_eos=True)[1]
        assert total + math.exp(eos) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_pruned_matches_vieira_et_al(seed, K):
    """With beam_size=K, every next-byte conditional equals Vieira et al.'s pruned algorithm."""
    model = ToyLM(seed)
    ref = Reference(model)
    text = "a bé ab ba"
    s = text.encode()
    want = [ref.next_byte_log_prob(s[:i], s[i], K) for i in range(len(s))]
    if -math.inf in want:
        # A small beam can lose every sequence that continues the string (probability 0 under
        # the pruned model). The scorer reports this instead of returning infinite surprisal.
        dead = want.index(-math.inf)
        with pytest.raises(ValueError, match="beam"):
            make(model, beam_size=K).byte_log_probs(text)
        got = make(model, beam_size=K)._run(s[:dead], with_eos=False)[0]
        want = want[:dead]
    else:
        got = make(model, beam_size=K).byte_log_probs(text)
    assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_large_beam_is_exact(seed):
    model = ToyLM(seed)
    text = "ab ba é"
    exact = make(model).prefix_log_probs(text)
    big = make(model, beam_size=10_000).prefix_log_probs(text)
    assert big == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_marginal_at_least_any_single_tokenization(seed):
    """->p(sigma) >= ->p_Delta(delta) for every delta with kappa(delta) == sigma."""
    model = ToyLM(seed)
    ref = Reference(model)
    text = b"ab ba"
    got = make(model).prefix_log_probs(text.decode())[-1]
    n_paths = 0

    def dfs(hist, dec, lp):
        nonlocal n_paths
        if dec == text:
            n_paths += 1
            assert got >= lp - 1e-12
            return
        nl = ref.next_lp(hist)
        for t, s in ref.tokens():
            if text.startswith(dec + s):
                dfs(hist + (t,), dec + s, lp + nl[t])

    dfs((), b"", 0.0)
    assert n_paths > 1


def test_uncoverable_string_raises():
    model = ToyLM(0)
    with pytest.raises(ValueError):
        make(model).prefix_log_probs("abc")
