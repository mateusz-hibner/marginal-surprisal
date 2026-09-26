"""Integration tests with real Hugging Face tokenizers and tiny models (need network access).

The algorithm itself is checked offline against the paper's definitions in test_paper.py.
"""

import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marginal_surprisal import MarginalScorer
from marginal_surprisal.vocab import token_surface_bytes

ACCENTED = "Il caffè è buono"

# (tokenizer family, tiny causal LM)
MODELS = {
    "byte-level BPE": "sshleifer/tiny-gpt2",
    "sentencepiece": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}
TOKENIZERS = {**MODELS, "qwen": "Qwen/Qwen2.5-0.5B"}


def _load(loader, name):
    try:
        return loader.from_pretrained(name)
    except OSError as e:  # offline / hub unreachable
        pytest.skip(f"cannot download {name}: {e}")


@pytest.fixture(scope="module", params=list(MODELS))
def model_and_tokenizer(request):
    name = MODELS[request.param]
    torch.manual_seed(0)
    return _load(AutoModelForCausalLM, name), _load(AutoTokenizer, name)


@pytest.fixture(scope="module")
def scorer(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    # Tiny random models are nearly uniform, the worst case for pruning: with the paper's K=5
    # the beam can lose every sequence that continues the string. Real LMs are far more peaked.
    return MarginalScorer(model, tokenizer, device=torch.device("cpu"), beam_size=20)


def canonical_log_prob(scorer: MarginalScorer, text: str) -> float:
    """->p_Delta of the canonical tokenization (no EOS)."""
    ids = [scorer.start_id] + scorer.tokenizer.encode(text, add_special_tokens=False)
    with torch.no_grad():
        logits = scorer.model(torch.tensor([ids])).logits[0, :-1]
    lp = torch.log_softmax(logits.double(), dim=-1)
    return lp.gather(1, torch.tensor(ids[1:])[:, None]).sum().item()


@pytest.mark.parametrize("family", list(TOKENIZERS))
def test_surface_strings_round_trip(family):
    tokenizer = _load(AutoTokenizer, TOKENIZERS[family])
    surfaces = token_surface_bytes(tokenizer)
    ids = tokenizer.encode(ACCENTED, add_special_tokens=False)
    pieces = [surfaces[i] for i in ids]
    assert None not in pieces
    joined = b"".join(pieces)
    assert joined in (ACCENTED.encode(), (" " + ACCENTED).encode())


def test_accented_text_is_covered(scorer):
    lp = scorer.prefix_log_probs(ACCENTED)
    assert all(math.isfinite(x) for x in lp)


def test_exact_prefix_prob_at_least_canonical(model_and_tokenizer):
    """The canonical tokenization is one member of the prefix cover."""
    model, tokenizer = model_and_tokenizer
    exact = MarginalScorer(model, tokenizer, device=torch.device("cpu"), beam_size=None)
    text = "The cat"
    assert exact.prefix_log_probs(text)[-1] >= canonical_log_prob(exact, text) - 1e-6


def test_surprisal_is_finite_and_positive(scorer):
    s1, s2 = scorer.surprisal("The cat sat", ["on", "the"])
    assert 0 < s1 < math.inf and 0 < s2 < math.inf


def test_log_prob_includes_eos(scorer):
    text = "The cat"
    assert scorer.log_prob(text) < scorer.prefix_log_probs(text)[-1]
