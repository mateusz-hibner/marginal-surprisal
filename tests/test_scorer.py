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


@pytest.fixture(scope="module", params=list(MODELS))
def scorer(request):
    name = MODELS[request.param]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return MarginalScorer(model, tokenizer, device=torch.device("cpu"))


def canonical_log_prob(scorer: MarginalScorer, text: str) -> float:
    ids = [scorer.start_id] + scorer.tokenizer.encode(text, add_special_tokens=False)
    with torch.no_grad():
        logits = scorer.model(torch.tensor([ids])).logits[0, :-1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(1, torch.tensor(ids[1:])[:, None]).sum().item()


@pytest.mark.parametrize("family", list(TOKENIZERS))
def test_surface_strings_round_trip(family):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[family])
    surfaces = token_surface_bytes(tokenizer)
    ids = tokenizer.encode(ACCENTED, add_special_tokens=False)
    pieces = [surfaces[i] for i in ids]
    assert None not in pieces
    joined = b"".join(pieces)
    assert joined in (ACCENTED.encode(), (" " + ACCENTED).encode())


def test_accented_text_has_lattice_path(scorer):
    assert scorer.log_prob(ACCENTED) > -math.inf


@pytest.mark.parametrize("text", ["The cat sat on the mat", ACCENTED])
def test_marginal_at_least_canonical(scorer, text):
    assert scorer.log_prob(text) >= canonical_log_prob(scorer, text) - 1e-4


def test_chain_rule(scorer):
    context, words = "The cat sat", ["on", "the"]
    s1, s2 = scorer.surprisal(context, words)
    full = scorer.log_prob(f"{context} {words[0]} {words[1]}")
    ctx = scorer.log_prob(context)
    assert s1 + s2 == pytest.approx(ctx - full, abs=1e-4)
    assert s1 > 0 and s2 > 0
