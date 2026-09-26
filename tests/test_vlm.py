"""Offline checks that vision-language models work as text-only language models.

Tiny randomly initialised Qwen2.5-VL / Qwen3-VL models are built from transformers' own classes
(no download), and the scorer is checked against the same brute-force references as in
test_paper.py. A small byte-level BPE tokenizer is trained on the fly to check the handling of
VLM special tokens and of the start token.
"""

import math

import pytest
import torch
import transformers

from marginal_surprisal import MarginalScorer
from marginal_surprisal.vocab import special_token_ids, start_token_id, token_surface_bytes

from test_paper import EOS, START, SURFACES, Reference

VOCAB = 320


def _tiny_vlm(kind: str, vocab_size: int = VOCAB):
    text = dict(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32)
    ids = dict(image_token_id=vocab_size - 4, video_token_id=vocab_size - 3,
               vision_start_token_id=vocab_size - 2, vision_end_token_id=vocab_size - 1)
    if kind == "qwen2.5-vl":
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        cfg_cls = getattr(transformers, "Qwen2_5_VLConfig", None)
        text["rope_scaling"] = {"type": "mrope", "mrope_section": [2, 1, 1]}
    else:
        cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        cfg_cls = getattr(transformers, "Qwen3VLConfig", None)
        text.update(head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1],
                                              "mrope_interleaved": True})
        vision.update(num_position_embeddings=16, deepstack_visual_indexes=[])
    if cls is None:
        pytest.skip(f"{kind} not available in transformers {transformers.__version__}")
    torch.manual_seed(0)
    model = cls(cfg_cls(text_config=text, vision_config=vision, **ids)).eval()
    # Sharpen the random head so the toy distribution is not flat.
    with torch.no_grad():
        model.get_output_embeddings().weight.mul_(20)
    return model


KINDS = ["qwen2.5-vl", "qwen3-vl"]


@pytest.fixture(scope="module", params=KINDS)
def vlm(request):
    return _tiny_vlm(request.param)


def _toy_scorer(model, beam_size=None, batch_size=32):
    return MarginalScorer(model, surfaces=SURFACES, start_id=START, eos_id=EOS,
                          beam_size=beam_size, batch_size=batch_size, device=torch.device("cpu"))


def test_padded_batch_equals_single(vlm):
    scorer = _toy_scorer(vlm)
    seqs = [(), (3,), (3, 4, 5, 6), (7, 8)]
    batched = scorer._forward(seqs)
    for seq, lp in zip(seqs, batched):
        assert torch.allclose(lp, scorer._forward([seq])[0], atol=1e-5)


@pytest.mark.parametrize("text", ["ab ba", "a bé"])
def test_exact_matches_prefix_cover_through_vlm(vlm, text):
    ref = Reference(vlm)
    lp = _toy_scorer(vlm).prefix_log_probs(text)
    s = text.encode()
    for i in range(len(s) + 1):
        assert lp[i] == pytest.approx(math.log(ref.prefix_prob(s[:i])), abs=1e-4)


def test_pruned_matches_vieira_through_vlm(vlm):
    ref = Reference(vlm)
    text = "a bé ab"
    s = text.encode()
    got = _toy_scorer(vlm, beam_size=3).byte_log_probs(text)
    want = [ref.next_byte_log_prob(s[:i], s[i], 3) for i in range(len(s))]
    assert got == pytest.approx(want, abs=1e-4)


def test_fewer_logits_than_embeddings_is_handled():
    """Like Molmo 2: extra input embeddings (image tokens) that have no logit are never scored."""
    inner = _tiny_vlm("qwen2.5-vl")

    class TruncatedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def get_input_embeddings(self):
            return inner.get_input_embeddings()

        def forward(self, **kwargs):
            out = inner(**kwargs)
            out.logits = out.logits[..., :16]
            return out

    scorer = _toy_scorer(TruncatedHead())
    assert int(scorer._ids.max()) < 16
    assert math.isfinite(scorer.prefix_log_probs("ab")[-1])


# --------------------------------------------------------------------------- tokenizer handling


@pytest.fixture(scope="module")
def bpe_tokenizer():
    """Byte-level BPE with Qwen-style specials: <|endoftext|>, <|im_end|> (eos), image tokens."""
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.ByteLevelBPETokenizer()
    corpus = ["The cat sat on the mat.", "Il caffè è buono.", "the dog barked at the mailman"] * 50
    tok.train_from_iterator(corpus, vocab_size=300, min_frequency=1,
                            special_tokens=["<|endoftext|>"])
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok._tokenizer,
                                                eos_token="<|im_end|>", pad_token="<|endoftext|>")
    # VLM markers registered as special added tokens that are NOT in the special-tokens map.
    fast.add_tokens(["<|vision_start|>", "<|image_pad|>"], special_tokens=True)
    fast.add_tokens(["<think>"], special_tokens=False)  # a normal added token stays text
    return fast


def test_vlm_special_tokens_have_no_surface(bpe_tokenizer):
    surfaces = token_surface_bytes(bpe_tokenizer)
    for t in ["<|vision_start|>", "<|image_pad|>", "<|endoftext|>", "<|im_end|>"]:
        tid = bpe_tokenizer.convert_tokens_to_ids(t)
        assert tid in special_token_ids(bpe_tokenizer)
        assert surfaces[tid] is None
    assert surfaces[bpe_tokenizer.convert_tokens_to_ids("<think>")] == b"<think>"
    text = "Il caffè è buono. The cat"
    ids = bpe_tokenizer.encode(text, add_special_tokens=False)
    assert b"".join(surfaces[i] for i in ids) == text.encode()


def test_start_token_prefers_processor_then_endoftext(bpe_tokenizer):
    # No bos, tokenizer prepends nothing: <|endoftext|> (Qwen convention), not eos <|im_end|>.
    assert start_token_id(bpe_tokenizer) == bpe_tokenizer.convert_tokens_to_ids("<|endoftext|>")

    class Processor:  # like Molmo 2's: prepends bos-or-eos to text-only input
        def __call__(self, text):
            ids = bpe_tokenizer.encode(text, add_special_tokens=False)
            return {"input_ids": [[bpe_tokenizer.eos_token_id, *ids]]}

    assert start_token_id(bpe_tokenizer, Processor()) == bpe_tokenizer.eos_token_id

    class Broken:
        def __call__(self, text):
            raise TypeError("needs images")

    assert start_token_id(bpe_tokenizer, Broken()) == start_token_id(bpe_tokenizer)


@pytest.mark.parametrize("kind", KINDS)
def test_from_pretrained_loads_vlm(tmp_path, bpe_tokenizer, kind):
    model = _tiny_vlm(kind)
    model.save_pretrained(tmp_path)
    bpe_tokenizer.save_pretrained(tmp_path)
    scorer = MarginalScorer.from_pretrained(str(tmp_path), beam_size=5, dtype=torch.float32)
    assert type(scorer.model).__name__ == type(model).__name__
    assert scorer.start_id == bpe_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    direct = MarginalScorer(model, bpe_tokenizer, beam_size=5)
    text = "The cat sat on the mat."
    assert scorer.prefix_log_probs(text) == pytest.approx(direct.prefix_log_probs(text), abs=1e-4)
    s1, s2 = scorer.surprisal("The cat sat on the", ["mat.", "The"])
    assert math.isfinite(s1) and math.isfinite(s2)
