"""Sanity-check MarginalScorer on a real checkpoint before using it for an experiment.

    uv run python scripts/check_model.py gpt2
    uv run python scripts/check_model.py Qwen/Qwen2.5-VL-7B-Instruct --device-map auto
    uv run python scripts/check_model.py allenai/Molmo2-8B --trust-remote-code --device-map auto

Checks, in order:
  1. loading: model class, start token, vocabulary / logit sizes, excluded special tokens;
  2. vocabulary: the tokenizer's own tokenization of sample texts decodes back byte-exactly;
  3. padding: a right-padded batch gives the same next-token distribution as single sequences;
  4. correctness: exact mode equals a brute-force sum over the prefix cover on short strings;
  5. beam: surprisals with K=5 (the paper's setting) vs K=20 on sample sentences, and timing.
Precision matters: bfloat16 weights give logits with ~3 significant digits, so differences of a
few hundredths of a nat in checks 3-5 are rounding, not bugs. Use --dtype float32 to rule that out.
"""

import argparse
import math
import time

import torch

from marginal_surprisal import MarginalScorer
from marginal_surprisal.vocab import special_token_ids

SAMPLES = [
    "The cat sat on the mat.",
    "Yesterday, the old man's dog barked at the mailman for an hour.",
    "Il caffè è buono, naïve café — “quoted”.",
]
SENTENCES = [
    ("The cat sat on the", ["mat.", "Then", "it", "slept."]),
    ("After the long meeting, the manager finally", ["agreed", "to", "the", "proposal."]),
]


def status(ok: bool, warn: bool = False) -> str:
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def brute_force_prefix_log_prob(scorer: MarginalScorer, sigma: bytes) -> float:
    """log sum over the prefix cover of sigma, every sequence scored with its own history."""
    surfaces = list(zip(scorer._surfaces, scorer._ids.tolist()))
    terms = []

    def dfs(hist, dec, lp):
        nl = scorer._forward([hist])[0]
        for s, t in surfaces:
            d2 = dec + s
            if d2.startswith(sigma):
                terms.append(lp + nl[t].item())
            elif sigma.startswith(d2):
                dfs(hist + (t,), d2, lp + nl[t].item())

    dfs((), b"", 0.0)
    return torch.logsumexp(torch.tensor(terms, dtype=torch.float64), 0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="auto", help="auto, float32, bfloat16, float16")
    ap.add_argument("--device-map", default=None, help='e.g. "auto" to use the GPU(s)')
    args = ap.parse_args()
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)

    print(f"== 1. Loading {args.model}")
    t0 = time.time()
    scorer = MarginalScorer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, dtype=dtype,
        device_map=args.device_map, beam_size=5,
    )
    tok = scorer.tokenizer
    n_logits = scorer._forward([()])[0].shape[-1]
    print(f"   {type(scorer.model).__name__}, dtype {next(scorer.model.parameters()).dtype}, "
          f"device {scorer.device}, loaded in {time.time() - t0:.0f}s")
    print(f"   start token: {tok.convert_ids_to_tokens(scorer.start_id)!r} (id {scorer.start_id}); "
          f"dummy-prefix space: {scorer.dummy_prefix}")
    print(f"   tokenizer size {len(tok)}, logits {n_logits}, tokens with a text surface "
          f"{len(scorer._surfaces)}, special tokens excluded {len(special_token_ids(tok))}")

    print("\n== 2. Vocabulary round trip")
    surf = dict(zip(scorer._ids.tolist(), scorer._surfaces))
    all_ok = True
    for text in SAMPLES:
        ids = tok.encode(text, add_special_tokens=False)
        missing = [i for i in ids if i not in surf]
        joined = b"".join(surf.get(i, b"?") for i in ids)
        ok = not missing and joined in (text.encode(), (" " + text).encode())
        all_ok &= ok
        print(f"   {status(ok)}  {text!r}" + (f"  (tokens without surface: {missing})" if missing else ""))

    print("\n== 3. Padded batch vs single sequences")
    ids = tok.encode(SAMPLES[1], add_special_tokens=False)
    seqs = [(), tuple(ids[:1]), tuple(ids[:7]), tuple(ids[:3])]
    batched = scorer._forward(seqs)
    diff = max((b - scorer._forward([s])[0]).abs().max().item() for s, b in zip(seqs, batched))
    print(f"   {status(diff < 1e-3, warn=diff < 0.1)}  max |log p difference| = {diff:.1e}")

    print("\n== 4. Exact mode vs brute-force prefix cover")
    exact = MarginalScorer(scorer.model, tok, start_id=scorer.start_id, beam_size=None)
    for text in [" the", " cat."]:
        s = exact._internal(text)
        got = exact.prefix_log_probs(text)[-1]
        want = brute_force_prefix_log_prob(exact, s)
        d = abs(got - want)
        print(f"   {status(d < 1e-3, warn=d < 0.05)}  {text!r}: scorer {got:.4f}, "
              f"brute force {want:.4f}, |diff| {d:.1e}")

    print("\n== 5. Beam size: K=5 (paper) vs K=20")
    k20 = MarginalScorer(scorer.model, tok, start_id=scorer.start_id, beam_size=20)
    for context, words in SENTENCES:
        try:
            t0 = time.time()
            a = scorer.surprisal(context, words)
            t5 = time.time() - t0
        except ValueError as e:
            print(f"   FAIL  K=5 beam lost the string: {e}")
            continue
        b = k20.surprisal(context, words)
        d = max(abs(x - y) for x, y in zip(a, b))
        print(f"   {status(d < 0.05, warn=d < 0.5)}  {context!r} + {words}")
        print(f"         K=5 {[round(x, 3) for x in a]}  ({t5:.1f}s)")
        print(f"         K=20 {[round(x, 3) for x in b]}  max |diff| {d:.3f} nats")


if __name__ == "__main__":
    main()
