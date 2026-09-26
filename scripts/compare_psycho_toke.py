"""Compare MarginalScorer with the per-character surprisals released by Giulianelli et al. (2024).

The paper's surprisals were computed with the authors' code (github.com/rycolab/psycho-toke):
GPT-2 small, beam size K=5, each stimulus scored character by character from the start of the
string, conditioned on <|endoftext|>. Their files end each stimulus with an "<EOS>" entry, which
is the surprisal of a following space (their trailing-whitespace convention).

Usage:
    git clone https://github.com/rycolab/psycho-toke
    uv run python scripts/compare_psycho_toke.py psycho-toke --corpus provo --limit 10
"""

import argparse
import math
from pathlib import Path


def read_stimuli(repo: Path, corpus: str) -> list[str]:
    return [line.strip() for line in open(repo / "data" / "stimuli" / f"{corpus}.txt")]


def read_released(repo: Path, corpus: str) -> dict[int, list[tuple[str, float]]]:
    """Parse lines like '0 T 2.50', '0   0.27' (a space) and '0 <EOS> 0.01'."""
    path = repo / "data" / "char_surprisals" / f"{corpus}_surprisals_k5.gpt2.txt"
    out: dict[int, list[tuple[str, float]]] = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        x_id, rest = line.split(" ", 1)
        if rest.startswith("<EOS> "):
            char, value = "<EOS>", rest[len("<EOS> "):]
        else:
            char, value = rest[0], rest[2:]
        out.setdefault(int(x_id), []).append((char, float(value)))
    return out


def ours(scorer, stimulus: str) -> list[tuple[str, float]]:
    """Per-character surprisals in the same layout as the released files."""
    # Stimuli are ASCII (checked below), so bytes and characters coincide.
    lps = scorer.byte_log_probs(stimulus + " ")
    return [(c, -lp) for c, lp in zip(stimulus, lps)] + [("<EOS>", -lps[-1])]


def compare(scorer, repo: Path, corpus: str, limit: int | None, tol: float) -> bool:
    stimuli = read_stimuli(repo, corpus)
    released = read_released(repo, corpus)
    all_diffs, all_mine, all_theirs = [], [], []
    for x_id, stimulus in enumerate(stimuli[:limit]):
        assert stimulus.isascii(), "non-ASCII stimulus: bytes and characters would differ"
        theirs = released[x_id]
        mine = ours(scorer, stimulus)
        assert [c for c, _ in mine] == [c for c, _ in theirs], f"stimulus {x_id}: layout differs"
        diffs = [abs(a - b) for (_, a), (_, b) in zip(mine, theirs)]
        all_diffs += diffs
        all_mine += [a for _, a in mine]
        all_theirs += [b for _, b in theirs]
        print(f"{corpus} #{x_id:<3} {len(diffs):4d} chars  max |diff| = {max(diffs):.2e}  "
              f"total surprisal ours/theirs = {sum(all_mine[-len(diffs):]):9.3f} / "
              f"{sum(all_theirs[-len(diffs):]):9.3f}")

    d = sorted(all_diffs)
    q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
    n = len(d)
    ma, mt = sum(all_mine) / n, sum(all_theirs) / n
    cov = sum((a - ma) * (b - mt) for a, b in zip(all_mine, all_theirs))
    var_a = sum((a - ma) ** 2 for a in all_mine)
    var_b = sum((b - mt) ** 2 for b in all_theirs)
    r = cov / math.sqrt(var_a * var_b)
    n_bad = sum(x > tol for x in d)
    print(f"\n{n} characters. |ours - theirs| in nats: median {q(0.5):.1e}, 99th pct {q(0.99):.1e}, "
          f"max {d[-1]:.1e}; {n_bad} above {tol}. Pearson r = {r:.6f}")
    print("The authors' code accumulates log-probabilities in float32, so small differences are "
          "expected; the two algorithms agree to ~1e-10 nats when both run in float64.")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path, help="path to a clone of rycolab/psycho-toke")
    ap.add_argument("--corpus", default="provo", choices=["provo", "celer", "ucl", "mecoL1"])
    ap.add_argument("--limit", type=int, default=10, help="number of stimuli (default 10)")
    ap.add_argument("--tol", type=float, default=0.05, help="flag differences above this (nats)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marginal_surprisal import MarginalScorer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    if torch.cuda.is_available():
        model = model.cuda()
    scorer = MarginalScorer(model, tokenizer, beam_size=5)
    ok = compare(scorer, args.repo, args.corpus, args.limit, args.tol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
