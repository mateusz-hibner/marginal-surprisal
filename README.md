# marginal-surprisal

Character-level surprisal from Hugging Face causal LMs, computed as in Giulianelli, Malagutti,
Gastaldi, DuSell, Vieira & Cotterell (2024), *On the Proper Treatment of Tokenization in
Psycholinguistics*. It marginalizes over all tokenizations of the string, using Vieira et al.
(2024), *From Language Models over Tokens to Language Models over Characters*, instead of scoring
only the tokenizer's canonical split.

## Install

```bash
uv add git+https://github.com/<user>/marginal-surprisal   # or: uv add --editable ../marginal-surprisal
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from marginal_surprisal import MarginalScorer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
scorer = MarginalScorer(model, tokenizer)  # beam_size=5, as in the paper; None = exact

target, post_target = scorer.surprisal("The cat sat on the", ["mat", "yesterday"])
```

- `surprisal(context, continuations, sep=" ")`: for each region `sep + continuation`, returns
  -log p(region | everything before it), in nats. Regions carry their leading whitespace.
- `prefix_log_probs(text)`: `lp[i]` = log →p(s[:i]), the character-level prefix probability of
  the first `i` bytes.
- `byte_log_probs(text)`: log p(next byte | preceding bytes) for every byte.
- `log_prob(text)`: log p(text) of the complete string, i.e. followed by EOS.

## What is computed

The prefix probability of a character string σ is a sum over its **prefix cover**, the token
sequences whose decoding starts with σ, where only the last token may run past the end of σ:

    →p(σ) = Σ_{δ ∈ C(σ)} →p_Δ(δ),   C(σ) = { δ₁…δₘ : κ(δ₁…δₘ₋₁) ≺ σ ⪯ κ(δ₁…δₘ) }

Each token sequence is scored with the model conditioned on its **own** token history, starting
from the token the tokenizer itself prepends (bos, or eos when it prepends nothing). Without pruning,
the surprisal of a region is exactly `log →p(context) − log →p(context · region)`. So " mat" gets credit from `Ġmat`,
`Ġm·at` and also `Ġmatter`, and the context is not assumed to end on a token boundary.

**Pruning.** The cover grows exponentially with string length. As in the paper (beam size 5), it is
pruned after every character with Vieira et al.'s `prune_top_K_buckets`. Cover members are grouped
by their token sequence without its last token, if that token runs past the current position, or
by the whole sequence if it ends exactly there. The K most probable groups are kept. Next-character
probabilities are normalized by the mass of the pruned cover (Vieira et al.'s
`next_character_probability`), and region surprisals are sums of these. With `beam_size=None`
nothing is pruned and every quantity is exact.

A small beam can drop every sequence that continues the string. The pruned model then gives the
next character probability 0, and the scorer raises `ValueError` rather than returning infinite
surprisal. This is most likely with a nearly uniform model. If it happens, increase `beam_size`.

**Characters are bytes** of the UTF-8 encoding, as in the paper ("Σ … typically bytes"). Byte-level
BPE (GPT-2, Llama-3, Qwen, Pythia, OPT) and SentencePiece byte fallback often split a character such
as "è" across tokens. Region boundaries always fall on character boundaries, where byte- and
character-level prefix probabilities coincide. Several token ids with the same surface are
separate token sequences and are all counted.

**Vocabulary.** Each id is mapped to its surface bytes with the tokenizer's own decoder.
SentencePiece models (Llama-2, XGLM, Mistral, Gemma) see a leading dummy-prefix space, which is
treated as part of the string. Special tokens other than EOS produce no text and are excluded.

**Cost.** Each step runs one batched forward pass for the at most K token sequences that end at
that byte. There is no KV cache, so long texts cost roughly quadratic time.

## Tests

```bash
uv run pytest
```

`tests/test_paper.py` runs offline. It checks the scorer against independent brute-force
implementations of the definitions above (prefix cover, string probability with EOS, normalization
of the next-character distribution) and against a direct transcription of Vieira et al.'s pruned
algorithm, for K = 1, 2, 3, 5. `tests/test_scorer.py` downloads real tokenizers and tiny models.

## Reproducing the paper

The authors released their code and GPT-2 (K = 5) per-character surprisals for Provo, CELER, UCL
and MECO L1 at [rycolab/psycho-toke](https://github.com/rycolab/psycho-toke). Running both
implementations on the same model in float64, this package agrees with their `character_beam2`
to about 1e-10 nats per character. To compare against their released numbers with real GPT-2:

```bash
git clone https://github.com/rycolab/psycho-toke
uv run python scripts/compare_psycho_toke.py psycho-toke --corpus provo --limit 10
```

Their code accumulates log-probabilities in float32, so expect small differences rather than
digit-for-digit equality. On random models these were usually under 0.01 nats per character,
occasionally around 0.1. Their files end each stimulus with an
`<EOS>` entry, which is the surprisal of a following space.
