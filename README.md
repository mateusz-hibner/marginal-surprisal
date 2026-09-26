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
from marginal_surprisal import MarginalScorer

scorer = MarginalScorer.from_pretrained("gpt2")  # beam_size=5, as in the paper; None = exact

target, post_target = scorer.surprisal("The cat sat on the", ["mat", "yesterday"])
```

An already loaded model works too: `MarginalScorer(model, tokenizer)`.

- `surprisal(context, continuations, sep=" ")`: for each region `sep + continuation`, returns
  -log p(region | everything before it), in nats. Regions carry their leading whitespace.
- `prefix_log_probs(text)`: `lp[i]` = log →p(s[:i]), the character-level prefix probability of
  the first `i` bytes.
- `byte_log_probs(text)`: log p(next byte | preceding bytes) for every byte.
- `log_prob(text)`: log p(text) of the complete string, i.e. followed by EOS.

## Vision-language models (text only)

VLMs are used as text-only language models: no image is passed, and their image and chat
tokens are excluded from the vocabulary. `from_pretrained` loads them with their
image-text-to-text class.

```python
scorer = MarginalScorer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True,
                                        device_map="auto")
scorer = MarginalScorer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto")
scorer = MarginalScorer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", device_map="auto")
```

- **transformers version.** Molmo 2's custom code needs transformers 4.57.x (`<5`), which also
  supports Qwen2.5-VL and Qwen3-VL. The lock file pins 4.57.1. The package itself works with
  4.57 and 5.x.
- **Start token.** Every text is conditioned on what the model's own processor prepends to text:
  for Molmo 2 that is its bos token, which Ai2's converter sets equal to eos. For tokenizers that add nothing and have no bos, such as
  Qwen-VL, it is `<|endoftext|>`, Qwen's document separator. Override with `start_id=`.
- **Precision.** `dtype="auto"` loads bfloat16 weights for most VLMs. Their logits carry about
  3 significant digits, so surprisals are only accurate to a few hundredths of a nat. Pass
  `dtype=torch.float32` if that matters and memory allows.
- **Check each new model** with `scripts/check_model.py` (below) before an experiment.

## What is computed

The prefix probability of a character string σ is a sum over its **prefix cover**, the token
sequences whose decoding starts with σ, where only the last token may run past the end of σ:

    →p(σ) = Σ_{δ ∈ C(σ)} →p_Δ(δ),   C(σ) = { δ₁…δₘ : κ(δ₁…δₘ₋₁) ≺ σ ⪯ κ(δ₁…δₘ) }

Each token sequence is scored with the model conditioned on its **own** token history, starting
from the model's start token (see above). Without pruning,
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
treated as part of the string. Special tokens produce no text and are excluded, including added
tokens marked special (VLM image and chat markers). Tokens that have an input embedding but no
logit, such as Molmo 2's image tokens, are never scored.

**Cost.** Each step runs one batched forward pass for the at most K token sequences that end at
that byte. There is no KV cache, so long texts cost roughly quadratic time.

## Tests

```bash
uv run pytest
```

`tests/test_paper.py` runs offline. It checks the scorer against independent brute-force
implementations of the definitions above (prefix cover, string probability with EOS, normalization
of the next-character distribution) and against a direct transcription of Vieira et al.'s pruned
algorithm, for K = 1, 2, 3, 5. `tests/test_vlm.py` also runs offline. It repeats those checks through
tiny random Qwen2.5-VL and Qwen3-VL models, and checks padding, special-token handling, the start
token and `from_pretrained`. `tests/test_scorer.py` downloads real tokenizers and tiny models.

### Checking a real model

```bash
uv run python scripts/check_model.py allenai/Molmo2-8B --trust-remote-code --device-map auto
```

This loads the checkpoint and reports its start token and vocabulary. It then checks that the
tokenizer's splits decode back to the text exactly, that padded batches match single sequences,
and that exact mode equals a brute-force sum over the prefix cover. Finally it compares K = 5 with
K = 20 on sample sentences. Differences of a few hundredths of a nat in bfloat16 are rounding;
use `--dtype float32` to rule that out.

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
