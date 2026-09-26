# marginal-surprisal

Word surprisal from Hugging Face causal LMs, marginalized over **all tokenizations** of the
string (in the spirit of Giulianelli et al., 2024, *On the Proper Treatment of Tokenization in
Psycholinguistics*), rather than over the tokenizer's single canonical split.

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
scorer = MarginalScorer(model, tokenizer)  # build once per model; the vocab scan is cached

target, post_target = scorer.surprisal("The cat sat on the", ["mat", "yesterday"])
```

`surprisal(context, continuations, sep=" ")` returns, for each continuation, the surprisal in
nats of that word given the context and all earlier continuations. `log_prob(text)` returns the
marginal log-probability of a string.

## How it works

- Each vocabulary id is mapped to its surface string with the tokenizer's own decoder, so this
  works with byte-level BPE (GPT-2, Llama-3, Qwen, Pythia, OPT) and SentencePiece (Llama-2,
  XGLM, Mistral, Gemma) alike, and with non-ASCII text.
- A character lattice holds every token matching every span of the string. A forward DP sums
  the probabilities of all paths in log space.
- Every prefix is conditioned on the start token the tokenizer itself prepends (bos, or eos
  when it prepends nothing).

**Approximation:** the next-token distribution at character position *j* is conditioned on the
*canonical* tokenization of the prefix, not on the path the DP took to reach *j*.

**Limitation:** tokens that decode to partial UTF-8 bytes (byte-fallback pieces) are skipped.
A string whose characters can only be built from such tokens raises `ValueError`.

## Tests

```bash
uv run pytest
```
