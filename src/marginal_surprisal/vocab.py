import re

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_SP_SPACE = "▁"
_SP_BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _byte_level_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: printable stand-in character -> raw byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_level_decoder()


def is_byte_level(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Whether token strings use GPT-2-style byte-to-unicode encoding (Ġ, Ã¨, ...)."""
    probe = " è"
    tokens = "".join(tokenizer.tokenize(probe))
    return all(c in _BYTE_DECODER for c in tokens) and (
        bytes(_BYTE_DECODER[c] for c in tokens) == probe.encode()
    )


def token_surface_bytes(tokenizer: PreTrainedTokenizerBase) -> list[bytes | None]:
    """Return the raw UTF-8 bytes each token id stands for, or None if it has no surface.

    Working in bytes rather than characters matters: byte-level BPE (GPT-2, Llama-3, Qwen) and
    SentencePiece byte fallback often split a multi-byte character such as "è" across tokens,
    and the canonical tokenization itself may use such partial-character tokens.

    None is returned for special tokens and tokens with an empty surface. Special tokens include
    added tokens flagged as special (chat and image markers such as <|im_start|> or
    <|image_pad|> in VLM tokenizers), which never stand for text.
    """
    n = len(tokenizer)
    ids = list(range(n))
    special = special_token_ids(tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    byte_level = is_byte_level(tokenizer)
    decoded = tokenizer.batch_decode(
        [[i] for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    surfaces: list[bytes | None] = []
    for i, tok, text in zip(ids, tokens, decoded):
        if i in special or tok is None:
            surfaces.append(None)
            continue
        if byte_level and all(c in _BYTE_DECODER for c in tok):
            surface = bytes(_BYTE_DECODER[c] for c in tok)
        elif m := _SP_BYTE.fullmatch(tok):
            surface = bytes([int(m.group(1), 16)])
        else:
            # SentencePiece decoders strip the leading space of the first token in a
            # sequence, which for a single-token decode is the token's own space.
            if tok.startswith(_SP_SPACE) and not text.startswith(" "):
                text = " " + text
            if "�" in text:
                surfaces.append(None)
                continue
            surface = text.encode()
        surfaces.append(surface or None)
    return surfaces


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Ids of special tokens: the tokenizer's special tokens plus added tokens marked special."""
    special = set(tokenizer.all_special_ids)
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special.update(i for i, tok in added.items() if getattr(tok, "special", False))
    return special


def has_dummy_prefix(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Whether the tokenizer silently prepends a space to the input (SentencePiece dummy prefix)."""
    tokens = tokenizer.tokenize("a")
    return bool(tokens) and tokens[0].startswith(_SP_SPACE)


def start_token_id(tokenizer: PreTrainedTokenizerBase, processor=None) -> int:
    """Id the model conditions on at the start of a text.

    In order of preference:
      1. whatever the model's processor prepends to a text-only input (Molmo 2 inserts bos/eos);
      2. whatever the tokenizer prepends (<s> for Llama, </s> for XGLM/OPT);
      3. the bos token (GPT-2, Pythia);
      4. <|endoftext|>, the document separator of tokenizers without bos (Qwen, incl. Qwen-VL);
      5. the eos token.
    Pass start_id to MarginalScorer to override.
    """
    bare = tokenizer.encode("a", add_special_tokens=False)
    candidates = []
    if processor is not None:
        try:
            out = processor(text="a")
            ids = out["input_ids"]
            ids = ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids
            candidates.append([int(i) for i in ids])
        except Exception:  # processors differ; fall back to the tokenizer
            pass
    candidates.append(tokenizer.encode("a", add_special_tokens=True))
    for with_special in candidates:
        for k in range(len(with_special) - len(bare) + 1):
            if with_special[k : k + len(bare)] == bare:
                if k > 0:
                    return with_special[k - 1]
                break

    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(endoftext, int) and endoftext != tokenizer.unk_token_id:
        return endoftext
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer has no bos or eos token; pass start_id explicitly.")
