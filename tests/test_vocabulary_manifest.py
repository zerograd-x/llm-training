from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from llm_training import ArtifactManifest, expand_vocabulary


def make_base(path, tied=True):
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "question",
        "answer",
        "red",
        "blue",
        "green",
        "one",
        "two",
        "three",
        "four",
    ]
    raw = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    torch.manual_seed(123)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            bos_token_id=1,
            pad_token_id=0,
            tie_word_embeddings=tied,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, tokenizer


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("initialization", ["mean", "random"])
def test_vocabulary_identity_and_old_rows(tmp_path, tied, initialization):
    original, tokenizer = make_base(tmp_path / "base", tied)
    before = torch.get_rng_state()
    result = expand_vocabulary(
        tmp_path / "base",
        tmp_path / "expanded",
        ["red", "<new_a>", "<new_a>", "<new_b>"],
        initialization=initialization,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert result.n_tokens_added == 2 and sorted(result.token_ids.values()) == [12, 13]
    model = AutoModelForCausalLM.from_pretrained(result.model_path)
    for old, new in [
        (original.get_input_embeddings(), model.get_input_embeddings()),
        (original.get_output_embeddings(), model.get_output_embeddings()),
    ]:
        torch.testing.assert_close(old.weight, new.weight[:12], atol=0, rtol=0)
    loaded = AutoTokenizer.from_pretrained(result.model_path)
    assert all(
        loaded.get_vocab()[token] == index
        for token, index in tokenizer.get_vocab().items()
    )
    ArtifactManifest.load(result.model_path).validate(
        result.model_path, base_path=tmp_path / "base"
    )
    with torch.no_grad():
        original.get_input_embeddings().weight.add_(1)
    original.save_pretrained(tmp_path / "wrong-base")
    with pytest.raises(ValueError, match="Base model fingerprint"):
        ArtifactManifest.load(result.model_path).validate(
            result.model_path, base_path=tmp_path / "wrong-base"
        )
    (Path(result.model_path) / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        ArtifactManifest.load(result.model_path).validate(result.model_path)
