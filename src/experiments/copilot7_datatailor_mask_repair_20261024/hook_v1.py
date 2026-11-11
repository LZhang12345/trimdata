from contextlib import contextmanager
import copy
from pathlib import Path

from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_v5_20261024 import structured_preprocess as qualified
from experiments.root_native_qualification_v6_20261024.masking import author_v1 as prefix_oracle


CONTRACT = "s7-datatailor-structured-v5-labels-input-identical-ordinary-mask-parity-v1"


@contextmanager
def installed(module, observe, *, independent_qualification_oracle=False):
    """Only a new reviewed runtime may install this reversible in-memory hook."""
    import torch
    expected = c.ROOT / c.NATIVE / "llava/train/train.py"
    c.require(Path(module.__file__).resolve() == expected.resolve(),
              "Exact pinned native preprocessing module required")
    original = module.preprocess_v1
    c.require(Path(original.__code__.co_filename).resolve() == expected.resolve(),
              "Foreign or already hooked native preprocessor")
    from experiments.root_native_qualification_20261023 import masking
    original_oracle = masking.author_v1

    def repaired(sources, tokenizer, has_image=False):
        old = original(copy.deepcopy(sources), tokenizer, has_image=has_image)
        new = qualified.preprocess_structured(module, copy.deepcopy(sources), tokenizer, has_image=has_image)
        c.require(set(old) == set(new) == {"input_ids", "labels"},
                  "Native preprocessing output interface changed")
        c.require(old["input_ids"].dtype == new["input_ids"].dtype
                  and old["labels"].dtype == new["labels"].dtype
                  and torch.equal(old["input_ids"], new["input_ids"]),
                  "Structured repair changed original input IDs or tensor types")
        c.require(old["labels"].shape == new["labels"].shape == old["input_ids"].shape,
                  "Structured repair changed mask/sequence shape")
        original_counts = (old["labels"][:, 1:] != module.IGNORE_INDEX).sum(dim=1).tolist()
        repaired_counts = (new["labels"][:, 1:] != module.IGNORE_INDEX).sum(dim=1).tolist()
        for index, count in enumerate(original_counts):
            if count > 0:
                c.require(torch.equal(old["labels"][index], new["labels"][index]),
                          "Previously valid native assistant mask changed; preserve and diagnose")
        observe(dict(contract=CONTRACT, typed_sources_sha256=c.key(sources),
                     input_ids_identical=True, ordinary_masks_identical=True,
                     native_shifted_target_counts=original_counts,
                     repaired_shifted_target_counts=repaired_counts,
                     labels_changed=[not torch.equal(a, b) for a, b in zip(old["labels"], new["labels"])],
                     positive_expanded_target_guard_unchanged=True))
        return new

    module.preprocess_v1 = repaired
    if independent_qualification_oracle:
        masking.author_v1 = prefix_oracle
    try:
        yield dict(contract=CONTRACT, original_native_source=c.desc(expected),
                   structured_source=c.desc(qualified.__file__),
                   hook_source=c.desc(__file__), training_updates=0,
                   independent_prefix_oracle=independent_qualification_oracle,
                   feature_constructor_or_math_modified=False)
    finally:
        module.preprocess_v1 = original
        masking.author_v1 = original_oracle
