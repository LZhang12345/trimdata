"""Reuse the actual-qualified prefix oracle, independent of V4 round sums."""
from experiments.root_native_qualification_v6_20261024.masking import author_v1

CONTRACT = "qualified-full-prefix-native-v1-independent-oracle-v6"
REPAIRED_ID = "642651:djWVD94_0"
RECORDS_SHA256 = "3cf04ac5cb0b74c39de779ef94250bb4a3fb7d27fc09eb36ff6f16d4a9db3e90"
PREFIX_SOURCE = "experiments/root_native_qualification_v6_20261024/masking.py"


def structured_v1(row, tokenizer, template):
    result = author_v1(row, tokenizer, template)
    return dict(result, oracle=CONTRACT, role_boundaries_from_structured_turns=True,
                qualified_prefix_source=PREFIX_SOURCE)
