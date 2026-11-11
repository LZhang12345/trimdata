"""Derive an explicit smaller random control without mutating a paid selection."""
import random


def derive(indices, group_ids, global_group_sizes, *, population_n, input_k, output_k, seed):
    if (type(indices) is not list or type(group_ids) is not list
            or type(global_group_sizes) is not dict
            or type(population_n) is not int or population_n <= 0
            or type(input_k) is not int or type(output_k) is not int
            or not 0 < output_k < input_k <= population_n
            or len(indices) != input_k or len(group_ids) != input_k
            or type(seed) is not int):
        raise ValueError('Exact population, input/output budgets, groups and independent seed are required')
    if any(type(index) is not int or not 0 <= index < population_n for index in indices):
        raise ValueError('Selected indices must be valid population indices')
    if len(set(indices)) != input_k:
        raise ValueError('Paid input selection must contain unique indices')
    counts = {}
    for group in group_ids:
        if type(group) is not str or not group:
            raise ValueError('Every selected conversation requires a canonical group ID')
        counts[group] = counts.get(group, 0) + 1
    for group, count in counts.items():
        expected = global_group_sizes.get(group)
        if type(expected) is not int or expected <= 0 or count != expected:
            raise ValueError('Paid input selection splits a construction or alias group')
    drop = set(random.Random(seed).sample(range(input_k), input_k - output_k))
    remaining = {}
    for position, group in enumerate(group_ids):
        if position not in drop:
            remaining[group] = remaining.get(group, 0) + 1
    if any(count != global_group_sizes[group] for group, count in remaining.items()):
        raise ValueError('Derived random selection splits a construction or alias group')
    selected = [index for position, index in enumerate(indices) if position not in drop]
    if len(selected) != output_k or len(set(selected)) != output_k:
        raise ValueError('Derived selection does not meet the exact budget')
    return {
        'schema': 'tool1-uniform-matched-k-v1',
        'method': 'uniform-random-subset',
        'population_n': population_n,
        'input_k': input_k,
        'selected_k': output_k,
        'derivation_seed': seed,
        'selected_indices': selected,
        'source_uniformity_and_independence_require_external_proof': True,
        'not_a_model_or_scientific_qualification': True,
    }
