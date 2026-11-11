import json
import math
from pathlib import Path

from . import common as c


def scalar(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    raise ValueError("Unsupported identity JSON scalar type: " + type(value).__name__)


def type_name(value):
    cls = type(value)
    return cls.__name__ if cls.__module__ == "builtins" else cls.__module__ + "." + cls.__name__


def json_domain(value, path="$", conversions=None):
    if conversions is None:
        conversions = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            c.require(isinstance(key, (str, int)) and not isinstance(key, bool),
                      "Invalid identity mapping-key type at " + path)
            wire_key = str(scalar(key))
            c.require(wire_key not in result, "Colliding identity mapping keys at " + path)
            if isinstance(key, int):
                conversions.append(dict(path=path, kind="integer_mapping_key",
                                        Python_type=type_name(key), Python_value=int.__int__(key),
                                        JSON_key=wire_key))
            result[wire_key] = json_domain(item, path + "/" + wire_key, conversions)
        return result
    if isinstance(value, (list, tuple)):
        if isinstance(value, tuple):
            conversions.append(dict(path=path, kind="tuple_to_JSON_array",
                                    Python_type="tuple", length=len(value)))
        return [json_domain(item, path + "/" + str(index), conversions)
                for index, item in enumerate(value)]
    c.require(value is None or isinstance(value, (str, bool, int, float)),
              "Invalid identity value type at " + path)
    result = scalar(value)
    c.require(not isinstance(result, float) or math.isfinite(result), "Nonfinite identity value at " + path)
    if type(result) is not type(value):
        conversions.append(dict(path=path, kind="JSON_scalar_subclass", Python_type=type_name(value),
                                JSON_type=type(result).__name__))
    return result


def typed_tree(value):
    if isinstance(value, dict):
        return dict(type="dict", entries=[
            dict(key=typed_tree(key), value=typed_tree(item)) for key, item in value.items()])
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, items=[typed_tree(item) for item in value])
    if value is None or isinstance(value, (str, bool, int)):
        return dict(type=type_name(value), value=scalar(value))
    if isinstance(value, float):
        result = scalar(value)
        return dict(type=type_name(value), value=result if math.isfinite(result) else repr(result),
                    finite=math.isfinite(result))
    return dict(type=type(value).__module__ + "." + type(value).__name__, supported=False)


def structural_diff(expected, actual, path="$"):
    if type(expected) is not type(actual):
        return [dict(path=path, kind="type", expected_type=type(expected).__name__,
                     actual_type=type(actual).__name__, expected=expected, actual=actual)]
    if isinstance(expected, dict):
        changes = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                changes.append(dict(path=path + "/" + key, kind="missing", expected=expected[key]))
            elif key not in expected:
                changes.append(dict(path=path + "/" + key, kind="extra", actual=actual[key]))
            else:
                changes.extend(structural_diff(expected[key], actual[key], path + "/" + key))
        return changes
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [dict(path=path, kind="length", expected=len(expected), actual=len(actual))]
        return [change for index, (left, right) in enumerate(zip(expected, actual))
                for change in structural_diff(left, right, path + "/" + str(index))]
    return [] if expected == actual else [dict(path=path, kind="value", expected=expected, actual=actual)]


def persist_and_compare(live, expected, output, expected_commitment):
    output = Path(output)
    types_ref = c.seal(output / "ACTUAL_MODEL_IDENTITY_PYTHON_TYPES.json",
                      dict(schema="session6-complete-pre-normalization-identity-tree-v6",
                           actual=typed_tree(live), fields_dropped=[]))
    conversions = []
    actual = json_domain(live, conversions=conversions)
    c.require(actual == json.loads(c.canonical(live)),
              "Identity JSON-domain conversion disagrees with original producer serialization")
    actual_ref = c.seal(output / "ACTUAL_MODEL_IDENTITY.json", actual)
    changes = structural_diff(expected, actual)
    components = {}
    for key in sorted(set(expected) | set(actual)):
        components[key] = dict(
            expected_sha256=c.key(expected[key]) if key in expected else None,
            actual_sha256=c.key(actual[key]) if key in actual else None,
            differences=[change for change in changes
                         if change["path"] == "$/" + key or change["path"].startswith("$/" + key + "/")])
    comparison = c.seal(output / "ACTUAL_MODEL_IDENTITY_COMPARISON.json", dict(
        schema="session6-strict-JSON-domain-model-identity-comparison-v6",
        expected_identity_sha256=expected_commitment, actual_identity=actual_ref,
        actual_identity_sha256=c.key(actual), pre_normalization_types=types_ref,
        original_producer_JSON_conversions=conversions,
        Python_equal_before_persistence=live == expected,
        exact_JSON_domain_equal=not changes, component_comparisons=components,
        structural_differences=changes, fields_dropped=[], comparison_relaxed=False,
        NN_forward_permitted=not changes and c.key(actual) == expected_commitment))
    c.require(not changes and actual == expected and c.key(actual) == expected_commitment,
              "Actual inference model/runtime differs from qualified START; full identity and component diff saved: "
              + comparison["path"])
    return actual
