"""Freeze the reviewed V2 engine with strict Root continuation-plan bindings."""
import os
from pathlib import Path
import re
import socket
import subprocess
import types

from . import consumer_v2 as engine


c = engine.c


def rows(descriptor, identity, roster, **kwargs):
    values = engine.rows(descriptor, identity, roster, **kwargs)
    c.require(all(type(value["length"]) is int and value["length"] > 0 for value in values.values()),
              "Native sequence lengths must be positive integers")
    return values


def identity_components(original):
    return dict(engine.identity_components(original), continuation_consumer_source=c.sha(__file__),
                inherited_v2_source=c.sha(engine.__file__),
                inherited_v1_source=c.sha(engine.prior.__file__))


def overlay_identity(original):
    return c.key(identity_components(original))


def validate_plan(config, bound, legacy_refs, output, original):
    c.require(bound["measurement_sources"] == config["sources"],
              "Supplied native measurement source map differs from Root CONFIG")
    plan = bound["composite_shards"][str(config["shard_index"])]
    c.require(plan["original_measurement"] == original
              and plan["legacy_databases"] == legacy_refs
              and str(Path(plan["overlay_output"]).absolute()) == str(Path(output).absolute()),
              "Root must bind the original function, every retained DB and one overlay per global shard")


def _engine_function(function):
    globals_map = dict(vars(engine), rows=rows, identity_components=identity_components,
                       overlay_identity=overlay_identity, load_complete=load_complete)
    result = types.FunctionType(function.__code__, globals_map, function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def acquire(config, pool, records, ids, legacy_refs, parity_ref, review_ref,
            native_module, output, attempt, should_stop, *, bound_configuration):
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                  text=True, timeout=30)
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    c.require(fields["JobId"] == os.environ["SLURM_JOB_ID"] and fields["JobState"] == "RUNNING"
              and fields["Partition"] == "general" and fields["QOS"] == "normal"
              and fields["Account"] == "cx" and fields["UserId"].split("(")[0] == "luyangz"
              and fields["NodeList"] == socket.gethostname().split(".")[0]
              and fields["TresPerNode"] == "gres/gpu:L40S:1",
              "Root-owned actual General allocation required before continuation-plan DATA")
    bound = c.checked(bound_configuration)
    original = engine.native.identity(config, pool)
    validate_plan(config, bound, legacy_refs, output, original)
    return _engine_function(engine.acquire)(config, pool, records, ids, legacy_refs, parity_ref, review_ref,
        native_module, output, attempt, should_stop, bound_configuration=bound_configuration)


def load_complete(manifest_ref, roster, expected_original):
    return _engine_function(engine.load_complete)(manifest_ref, roster, expected_original)


def load_full_pool(manifest_refs, ids, expected_pool, expected_original):
    return _engine_function(engine.load_full_pool)(manifest_refs, ids, expected_pool, expected_original)
