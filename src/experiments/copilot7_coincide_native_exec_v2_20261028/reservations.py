"""Existing measured storage grants only; paid features and old states untouched."""
from . import protocol as p

PER_CELL = 21 << 30


def function_identity(cfg, pool, selected):
    return p.key(dict(recipe=cfg["recipe"], pool_records=pool["records_json"]["sha256"],
        selected_records=selected["records_json"]["sha256"], training_seed=cfg["training_seed"],
        role="student", qualification_max_steps=None, replicate_site="babel"))


def validate(grant, cfg, pool, selected):
    function = function_identity(cfg, pool, selected)
    p.require(grant["status"] == "PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION"
              and grant["owner"] == "Root" and grant["output"] == cfg["output"]
              and grant["cell_id"] == cfg["id"] and grant["function_identity"] == function
              and type(grant["reserved_bytes"]) is int and grant["reserved_bytes"] == PER_CELL
              and grant["checkpoint_transaction_bytes"] == p.CAPSULE_BYTES
              and grant["retained_accepted_states"] == 2
              and grant["allow_new_intermediate_retirement"] is True
              and type(grant["other_outstanding_reserved_bytes"]) is int
              and grant["other_outstanding_reserved_bytes"] >= 0,
              "Actual separate persistent21GiB grant for this COINCIDE output/seed/function")
    return function
