"""Only the actual accepted COINCIDE set and its three declared paired seeds."""
from . import protocol as p
from .registry import registered_plan


def get_cell(plan, identifier, load=p.checked):
    p.require(plan["schema"] == "coincide-fixed10-native-v2-plan-v1" and plan["owner"] == "Root"
              and plan["sources"] == p.n.sources() and plan["adapter_sources"] == p.sources()
              and plan["recipe"] == p.native_config.RECIPE,
              "Exact new COINCIDE native plan and unchanged mathematical source")
    registry = registered_plan(plan["accepted_registry"], load)
    p.require([row["id"] for row in plan["cells"]] == [p.cell_id("coincide", s) for s in p.SEEDS],
              "No other budgets, methods or seeds in this finite plan")
    rows = [row for row in plan["cells"] if row["id"] == identifier]
    original = [row for row in registry["cells"] if row["id"] == identifier]
    p.require(len(rows) == len(original) == 1
              and all(rows[0][name] == original[0][name] for name in
                      ("method", "training_seed", "selection", "new_training_required", "duplicate_of")),
              "No selected-set rebind or duplicate trajectory")
    return rows[0]
