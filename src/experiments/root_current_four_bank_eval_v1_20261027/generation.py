"""Retain official loops/audits; validate Root's V4 PEFT bridge for every answer."""
from experiments.root_native_trained_evaluation_v3_20261026 import generation as original
from experiments.root_native_trained_loader_v4_20261026 import generation as bridge
from . import contract as c

AuditedJournal = original.AuditedJournal
validate_entry = original.validate_entry


def native(bank, cfg, rows, journal, loaded, store, current, deadline):
    model = loaded[1]
    before_calls = len(model._dsmm_native_inputs_bridge.records)
    before_answers = len(journal.present)
    result = original.native(bank, cfg, rows, journal, loaded, store, current, deadline)
    proof = bridge.readout(model)
    records = proof["records"][before_calls:]
    c.require(proof["policy"] == bridge.POLICY
              and len(records) == len(journal.present) - before_answers
              and all(record["input_keyword"] == "inputs"
                      and all(record[key] is True for key in
                              ("success", "input_object_unchanged", "all_keyword_objects_unchanged",
                               "prepare_hook_restored", "original_PEFT_generate_invoked",
                               "output_object_unchanged")) for record in records),
              "Every newly accepted answer needs its exact unchanged original PEFT/native bridge call")
    return result
