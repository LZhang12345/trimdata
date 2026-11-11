"""Unchanged c121 evaluation loops with semantic-row durable answer IO."""
import builtins
import copy
import importlib
import io
import json
from pathlib import Path
import time
import types

from experiments.session6_official_eval_v2_20261024 import common as c
from experiments.session6_start_evaluation_v1_20261024.model import generation_configuration

MODULES = {
    "loader": "llava.eval.model_vqa_loader",
    "freeform": "llava.eval.model_vqa",
    "scienceqa": "llava.eval.model_vqa_science",
    "mmbench": "llava.eval.model_vqa_mmbench",
}


class AcceptedQABoundary(Exception):
    pass


class Sink:
    def __init__(self, rows, journal, audit):
        self.rows, self.journal, self.audit = rows, journal, audit
        self.position, self.pending = 0, ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            c.require(line and self.position < len(self.rows), "Unexpected native answer count")
            row = self.rows[self.position]
            audit = dict(copy.deepcopy(self.audit()), semantic_key=row["semantic_key"])
            self.journal.append(row["semantic_key"], json.loads(line), audit)
            self.position += 1
        return len(text)

    def flush(self):
        return None

    def close(self):
        c.require(not self.pending, "Native evaluator stopped inside an uncommitted raw answer")


def evaluate(kind, args, roster, journal, loaded, deadline, source_descriptors, audit, stop_requested=None):
    c.require(kind in MODULES and args.temperature == 0
              and args.num_chunks == 1 and args.chunk_idx == 0
              and args.conv_mode == "vicuna_v1",
              "Frozen native greedy single-chunk generation required")
    if kind != "scienceqa":
        c.require(type(args.num_beams) is int and args.num_beams == 1,
                  "Pinned native explicit beam argument changed")
    c.require(kind != "mmbench" or args.all_rounds is False,
              "Cyclic MMBench needs a separately qualified semantic-round adapter")
    original = importlib.import_module(MODULES[kind])
    c.require(c.desc(original.__file__) == source_descriptors[MODULES[kind]],
              "Native generation source changed")
    model_name = original.get_model_name_from_path(args.model_path)
    c.require(not ("plain" in model_name and "finetune" not in model_name.lower()),
              "Native auto_mmtag requires a separately frozen effective-template policy")
    if kind == "loader":
        c.require(args.max_new_tokens == 128, "Pinned loader token horizon changed")
    if kind in ("scienceqa", "mmbench"):
        c.require(args.single_pred_prompt is True, "Pinned option-only instruction changed")
    c.require(len(loaded) == 4, "Actual loaded native model tuple required")
    model = loaded[1]
    generation = generation_configuration(model)
    c.require(journal.function.get("model_generation_config") == generation,
              "Actual model generation_config differs from the journal function binding")
    beams = generation["num_beams"] if kind == "scienceqa" else args.num_beams
    c.require(beams == 1, "Inherited native generation must satisfy the frozen greedy policy")
    binding = dict(model_generation_config_sha256=c.key(generation),
                   native_num_beams_keyword_present=kind != "scienceqa",
                   effective_num_beams=beams)
    missing = [row for row in roster if row["semantic_key"] not in journal.present]
    if not missing:
        return dict(status="ALREADY_COMPLETE_NO_NEURAL_REPLAY", new_answers=0,
                    generation_config_binding=binding)
    def generation_audit():
        c.require(generation_configuration(model) == generation,
                  "Native model generation_config changed during bank inference")
        value = copy.deepcopy(audit())
        c.require("generation_config_binding" not in value,
                  "Generation-config audit field is owned by the native kernel")
        value["generation_config_binding"] = copy.deepcopy(binding)
        return value
    sink = Sink(missing, journal, generation_audit)
    inputs = [row["native_input"] for row in missing]
    question_path, answer_path = str(Path(args.question_file)), str(Path(args.answers_file))
    def scoped_open(path, mode="r", *a, **kw):
        if str(Path(path)) == question_path:
            c.require("r" in mode, "Question bank is immutable")
            body = json.dumps(inputs) if kind == "scienceqa" else "".join(
                json.dumps(row) + "\n" for row in inputs)
            return io.StringIO(body)
        if str(Path(path)) == answer_path:
            c.require(mode == "w", "Unexpected official answer output mode")
            return sink
        return builtins.open(path, mode, *a, **kw)
    def loader(*a, **kw):
        c.require(a[0] == args.model_path and a[1] == args.model_base, "Native model request changed")
        return loaded
    def progress(iterable, *a, **kw):
        for item in iterable:
            if time.monotonic() >= deadline or (stop_requested is not None and stop_requested()):
                raise AcceptedQABoundary("All previous raw answers are committed")
            yield item
    view = dict(vars(original), open=scoped_open, args=args, tqdm=progress,
                load_pretrained_model=loader)
    if kind == "loader":
        parent = original.CustomDataset
        getitem = types.FunctionType(parent.__getitem__.__code__, view, parent.__getitem__.__name__,
                                    parent.__getitem__.__defaults__, parent.__getitem__.__closure__)
        view["CustomDataset"] = type("BoundNativeDataset", (parent,), {"__getitem__": getitem})
        view["create_data_loader"] = types.FunctionType(
            original.create_data_loader.__code__, view, original.create_data_loader.__name__,
            original.create_data_loader.__defaults__, original.create_data_loader.__closure__)
    if kind == "mmbench":
        class PandasView:
            def __getattr__(self, name):
                return getattr(original.pd, name)
            def read_table(self, path, *a, **kw):
                c.require(str(Path(path)) == question_path, "Unexpected MMBench input")
                frame = original.pd.read_table(path, *a, **kw)
                indices = [row["native_id"] for row in missing]
                c.require(len(indices) == len(set(indices)), "Ambiguous MMBench native IDs")
                return frame[frame["index"].isin(indices)]
        view["pd"] = PandasView()
    runner = types.FunctionType(original.eval_model.__code__, view, original.eval_model.__name__,
                                original.eval_model.__defaults__, original.eval_model.__closure__)
    before = len(journal.present)
    try:
        runner(args)
        c.require(sink.position == len(missing), "Native loop skipped required semantic questions")
        return dict(status="COMPLETE_NATIVE_GENERATION", new_answers=len(journal.present) - before,
                    generation_config_binding=binding)
    except AcceptedQABoundary:
        return dict(status="DURABLE_MISSING_ONLY_NATIVE_QA_CONTINUATION",
                    new_answers=len(journal.present) - before,
                    generation_config_binding=binding)
    finally:
        sink.close()
