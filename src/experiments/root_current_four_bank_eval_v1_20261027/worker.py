"""Root comparator final checkpoints, unchanged official inference and QA journals."""
import argparse
import fcntl
import inspect
import types
import os
import stat
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from experiments.root_native_trained_evaluation_v3_20261026 import worker as original
from experiments.root_native_trained_evaluation_v3_20261026 import banks, media
from . import generation
from experiments.root_official_textvqa_join_v1_20261027 import banks as joined_banks
from experiments.root_native_trained_eval_qual_v4_20261026 import worker as qualified
from experiments.root_native_trained_loader_v4_20261026 import model
from . import contract as c, historical, checks

u = c.u
ROOT, HERE = c.ROOT, c.HERE
PYTHON = qualified.PYTHON
budget = original.budget
publish_same = original.publish_same
accepted_prefix_count = original.accepted_prefix_count
accepted_progress = original.accepted_progress


def sources(base, phase3):
    result = original.sources(base, phase3)
    for mapping in (qualified.sources(base, phase3), historical.route.sources()):
        for path, digest in mapping.items():
            c.require(path not in result or result[path] == digest, "Conflicting bound dependency")
            result[path] = digest
    for directory in (HERE, ROOT / "experiments/root_comparator_four_bank_eval_v1_20261026"):
        for path in directory.iterdir():
            if path.suffix in (".py", ".sbatch") and not path.name.startswith("test_"):
                result[str(path.relative_to(ROOT))] = u.sha(path.read_bytes())
    for relative in ('experiments/root_official_textvqa_join_v1_20261027/__init__.py',
                     'experiments/root_official_textvqa_join_v1_20261027/join.py',
                     'experiments/root_official_textvqa_join_v1_20261027/banks.py'):
        result[relative]=u.sha((ROOT/relative).read_bytes())
    return dict(sorted(result.items()))


def documents(binding, load=c.checked):
    cfg,base=checks.documents(binding, load, sources)
    c.source7_prefix_gate(cfg,load)
    return cfg,base


def current_storage(cfg, grant, used, statvfs=os.statvfs):
    other, inodes = grant["other_outstanding_reserved_bytes"], grant["reserved_inodes"]
    other_inodes = grant["other_outstanding_reserved_inodes"]
    c.require(type(other) is int and other >= 0 and type(inodes) is int and inodes > 0
              and type(other_inodes) is int and other_inodes >= 0
              and type(used) is int and 0 <= used <= cfg["output_limit_bytes"],
              "Explicit current result/collective storage accounting")
    fs = statvfs(cfg["output"])
    required = cfg["output_limit_bytes"] - used + other
    c.require(fs.f_bavail * fs.f_frsize >= required and fs.f_favail >= inodes + other_inodes,
              "Current evaluation bytes/inodes insufficient; preserve all paid QA records")
    return dict(available_bytes=fs.f_bavail * fs.f_frsize, available_inodes=fs.f_favail,
                required_remaining_bytes=required, reservation_other_bytes=other,
                this_is_not_a_global_uncommitted_capacity_grant=True)


def prior_attempt(binding, actual, out, load=u.checked, describe=u.desc):
    if not actual["restart"]:
        return None
    prior = out / ("r" + str(actual["restart"] - 1))
    previous = load(describe(prior / "RESULT.json"))
    intent = load(describe(prior / "REQUEUE_INTENT.json"))
    c.require(previous["status"] == "ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION"
              and previous["new_answers"] > 0 and previous["configuration"] == binding["configuration"]
              and previous["actual"]["job_id"] == actual["job_id"]
              and previous["actual"]["nonce"] == actual["nonce"]
              and previous["actual"]["restart"] == actual["restart"] - 1
              and intent["next_restart"] == actual["restart"]
              and intent["job_id"] == actual["job_id"] and intent["nonce"] == actual["nonce"]
              and intent["configuration"] == binding["configuration"]
              and intent["previous_result"] == describe(prior / "RESULT.json")
              and not (prior / "FAILURE.json").exists(), "Failed/foreign predecessor cannot auto-resume")
    return previous


def actor(binding, cfg):
    job, restart = binding["job_id"], os.environ.get("SLURM_RESTART_COUNT", "0")
    c.require(os.getuid() == 2708899 and job.isdecimal() and os.environ.get("SLURM_JOB_ID") == job
              and restart.isdecimal() and int(restart) < cfg["max_physical_attempts"]
              and not os.environ.get("SLURM_ARRAY_JOB_ID") and ROOT == c.SOURCE_ROOT
              and Path(sys.executable).absolute() == Path(PYTHON),
              "Own current finite scalar evaluator actor")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    fields = u.fields(raw)
    expected = dict(JobId=job, JobState="RUNNING", JobName=c.NAME, UserId="luyangz(2708899)",
        Partition="general", Account="cx", QOS="normal", NumCPUs="8", NumTasks="1", NumNodes="1",
        Restarts=restart, TimeLimit="06:00:00", TresPerNode="gres/gpu:L40S:1",
        Command=str(HERE / "run.sbatch"), WorkDir=str(ROOT), Comment=c.COMMENT + cfg["nonce"],
        NodeList=socket.gethostname().split(".")[0])
    c.require(all(fields.get(key) == value for key, value in expected.items()), "Current evaluation actor differs")
    tres = dict(value.split("=", 1) for value in fields["AllocTRES"].split(",") if "=" in value)
    c.require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M")
              and tres.get("gres/gpu") == "1", "Current one L40S/CPU8/64GiB required")
    c.require(subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
              == (HERE / "run.sbatch").read_bytes(), "Own exact frozen spool")
    c.require(os.environ.get("TZ") == "America/New_York", "Exact scheduler timezone")
    time.tzset()
    return dict(job_id=job, restart=int(restart), node=fields["NodeList"], nonce=cfg["nonce"], scheduler=raw,
        end_epoch=int(time.mktime(time.strptime(fields["EndTime"], "%Y-%m-%dT%H:%M:%S"))),
        process_nonce=uuid.uuid4().hex, pid=os.getpid())


def function(cfg, identity, roster, complete, bank):
    return checks.function(cfg, identity, roster, complete, bank)


def strict_loaded_support(proof):
    values, injection = proof["portable_value_verification"], proof["native_adapter_construction"]
    c.require(values["exact_parameter_inventory"] is True and values["frozen_parameters"] == 682
              and values["trained_parameters"] == 452 and values["LoRA_tensors"] == 448
              and values["projector_tensors"] == 4 and values["portable_trainable_values_exact"] is True
              and len(injection["language_adapter_targets"]) == 224
              and len(injection["loaded_vision_generic_targets_excluded"]) == 72
              and proof["persistent_buffer_verification"]["exact_persistent_buffer_support"] is True
              and proof["persistent_buffer_verification"]["persistent_buffer_values_exact"] is True,
              "Every final checkpoint must reconstruct full1134/452/224/72 support and buffers")



def joined_bank_data(cfg,attempt):
    # This happens before model construction, with original answer-free rosters.
    _,qualified=c.official_join_preflight(cfg)
    bankdata={}
    for bank in cfg['banks']:
        value,receipt,mapping=joined_banks.load(bank,cfg,source_root=ROOT)
        bankdata[bank]=value
        if bank=='TextVQA':
            c.require(receipt=={k:v for k,v in qualified.items()if k!='mapping'},
                'Actual current native/GT join differs from independently qualified input mapping')
            mappingref=u.publish(attempt/'TEXTVQA_JOIN_MAPPING.json',mapping)
            u.publish(attempt/'TEXTVQA_JOIN_RECEIPT.json',dict(receipt,mapping=mappingref,
                actual_input_qualification=cfg['official_join_input_qualification'],
                actual_input_qualification_review=cfg['official_join_input_qualification_review']))
    return bankdata



def early_asset_readiness(cfg):
    # Metadata + stat/open1byte only. All original fullSHA checks in compose remain.
    training=u.checked(cfg['training_configuration'])
    from experiments.session6_start_evaluation_v1_20261024 import model as start
    receipt=u.checked(training['providers']['assets']);c.require(receipt['status']=='PASS_ASSETS','Exact original asset receipt')
    cache=Path('/data/hf_cache').resolve();rows=[]
    for role,pin in start.PINS.items():
        asset=receipt['assets'][role]
        c.require((asset['repository'],asset['revision'])==pin and asset['files']and Path(asset['snapshot']).resolve().is_relative_to(cache),'Exact existing model/processor assets')
        for ref in asset['files']:
            path=Path(ref['path']);resolved=path.resolve(strict=True);c.require(resolved.is_relative_to(cache),'Bound alias resolves outside approved cache')
            before=path.stat()
            c.require(stat.S_ISREG(before.st_mode)and before.st_size==ref['bytes']and before.st_size>0,'Bound asset missing/type/size mismatch')
            with path.open('rb')as inp:
                opened=os.fstat(inp.fileno());byte=inp.read(1);after=os.fstat(inp.fileno())
            sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            c.require(len(byte)==1 and sig(before)==sig(opened)==sig(after)==sig(path.stat()),'Bound asset readability/identity changed')
            rows.append(dict(role=role,expected={k:ref[k]for k in('path','bytes','sha256')},resolved=str(resolved),one_byte_readable=True,full_SHA_checked_by_this_fast_probe=False))
    return dict(status='EARLY_BOUND_MODEL_ASSET_STAT_READABILITY_NOT_FULL_SHA_OR_MODEL_QUALIFICATION',rows=rows,
        full_SHA_and_tensor_and_model_checks_still_required_by_unchanged_compose=True,neural_forwards=0)


def prefix_bridge_snapshot(loaded_model,function):
    # Journal append occurs INSIDE the unchanged pinned Recorder context. Do not
    # mutate model.generate to invoke the final (outside-context) readout API.
    bridge=loaded_model._dsmm_native_inputs_bridge;api=model.generation
    c.require(type(bridge)is api.PositionalInputsBridge and bridge.model is loaded_model,
        'Exact installed native positional bridge required')
    wrapped=loaded_model.generate
    expected=[x for x in qualified.audit.Recorder.installed.__wrapped__.__code__.co_consts
        if isinstance(x,types.CodeType)and x.co_name=='generate']
    c.require(len(expected)==1 and isinstance(wrapped,types.FunctionType)and wrapped.__code__ is expected[0],
        'Prefix occurs only inside exact unchanged official Recorder generate wrapper')
    closure=inspect.getclosurevars(wrapped).nonlocals;recorder=closure['self']
    c.require(type(recorder)is qualified.audit.Recorder and closure['original']is bridge
        and recorder.loaded[1]is loaded_model and recorder.function==function
        and recorder.pending is None and recorder.position==2,
        'Actual official audit consumed exactly two native generations before durable append')
    api.active(loaded_model)
    c.require(api.same_method(loaded_model.base_model.prepare_inputs_for_generation,
        loaded_model.base_model_prepare_inputs_for_generation),'Original PEFT preparation hook restored after actual call')
    return dict(status='ACTUAL_NATIVE_BRIDGE_PREFIX_INSIDE_UNCHANGED_OFFICIAL_AUDIT_CONTEXT',
        policy=api.POLICY,runtime_API_proof=bridge.proof,calls=len(bridge.records),records=list(bridge.records),
        adapter_active=True,adapter_merged=False,unchanged_official_Recorder_wrapper_authenticated=True,
        generate_or_preparation_hook_mutated_by_observer=False)


def install_prefix_observer(cfg,binding,actual,attempt,journal,rows,loaded3,proof,runtime,identity,identity_ref,complete_ref):
    if cfg['source7_prefix_mode']!='PRODUCER'or actual['restart']!=0:return lambda:None
    c.require(not journal.present,'Initial source7producer starts only from an actually empty formal journal')
    original=journal.append;written=[False]
    def append(*args,**kwargs):
        returned=original(*args,**kwargs)
        if len(journal.present)==2 and not written[0]:
            strict_loaded_support(proof)
            after=model.identity(loaded3,proof,runtime);c.require(after==identity,'Prefix generation changed full model/processor/tokenizer/buffers')
            bridge=prefix_bridge_snapshot(loaded3[1],journal.function);entries=qualified.official.entries(journal,rows)
            c.require(len(entries)==bridge['calls']==len(bridge['records'])==2
                and [e['semantic_key']for e in entries]==[r['semantic_key']for r in rows[:2]]
                and all(all(record[k]is True for k in('success','input_object_unchanged','all_keyword_objects_unchanged','prepare_hook_restored','original_PEFT_generate_invoked','output_object_unchanged'))for record in bridge['records']),
                'Exactly two first durably accepted actual native/PEFT bridge QAs')
            for row,entry in zip(rows[:2],entries):generation.validate_entry('POPE',entry,row,journal.function)
            post=u.publish(attempt/'PREFIX_MODEL_IDENTITY.json',after)
            u.publish(attempt/'COLD_MODEL_QA_PREFIX.json',dict(status='ACTUAL_SOURCE7_COLD_MODEL_TWO_NATIVE_QA_PREFIX_NOT_FULL_BANK',configuration=binding['configuration'],actual=actual,
                bank='POPE',accepted_native_QAs=2,semantic_QA_keys=[e['semantic_key']for e in entries],entries=entries,
                function=journal.function,model_reconstruction=u.desc(attempt/'MODEL_RECONSTRUCTION.json'),model_identity=identity_ref,post_prefix_model_identity=post,
                model_identity_unchanged_after_prefix=True,generation_bridge=bridge,student_completion=complete_ref,
                actual_bound19_asset_readiness=cfg['actual_bound19_asset_readiness'],actual_model_compose_in_this_fresh_process=True,
                no_extra_or_replayed_QAs=True,full_native_loop_continues_unchanged=True,cache_coldness_claimed=False,scientific_FULL_bank_or_SOTA_PASS=False))
            written[0]=True
        return returned
    journal.append=append
    def restore():journal.append=original
    return restore


def run(binding_path):
    binding = u.checked(u.desc(binding_path))
    cfg, base = documents(binding)
    actual, began = actor(binding, cfg), time.monotonic()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / "EXECUTION.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    attempt = out / ("r" + str(actual["restart"]))
    try:
        attempt.mkdir(exist_ok=False)
    except OSError:
        os.close(fd)
        raise
    try:
        previous = prior_attempt(binding, actual, out)
        grant = u.checked(binding["storage_reservation"])
        u.publish(attempt / "CURRENT_STORAGE.json", current_storage(cfg, grant, budget(cfg)))
        u.publish(attempt / 'EARLY_BOUND_ASSET_READINESS.json', early_asset_readiness(cfg))
        _, complete_ref, complete, physical_proofs = checks.complete_student(cfg)
        u.publish(attempt / "START.json", dict(actual=actual, configuration=binding["configuration"],
                  student_completion=complete_ref, historical_physical_proofs=physical_proofs))
        bankdata = joined_bank_data(cfg, attempt)
        ready = {}
        for bank, (roster, inventory, _, physical, snapshot) in bankdata.items():
            store = media.Store(physical, roster["records"], inventory, snapshot)
            try:
                names = sorted({row["native_input"]["image"] for row in roster["records"]})
                for name in names:
                    image = store.rgb(name)
                    image.close()
                ready[bank] = dict(questions=len(roster["records"]), all_encoded_and_RGB_verified=True,
                                   images=len(names), forwards=0)
            finally:
                store.close()
        u.publish(attempt / "FULL_MEDIA_READINESS.json", ready)
        publish_same(out / "STUDENT_INPUT.json", dict(training_configuration=cfg["training_configuration"],
                     student_completion=complete_ref, accepted=complete["accepted"], physical_proofs=physical_proofs))
        runtime = qualified.official.runtime(base, attempt)
        u.publish(attempt / "RUNTIME.json", runtime)
        with qualified.official.offline():
            loaded3, proof = model.compose(complete["accepted"], cfg["training_configuration"])
            strict_loaded_support(proof)
            identity = model.identity(loaded3, proof, runtime)
            identity_ref = publish_same(out / "MODEL_IDENTITY.json", identity)
            u.publish(attempt / "MODEL_RECONSTRUCTION.json", proof)
            loaded = (*loaded3, getattr(loaded3[1].config, "max_sequence_length", 2048))
            reports, new = {}, 0
            for bank, (roster, inventory, targets, physical, snapshot) in bankdata.items():
                rows, path = roster["records"], out / bank
                path.mkdir(exist_ok=True)
                fn = function(cfg, identity, roster, complete_ref, bank)
                publish_same(path / "FUNCTION.json", fn)
                journal = generation.AuditedJournal(path / "journal", fn, rows)
                store = media.Store(physical, rows, inventory, snapshot)
                restore_prefix=lambda:None
                try:
                    if bank=='POPE':restore_prefix=install_prefix_observer(cfg,binding,actual,attempt,journal,rows,loaded3,proof,runtime,identity,identity_ref,complete_ref)
                    for entry in qualified.official.entries(journal, rows):
                        generation.validate_entry(bank, entry, journal.roster[entry["semantic_key"]], fn)
                    accepted_prefix_count(previous, bank, len(journal.present))
                    before = len(journal.present)
                    if time.time() < actual["end_epoch"] - 300:
                        generation.native(bank, cfg, rows, journal, loaded, store, actual,
                                          time.monotonic() + max(0, actual["end_epoch"] - time.time() - 240))
                    new += len(journal.present) - before
                    reports[bank] = dict(questions=len(rows), accepted=len(journal.present),
                                        missing=len(journal.missing()), new=len(journal.present) - before)
                    if not journal.missing():
                        raw = journal.export(path / "PREDICTIONS.jsonl")
                        score = banks.score(bank, roster, journal, targets)
                        publish_same(path / "SCORE.json", dict(
                            status="COMPLETE_ROOT_COMPARATOR_NATIVE_BANK_AND_OFFICIAL_SCORING",
                            bank=bank, function=fn, predictions=raw, score=score,
                            student_completion=complete_ref, complete_questions=len(rows),
                            scientific_improvement_demonstrated=False))
                finally:
                    restore_prefix()
                    journal.close()
                    store.close()
                budget(cfg)
        post_identity = model.identity(loaded3, proof, runtime)
        c.require(post_identity == identity, "Generation changed final parameter/buffer/tokenizer/processor identity")
        post_ref = u.publish(attempt / "POST_GENERATION_MODEL_IDENTITY.json", post_identity)
        bridge = model.generation.readout(loaded3[1])
        c.require(bridge["calls"] == new, "Every new QA must have one actual V4 bridge call")
        bridge_ref = u.publish(attempt / "GENERATION_BRIDGE_READOUT.json", bridge)
        done = accepted_progress(previous, reports, new)
        value = dict(status="COMPLETE_ROOT_COMPARATOR_FOUR_OFFICIAL_BANKS" if done else
                     "ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION", actual=actual,
                     configuration=binding["configuration"], student_completion=complete_ref,
                     accepted_checkpoint=complete["accepted"], model_identity=identity_ref, banks=reports,
                     post_generation_model_identity=post_ref, generation_bridge_readout=bridge_ref,
                     new_answers=new, elapsed_seconds=time.monotonic() - began,
                     all_prior_and_current_costs_count=True, scientific_improvement_demonstrated=False)
        result_ref = u.publish(attempt / "RESULT.json", value)
        if done:
            publish_same(out / "COMPLETE.json", value)
        else:
            c.require(new > 0 and actual["restart"] + 1 < cfg["max_physical_attempts"],
                      "No progress or QA-derived attempt allowance exhausted")
            intent = u.publish(attempt / "REQUEUE_INTENT.json", dict(job_id=actual["job_id"],
                nonce=actual["nonce"], next_restart=actual["restart"] + 1, previous_result=result_ref,
                configuration=binding["configuration"], same_job_only=True, accepted_per_QA_journal=True))
            result = subprocess.run(["scontrol", "requeue", actual["job_id"]], text=True,
                                    capture_output=True, timeout=30)
            u.publish(attempt / "REQUEUE_ACK.json", dict(intent=intent, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr, ambiguous_ack_requires_Root_reconciliation=True))
            c.require(result.returncode == 0, "Ambiguous/failed continuation; never duplicate")
    except BaseException as error:
        u.publish(attempt / "FAILURE.json", dict(status="ROOT_COMPARATOR_FAILED_PRESERVE_QAS_NO_AUTO_RETRY",
                  actual=actual, error=repr(error), elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    run(parser.parse_args().binding)
