import argparse
from pathlib import Path

from . import common as c


def prepare(directory):
    directory = Path(directory)
    protocol = c.ROOT / "collaboration/session6-official-eval-v2-20261024/EVALUATION_PROTOCOL_V2.json"
    c.require(c.sha(protocol) == "f1ff1af008482dfba1ee9945065ee57d0e38592fef30ee75d0353545e9746416",
              "Exact pre-outcome evaluation protocol required")
    config = dict(schema=c.SCHEMA, owner=c.OWNER, job_name="dsmm-s6-start-model-v4eager",
        python=c.PYTHON, cache_root=c.CACHE, output=str(c.BASE / "output"),
        admission_lock="/scratch/user/u.lz341613/dsmm_root_initial24_aces_20261023/ADMISSION.lock",
        reserved_bytes=1024 ** 3, training_updates=0, benchmark_predictions=0,
        projector_materialization="EXACT_PINNED_FP16_ASSIGN_TRUE_NO_META",
        diagnostic_before_guard=True,
        causal_attention_binding="EXPLICIT_EAGER_INTERNAL_BEFORE_CONSTRUCTOR",
        predecessor=dict(job_id="2251389", state="FAILED1:0",
                         cause="ActualLlamaSdpaAttention+configsdpa;allFP16CUDA/frozen/noPEFTtrueand0parameterexceptions."),
        native_assets=dict(
            path="/scratch/user/u.lz341613/dsmm_native_campaign_20261023/bootstrap-v3/attempts/2249160-r0-1792802576/ASSETS.json",
            bytes=8717, sha256="154f1902fa283bf54a74f3080b67cd9eb3ec5a548acad3ee295252504da38750"),
        evaluation_protocol=dict(c.desc(protocol), path=str(c.BASE / "src" / protocol.relative_to(c.ROOT))),
        scope="Versioned2251389causalSDPAdispatchrepair: explicitlysetconfigeagerinternalBEFOREnativefrom_pretrained; kwargs-onlyeagerisignoredbyTF4.37whenpropertydefaultseager/internalNone. PreservedurableV3observations/strictactualeager/frozen/FP16CUDAandV2materializer.15minH100no-trainingcheck,0forward/image/download.",
        remaining="Real-pixel/processor numerical/generation qualification and full-bank inference/scoring remain separate.",
        sources=c.sources())
    local = c.seal(directory / "CONFIG.json", config)
    remote = dict(local, path=str(c.BASE / "control/CONFIG.json"))
    packet = c.seal(directory / "SOURCE_PACKET_V1.json", dict(
        schema="session6-start-model-qualification-source-config-v4-eager",
        configuration=remote, local_configuration=local, sources=config["sources"],
        authored_tests=c.desc(c.HERE / "test_qualification.py"),
        real_native_diagnostic_tests=c.desc(c.HERE / "test_diagnostic.py"),
        real_native_dispatch_tests=c.desc(c.HERE / "test_eager_dispatch.py"),
        actual_compute_completed=False, actual_benchmark_predictions=0,
        independent_review_required=True, global_claim_or_scheduler_operation=False))
    return dict(configuration=remote, packet=packet, source_files=len(config["sources"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    print(c.canonical(prepare(parser.parse_args().output)).decode())
