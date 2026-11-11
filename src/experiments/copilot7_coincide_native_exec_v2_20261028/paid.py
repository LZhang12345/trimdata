"""Exact already-paid provider identities, discovered from genuine custody exports."""
from . import protocol as p
from experiments.root_fulltrajectory_student_v4_20261025 import shared_runtime

H=str(p.HOME)+"/"
ENVIRONMENT=dict(path=H+"collaboration/root-native-launch-20261023/babel-overlay-v2/ENVIRONMENT.json",
    bytes=1436,sha256="69ea6a8af1080f6229fd1c769933f86a959face3a381c0caa260ec940dbcc187")
ASSETS=dict(path=H+"experiments/published_foundation_20261022/runtime-v1/outputs/assets/COMPLETE.json",
    bytes=10819,sha256="b6fe9f87e7fc20809f6176ad75014469e16fdfbc2120fa83cd1c512850bed027")
NATURAL=dict(path=H+"native_campaign_v3_20261023/src/collaboration/root-canonical-student-pilot-v1-20261025/BABEL_SEED17_NATURAL_EVIDENCE.json",
    bytes=863,sha256="b66fe2459fa68482f6e6bef57f516fb07a2adc9e691ca6bcef0ce31ad28f1f3f")
PROFILE=dict(path=H+"collaboration/root-babel-shared-runtime-v4-20261025/ACTUAL_SHARED_FS_PROFILE_V2.json",
    bytes=1139,sha256="49f7fdb35244ad826d2582d883ea32ab720af823445ae9ba49164ed7f0864309")
COMPATIBILITY=dict(path=H+"collaboration/root-babel-shared-runtime-v4-20261025/review-boundary/INDEPENDENT_ACTUAL_PROFILE_BRIDGE_V4.json",
    bytes=80280,sha256="201b1938cda26712bc83f086ce7b84f1ee2764505b94974e592134f036026ec3")
OPERATOR=dict(path=H+"collaboration/root-native-layout-gpu-qualification-v1-20261025/attempts/10931459-6fbb4150832f4c738694c831aadcffd3/RESULT.json",
    bytes=898692,sha256="4f9d479237d8fdd4db1140f73f43b994f80a448b75bbd282ab5918223fc68491")
OPERATOR_REVIEW=dict(path=H+"collaboration/root-native-layout-gpu-qualification-v1-20261025/INDEPENDENT_ACTUAL_GPU_READOUT_REVIEW_V1.json",
    bytes=4000,sha256="9ce5645d995433395749ba1d814a191c1ef3111bef9ac4afd45e92d650ab4509")
CANARY=dict(path=H+"collaboration/root-babel-unblock-20261026/canary-collector-v1/ACTUAL_SHARED_RUNTIME_CANARY_V1.json",
    bytes=108050,sha256="17f488f922340197169418df322f17df4bb87db2ec62d0cfe1d4dc70885c85b7")
CANARY_REVIEW=dict(path=H+"collaboration/root-babel-unblock-20261026/canary-collector-v1/INDEPENDENT_ACTUAL_SHARED_RUNTIME_CANARY_REVIEW_V1.json",
    bytes=110933,sha256="a0ac783769590ce50124642708508072f371ada2b5193e5b34033d6f61b070f1")
SOURCE_BRIDGE=dict(path=H+"collaboration/root-babel-shared-runtime-v4-20261025/review-boundary/INDEPENDENT_NUMERICAL_SOURCE_BRIDGE_V4.json",
    bytes=100137,sha256="531419a24b0f31d143d00c723fb69b87059bbc0749d7e9ed73f8fd3f7fac2f49")


def inventory():
    return dict(environment=ENVIRONMENT,assets=ASSETS,natural_evidence=NATURAL,
        historical_site_profile=PROFILE,historical_profile_compatibility=COMPATIBILITY,
        actual_GPU_operator_qualification=OPERATOR,actual_GPU_operator_review=OPERATOR_REVIEW,
        actual_v4_canary=CANARY,actual_v4_canary_review=CANARY_REVIEW,
        paid_logical_numerical_source_bridge=SOURCE_BRIDGE,
        longwall_configuration=shared_runtime.ANCHOR_CONFIG,
        longwall_readout=shared_runtime.ANCHOR_READOUT,longwall_review=shared_runtime.ANCHOR_REVIEW)


def reuse_review_request():
    return dict(status="REQUIRES_GENUINE_INDEPENDENT_A_NATIVE_PAID_RUNTIME_REUSE_REVIEW",
        required_status="PASS_INDEPENDENT_EXACT_COINCIDE_NATIVE_PAID_RUNTIME_REUSE_V2",
        reviewer=None,anchor_sources=p.anchor.sources(),target_sources=p.sources(),
        native_sources=p.n.sources(),numerical_bodies=p.numerical_bodies(),
        longwall_readout=shared_runtime.ANCHOR_READOUT,longwall_review=shared_runtime.ANCHOR_REVIEW,
        actual_GPU_operator_qualification=OPERATOR,actual_GPU_operator_review=OPERATOR_REVIEW,
        actual_v4_canary=CANARY,actual_v4_canary_review=CANARY_REVIEW,
        no_old_checkpoint_identity_migration=True,actual_new_GPU_result_claimed=False)


def gate(profile_ref,reuse_review_ref):
    return dict(schema="production-student-runtime-gate-contract-v1",status=p.GATE,
        native_sources=p.n.sources(),recipe=p.native_config.RECIPE,site_profile=profile_ref,
        environment=ENVIRONMENT,assets=ASSETS,site="babel",runtime=p.RUNTIME,
        qualification_evidence=NATURAL,runtime_reuse_review=reuse_review_ref,
        student_adapter=dict(package=p.PACKAGE,sources=p.sources()),
        ordinary_native_launch_forbidden=True,requires_separate_outer_authorization=True,
        natural_runtime_seed_policy=p.SEED_POLICY)
