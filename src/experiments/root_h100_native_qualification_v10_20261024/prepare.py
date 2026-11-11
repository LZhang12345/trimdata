"""Draft a separately reviewed H100 study; no scheduler operations or readiness fabrication."""
import argparse
from . import protocol as p


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("draft-design", "draft"):
        draft = sub.add_parser(command)
        for name in ("id", "target", "reference-config", "resume-config", "provenance",
                     "source-pool-metadata", "actual-prefix-proof", "raw-prefix-proof-copy", "report", "output"):
            draft.add_argument("--" + name, required=True)
        if command == "draft":
            draft.add_argument("--design", required=True)
    for command in ("bind-design", "bind-review"):
        bind = sub.add_parser(command)
        for name in ("draft", "review", "output"):
            bind.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.action in ("draft", "draft-design"):
        value = {
            "schema": "h100-transport-engineering-study-v10", "owner": "Root", "id": args.id,
            "purpose": "H100_TRANSPORT_ENGINEERING_ONLY", "total_steps": 8, "interrupt_step": 4,
            "minimum_launch_seconds": 600, "numeric_tolerance": p.TOLERANCE, "driver_sources": p.sources(),
            "native_target": {"package": "experiments.root_native_campaign_v7_20261024",
                              "source_packet": p.desc(args.target)},
            "native_configurations": {"reference": p.desc(args.reference_config), "resume": p.desc(args.resume_config)},
            "input_rebind_provenance": p.desc(args.provenance), "report": args.report,
            "source_pool_metadata": p.desc(args.source_pool_metadata),
            "actual_prefix_proof": p.desc(args.actual_prefix_proof),
            "raw_prefix_proof_copy": p.desc(args.raw_prefix_proof_copy),
        }
        if args.action == "draft-design":
            pair, _ = p.configs(value, reviewed=False, with_design=False)
            value = p.design_contract(value, pair)
        else:
            value["design"] = p.desc(args.design)
            p.configs(value, reviewed=False)
    else:
        value = p.read(args.draft)
        p.require("review" not in value, "Draft already bound to review")
        value["review"] = p.desc(args.review)
        if args.action == "bind-design":
            p.validate_design(value)
        else:
            p.configs(value)
    print(p.canonical(p.seal(args.output, value)).decode())


if __name__ == "__main__":
    main()
