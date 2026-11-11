"""Fail before acquisition when an expected native runtime receipt is incomplete."""
import argparse

from experiments.root_native_campaign_20261023 import common as c


CONSUMED_VERSION_KEYS = ("torch", "transformers", "numpy")


def require_datatailor_versions(environment):
    versions = environment["versions"]
    c.require(isinstance(versions, dict), "Expected native runtime versions must be an object")
    missing = [name for name in CONSUMED_VERSION_KEYS
               if name not in versions or not isinstance(versions[name], str) or not versions[name]]
    c.require(not missing, "Incomplete DataTailor runtime receipt: " + ", ".join(missing)
              + "; acquire a versioned exact-compatible actual environment addendum before release")
    c.require(versions["torch"].startswith("2.1.2") and versions["transformers"] == "4.37.2",
              "Wrong native DataTailor Torch/Transformers API")
    c.require(isinstance(environment["python_executable"], str) and environment["python_executable"],
              "Bind the actual isolated interpreter")
    return {name: versions[name] for name in CONSUMED_VERSION_KEYS}


def compare_datatailor_runtime(environment, observed_versions, executable):
    expected = require_datatailor_versions(environment)
    c.require(executable == environment["python_executable"], "Observed interpreter differs from expected runtime")
    for name, version in expected.items():
        c.require(name in observed_versions and observed_versions[name] == version,
                  "Actual native runtime version differs: " + name)
    return expected


def preflight(description):
    return require_datatailor_versions(c.checked(description))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment_descriptor", help="JSON file containing the exact expected environment descriptor")
    args = parser.parse_args()
    print(c.canonical(preflight(c.read(args.environment_descriptor))).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
