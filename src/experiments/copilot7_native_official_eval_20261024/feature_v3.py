"""Versioned NumPy expected-component repair and actual pre-constructor inventory."""
import argparse
from copy import deepcopy
import importlib
import importlib.metadata
from pathlib import Path
import sys

from . import common as c
from . import feature_v1 as first
from . import feature_v2 as prior
from experiments.copilot7_same_site_baselines_20261024.runtime_receipt_v1 import require_datatailor_versions


PARENT_ENV_SHA = "69ea6a8af1080f6229fd1c769933f86a959face3a381c0caa260ec940dbcc187"
PROVIDER_SHA = "1e65b158cff5b700771f39ad5ccf441a87ff30299bbc36c062091676aa0ee48f"
PROVIDER_CONFIG_SHA = "11778cdb9f04690520d5d8b4552fa6a7c1b4cbea4469f3b86251522a55dd5a55"


def expected_environment(parent_description, provider_description, provider_configuration):
    c.require(parent_description["sha256"] == PARENT_ENV_SHA
              and provider_description["sha256"] == PROVIDER_SHA
              and provider_configuration["sha256"] == PROVIDER_CONFIG_SHA,
              "Exact existing same-interpreter runtime providers required")
    parent = c.checked(parent_description)
    provider = c.checked(provider_description)
    configuration = c.checked(provider_configuration)
    c.require(provider["actual"]["job_id"] == "10911194" and provider["configuration"] == provider_configuration
              and configuration["native_environment"] == parent_description
              and configuration["native_python"] == parent["python_executable"],
              "Runtime inventory is not from the exact native overlay interpreter")
    for name in ("torch", "transformers"):
        c.require(provider["runtime"][name] == parent["versions"][name], "Existing runtime component differs: " + name)
    numpy_version = provider["runtime"]["numpy"]
    c.require(isinstance(numpy_version, str) and numpy_version, "Actual same-interpreter NumPy distribution is required")
    result = deepcopy(parent)
    c.require("numpy" not in result["versions"], "Do not replace an existing expected NumPy component")
    result["versions"]["numpy"] = numpy_version
    result["numpy_component_addendum"] = dict(
        schema="s7-expected-numpy-component-from-actual-distribution-v1",
        parent_environment=parent_description, actual_distribution_provider=provider_description,
        provider_configuration=provider_configuration, observed_distribution_version=numpy_version,
        imported_numpy_module_version_or_origin_claimed=False,
        mandatory_current_import_version_origin_and_binary_inventory_before_constructor=True,
        old_receipts_modified=False, model_forward_or_restart_qualified=False,
    )
    require_datatailor_versions(result)
    return result


def observe(environment):
    expected = require_datatailor_versions(environment)
    c.require(str(Path(sys.executable).absolute()) == environment["python_executable"],
              "Use the exact expected native interpreter")
    modules = {}
    for name, version in expected.items():
        module = importlib.import_module(name)
        distribution = importlib.metadata.distribution(name)
        c.require(module.__version__ == distribution.version == version, "Actual imported runtime differs: " + name)
        source = Path(module.__file__).absolute()
        installed = Path(distribution.locate_file(name + "/__init__.py")).absolute()
        c.require(source.resolve() == installed.resolve(), "Imported module is shadowed outside its observed distribution: " + name)
        modules[name] = dict(version=module.__version__, distribution_version=distribution.version, source=c.desc(source))
    numpy = importlib.metadata.distribution("numpy")
    binaries = {}
    c.require(numpy.files is not None, "NumPy installed-file inventory is absent")
    for relative in numpy.files:
        name = str(relative)
        if name.startswith(("numpy/", "numpy.libs/")) and name.endswith((".so", ".dylib", ".pyd")):
            path = Path(numpy.locate_file(relative)).absolute()
            c.require(path.is_file(), "Installed NumPy binary disappeared")
            binaries[name] = c.desc(path)
    c.require(binaries, "Actual NumPy compiled-component inventory required")
    return dict(modules=modules, numpy_binaries=binaries, no_model_constructor_or_neural_calls=True)


def run(binding_path):
    binding = c.read(binding_path)
    spec = c.checked(binding["configuration"])
    c.require(spec["phase"] == "datatailor32" and spec["site"] == "babel", "Only repaired Babel DT32 scope")
    c.verify_sources(spec["sources"])
    c.verify_sources(spec["dependencies"])
    review = c.checked(binding["review"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == binding["configuration"] and review["sources"] == spec["sources"]
              and review["dependencies"] == spec["dependencies"], "Independent exact V3 review required")
    actual = first.actual(binding, spec)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    capacity = c.os.statvfs(output)
    c.require(capacity.f_bavail * capacity.f_frsize >= 1 << 30 and capacity.f_favail >= 1000,
              "Insufficient measured bounded engineering storage")
    expected = c.checked(spec["inputs"]["environment"])
    provenance = spec["numpy_component_providers"]
    c.require(expected == expected_environment(provenance["parent"], provenance["actual_provider"],
                                               provenance["provider_configuration"]),
              "Expected runtime addendum changed from the exact actual component providers")
    observed = observe(expected)
    fingerprint = c.seal(output / "ACCEPTED_NUMPY_RUNTIME_COMPONENT.json", dict(
        environment=spec["inputs"]["environment"], runtime=observed,
        scope="ACTUAL_COMPONENT_ONLY_NOT_FEATURE_FORWARD_OR_RESTART_QUALIFICATION"))
    attempt = output / ("s7-attempt-" + actual["job_id"] + "-r" + str(actual["restart"]))
    c.seal(attempt / "ACTUAL_RUNTIME_BEFORE_CONSTRUCTOR.json", dict(
        actual=actual, expected_environment=spec["inputs"]["environment"],
        accepted_component=fingerprint, runtime=observed, no_constructor_or_neural_calls_at_this_boundary=True))
    return prior.run(binding_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
