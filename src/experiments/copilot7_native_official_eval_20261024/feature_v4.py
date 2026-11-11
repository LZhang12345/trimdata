"""Complete NumPy shared-library and parent-module-origin inventory."""
import argparse
from pathlib import Path
import re
import types

from . import common as c
from . import feature_v3 as prior


def observe(environment):
    result = prior.observe(environment)
    for name in ("torch", "transformers"):
        c.require(name in environment["module_sources"], "Exact parent module source missing: " + name)
        c.require(Path(result["modules"][name]["source"]["path"]).resolve()
                  == Path(environment["module_sources"][name]).resolve(),
                  "Imported source differs from the bound parent environment: " + name)
    distribution = prior.importlib.metadata.distribution("numpy")
    c.require(distribution.files is not None, "NumPy installed-file inventory is absent")
    binaries = {}
    for relative in distribution.files:
        name = str(relative)
        if name.startswith(("numpy/", "numpy.libs/")) and re.search(r"\.(?:so(?:\.\d+)*|dylib|pyd)$", name):
            path = Path(distribution.locate_file(relative)).absolute()
            c.require(path.is_file(), "Installed NumPy shared library disappeared")
            binaries[name] = c.desc(path)
    c.require(binaries, "Complete actual NumPy compiled-component inventory required")
    result["numpy_binaries"] = binaries
    result["numpy_binary_inventory_rule"] = "numpy-or-numpy.libs-so-versioned-sonames-dylib-pyd-v1"
    result["parent_torch_transformers_module_origins_checked"] = True
    return result


def run(binding_path):
    runner = types.FunctionType(prior.run.__code__, dict(vars(prior), observe=observe))
    return runner(binding_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
