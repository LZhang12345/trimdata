"""Declared native-image paths backed by exact existing file/ZIP bytes only."""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path, PurePosixPath
import warnings
import zipfile
from collections import Counter

from experiments.session6_official_eval_v2_20261024 import common as c


def fingerprint(path):
    p = Path(path)
    s = p.stat()
    c.require(p.is_file(), "Regular cached image/archive required")
    return dict(path=str(p), bytes=s.st_size, inode=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


class Store:
    def __init__(self, cfg, roster, inventory, archive_snapshot=None):
        self.cfg, self.images = cfg, inventory["images"]
        self.mapping = {r["native_input"]["image"]: r["image_reference"] for r in roster}
        self.archive_snapshot = archive_snapshot
        self.virtual = Path(cfg["output"]) / "native-image-view"
        self.opened = None
        self.members = None
        self.archives = {}

    def archive(self, path):
        if path not in self.archives:
            z = zipfile.ZipFile(path)
            self.archives[path] = (z, Counter(z.namelist()))
        return self.archives[path]

    def close(self):
        for z, _ in self.archives.values():
            z.close()
        self.archives.clear()

    def payload(self, name):
        c.require(name in self.mapping and not PurePosixPath(name).is_absolute()
                  and ".." not in PurePosixPath(name).parts, "Undeclared/unsafe native image")
        expected = self.images[self.mapping[name]]
        locator = expected.get("storage_locator")
        if locator and locator["kind"] == "shared_original_file":
            ref = locator["file"]
            c.require(Path(ref["path"]).resolve().is_relative_to(Path(self.cfg["cache_root"]).resolve()),
                      "Original file outside shared cache")
            c.require(c.desc(ref["path"]) == ref, "Original cached file changed")
            raw = Path(ref["path"]).read_bytes()
        else:
            if locator:
                c.require(locator["kind"] == "verified_original_bytes_in_train2017_member",
                          "Unsupported original storage locator")
                snapshot = locator["archive_fingerprint"]
                archive, member = snapshot["path"], locator["member"]
            else:
                snapshot = self.archive_snapshot
                c.require(snapshot is not None, "Exact source archive fingerprint required")
                archive = snapshot["path"]
                spec = self.cfg["archive_mapping"]
                image = PurePosixPath(name)
                if spec["kind"] == "prefix":
                    member = spec["prefix"] + name
                else:
                    c.require(spec["kind"] == "seed_original" and image.parts[0] == "SEED-Bench-image",
                              "Original SEED namespace required")
                    z, names = self.archive(archive)
                    choices = [x for x in (image.name, str(image)) if names[x] == 1]
                    c.require(len(choices) == 1, "Exact original SEED member absent/ambiguous")
                    member = choices[0]
            c.require(Path(archive).resolve().is_relative_to(Path(self.cfg["cache_root"]).resolve()) and
                      fingerprint(archive) == snapshot and not PurePosixPath(member).is_absolute()
                      and ".." not in PurePosixPath(member).parts,
                      "Source archive changed or member unsafe")
            z, names = self.archive(archive)
            c.require(names[member] == 1, "Duplicate/absent ZIP image")
            entry = z.getinfo(member)
            c.require(not entry.is_dir() and entry.file_size == expected["encoded_bytes"]
                      and (entry.external_attr >> 16) & 0o170000 != 0o120000,
                      "Original ZIP member type/size differs")
            with z.open(entry) as h:
                raw = h.read(64 * 1024**2 + 1)
            c.require(fingerprint(archive) == snapshot, "Source archive changed during image read")
        c.require(0 < len(raw) <= 64 * 1024**2 and len(raw) == expected["encoded_bytes"]
                  and hashlib.sha256(raw).hexdigest() == expected["encoded_sha256"],
                  "Actual native encoded image differs")
        return raw, expected

    def rgb(self, name):
        from PIL import Image
        raw, expected = self.payload(name)
        opening = self.opened or Image.open
        threshold = Image.MAX_IMAGE_PIXELS
        try:
            # The existing SEED qualification explicitly permits <=128MP.
            Image.MAX_IMAGE_PIXELS = 128_000_000
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with opening(io.BytesIO(raw)) as image:
                    c.require(image.width * image.height <= 128_000_000, "Native image exceeds original bound")
                    value = image.convert("RGB")
                    value.load()
        finally:
            Image.MAX_IMAGE_PIXELS = threshold
        c.require(value.size == (expected["width"], expected["height"]) and
                  hashlib.sha256(value.tobytes()).hexdigest() == expected["rgb_sha256"],
                  "Actual native RGB/dimensions differ")
        return value

    @contextmanager
    def installed(self):
        from PIL import Image
        previous = Image.open
        c.require(self.opened is None, "Nested image adapter forbidden")
        self.opened = previous
        def opened(path, *args, **kwargs):
            c.require(not args and not kwargs and isinstance(path, (str, Path)),
                      "Native loader requested undeclared image input")
            target = Path(path)
            c.require(target.is_absolute() and target.is_relative_to(self.virtual),
                      "Native loader image outside declared virtual namespace")
            return self.rgb(str(target.relative_to(self.virtual)))
        Image.open = opened
        try:
            yield
        finally:
            Image.open, self.opened = previous, None
