"""Restricted streaming legacy-role projection; skipped JSON strings are not decoded.

This is an input-identity prerequisite, never a protected-bank completion receipt.
Only structural keys and image/question ID scalars inside top-level ``roles`` are
decoded. All other JSON values are lexically validated and discarded. The full
source is byte-hashed, but closed question/answer strings are never materialized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

LEGACY_BYTES = 108392717
LEGACY_SHA = "151b44c923f449e7472c64815357a65a09976bb70cfff33d24467b6d7c62ca8f"
MAX_DEPTH = 96
NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def decode_metadata(raw):
    """The single decoding seam; deliberately never called by skip_value."""
    return json.loads(raw)


class Reader:
    def __init__(self, handle, maximum, block=65536):
        self.handle, self.maximum, self.block = handle, maximum, block
        self.buf, self.pos, self.eof = b"", 0, False
        self.hasher, self.bytes = hashlib.sha256(), 0
        self.decoded_strings = 0

    def peek(self):
        if self.pos == len(self.buf) and not self.eof:
            self.buf = self.handle.read(self.block)
            self.pos = 0
            if self.buf:
                self.bytes += len(self.buf)
                require(self.bytes <= self.maximum, "Legacy source exceeds exact byte budget")
                self.hasher.update(self.buf)
            else:
                self.eof = True
        return self.buf[self.pos] if self.pos < len(self.buf) else None

    def take(self):
        value = self.peek()
        require(value is not None, "Truncated JSON source")
        self.pos += 1
        return value

    def ws(self):
        while self.peek() in (9, 10, 13, 32):
            self.take()

    def expect(self, value):
        self.ws()
        require(self.take() == ord(value), "Malformed JSON punctuation")

    def string(self, decode=False):
        self.ws()
        require(self.take() == 34, "JSON string required")
        out = bytearray(b'"') if decode else None
        # Validate UTF-8 without materializing or decoding skipped content.
        continuation, low, high = 0, 128, 191
        while True:
            value = self.take()
            if out is not None:
                out.append(value)
                require(len(out) <= 4096, "Metadata key/ID exceeds bounded size")
            if continuation:
                require(low <= value <= high, "Malformed UTF-8 in JSON string")
                continuation -= 1
                low, high = 128, 191
                continue
            if value == 34:
                break
            require(value >= 32, "Unescaped control in JSON string")
            if value == 92:
                escaped = self.take()
                if out is not None:
                    out.append(escaped)
                if escaped == 117:
                    for _ in range(4):
                        digit = self.take()
                        require(digit in b"0123456789abcdefABCDEF", "Malformed unicode JSON escape")
                        if out is not None:
                            out.append(digit)
                else:
                    require(escaped in b'"\\/bfnrt', "Malformed JSON escape")
            elif value >= 128:
                if 194 <= value <= 223:
                    continuation = 1
                elif value == 224:
                    continuation, low = 2, 160
                elif 225 <= value <= 236 or 238 <= value <= 239:
                    continuation = 2
                elif value == 237:
                    continuation, high = 2, 159
                elif value == 240:
                    continuation, low = 3, 144
                elif 241 <= value <= 243:
                    continuation = 3
                elif value == 244:
                    continuation, high = 3, 143
                else:
                    raise ValueError("Malformed UTF-8 in JSON string")
        if decode:
            self.decoded_strings += 1
            return decode_metadata(bytes(out))

    def atom(self, decode=False):
        self.ws()
        out = bytearray()
        while self.peek() is not None and self.peek() not in b" \t\r\n,]}":
            out.append(self.take())
            require(len(out) <= 128, "Oversized JSON atom")
        raw = bytes(out)
        require(raw in (b"null", b"true", b"false") or NUMBER.fullmatch(raw), "Malformed JSON atom")
        return decode_metadata(raw) if decode else None

    def skip_value(self, depth=0):
        require(depth <= MAX_DEPTH, "JSON nesting limit exceeded")
        self.ws()
        value = self.peek()
        if value == 34:
            self.string(False)
        elif value in (123, 91):
            opening, closing = ("{", "}") if value == 123 else ("[", "]")
            self.expect(opening)
            self.ws()
            if self.peek() == ord(closing):
                self.take()
                return
            while True:
                if opening == "{":
                    self.string(False)
                    self.expect(":")
                self.skip_value(depth + 1)
                self.ws()
                token = self.take()
                if token == ord(closing):
                    return
                require(token == 44, "Malformed JSON container separator")
        else:
            self.atom(False)

    def object(self, callback):
        self.expect("{")
        self.ws()
        seen = set()
        if self.peek() == 125:
            self.take()
            return
        while True:
            name = self.string(True)
            require(name not in seen, "Duplicate structural JSON key")
            seen.add(name)
            self.expect(":")
            callback(name)
            self.ws()
            token = self.take()
            if token == 125:
                return
            require(token == 44, "Malformed object separator")

    def ids(self):
        self.expect("[")
        result = []
        self.ws()
        if self.peek() == 93:
            self.take()
            return result
        while True:
            self.ws()
            value = self.string(True) if self.peek() == 34 else self.atom(True)
            require(type(value) in (int, str) and not isinstance(value, bool), "ID must be integer or string")
            require(not isinstance(value, str) or (value and value == value.strip()), "Empty/noncanonical ID")
            require(not isinstance(value, int) or value >= 0, "Negative ID")
            result.append(value)
            require(len(result) <= 10000000, "Role ID roster exceeds bounded size")
            self.ws()
            token = self.take()
            if token == 93:
                return result
            require(token == 44, "Malformed ID-array separator")


def extract(handle, size, digest, block=65536):
    reader = Reader(handle, size, block)
    found, roles = [], {}

    def bank():
        result = {}
        def field(name):
            if name in ("image_ids", "question_ids"):
                result[name] = reader.ids()
            else:
                reader.skip_value()
        reader.object(field)
        require(set(result) == {"image_ids", "question_ids"}, "Role must preserve image/question ID fields")
        return result

    def role_field(name):
        if name == "banks":
            banks = {}
            def named_bank(key):
                require(key and key != "confirmation", "Invalid named role")
                banks[key] = bank()
            reader.object(named_bank)
            require(banks, "Empty bank roles")
            roles["banks"] = banks
        elif name == "confirmation":
            roles[name] = bank()
        else:
            reader.skip_value()

    def root_field(name):
        if name == "roles":
            found.append(True)
            reader.object(role_field)
        else:
            reader.skip_value()

    reader.object(root_field)
    reader.ws()
    require(reader.peek() is None, "Trailing bytes after top-level JSON")
    require(found and set(roles) == {"banks", "confirmation"}, "Exact banks and confirmation metadata required")
    require(reader.bytes == size and reader.hasher.hexdigest() == digest, "Exact full legacy source bytes/SHA differ")
    return roles, {"source_bytes_hashed": reader.bytes, "source_sha256": reader.hasher.hexdigest(),
                   "decoded_metadata_strings": reader.decoded_strings,
                   "closed_payload_strings_decoded": 0,
                   "projection": "only roles.banks[*].{image_ids,question_ids} and roles.confirmation.{image_ids,question_ids}"}


def seal(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        require(path.read_bytes() == payload, "Immutable output collision")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def describe(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(source, output):
    source, output = Path(source), Path(output)
    require(source.is_file() and source.stat().st_size == LEGACY_BYTES, "Exact registered legacy source missing")
    # Runtime must be an allocated registered CPU step within an owned GPU job.
    from experiments.root_native_input_authority_20261023.pool_authority_v1 import require_allocation
    allocation = require_allocation()
    with source.open("rb") as handle:
        roles, audit = extract(handle, LEGACY_BYTES, LEGACY_SHA)
    seal(output / "ROLES.json", roles)
    receipt = {"schema": "root-restricted-legacy-role-projection-v1",
               "status": "PASS_ROLES_IDS_ONLY_FULL_SOURCE_HASHED_NOT_PROTECTED_BANK_COMPLETE",
               "source": {"path": str(source), "bytes": LEGACY_BYTES, "sha256": LEGACY_SHA},
               "roles": describe(output / "ROLES.json"), "roles_sha256": hashlib.sha256(canonical(roles)).hexdigest(),
               "audit": audit, "allocation": allocation,
               "confirmation_QA_access": "image_only_closed_no_QA_contents",
               "producer": describe(__file__)}
    seal(output / "PROJECTION_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), sort_keys=True))
