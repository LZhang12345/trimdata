"""Explicit bounded authorized-input reader; frozen legacy Reader unchanged.

The original4096-byte metadata bound is too narrow for a general authorized
question. Only this new reader permits at most oneMiB. UTF8/escape validation,
lexical skip-without-decoding and all other structural semantics are inherited.
This class is never used to request a closed QA string; callers whitelist fields.
"""
from experiments.root_official_protected_inputs_v1_20261024.roles_projection_v2 import Reader as MetadataReader,require,decode_metadata
class Reader(MetadataReader):
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
                require(len(out) <= (1 << 20), "Authorized input string exceeds bounded oneMiB scope")
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

