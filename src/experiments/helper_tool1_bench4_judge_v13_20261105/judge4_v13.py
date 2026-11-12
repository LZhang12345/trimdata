"""BENCH4 GPT-judge kernels v1.1 (stdlib only): MM-Vet official grader and LLaVA-Bench-in-the-Wild relative score.

v1.1 supersedes the never-run v1 judge4.py (helper_tool1_bench4_eval_v1_20261104; no v1 judge plan exists) and the
unbound draft judge4_v1_1.py; author-approved final policy of 2026-11-04 ("do not save API; use the newest models"):
  PRIMARY       gpt-4o-2024-08-06, temperature 0, one run, every cell, MM-Vet + LLaVA-W (unchanged from v1; official
                parse incl. the official MM-Vet 0.0 after five failed retries and LLaVA-W [-1, -1] reviews).
  CALIBRATION   gpt-4-0613 (the official MM-Vet judge), temperature 0, MM-Vet only, CALIBRATION_CELLS in priority
                order uniform-s17, coincide-s17, uniform-s29, coincide-s29, uniform-s43, coincide-s43.
  SUPPLEMENTARY gpt-5.6-terra (newest judge; alias verified in the /v1/models list inside the job, the dated
                snapshot returned by the API is recorded and must stay identical across the campaign), every cell,
                both banks, the same official prompts and parse formats. As a reasoning model it is sent only
                `max_completion_tokens` (temperature/max_tokens are not sent; model-default reasoning effort); a final
                parse failure is recorded as MISSING and excluded from the aggregate (never guessed, never 0.0).
Spend order is enforced with a priority reservation over the whole campaign roster: a run of rank r may spend only
while spend + call maximum + the estimated cost of every not-yet-complete run of a higher priority (lower rank) stays
under STOP_USD. PRIMARY (rank 0) reserves nothing, CALIBRATION (ranks 1..6) reserves every unfinished PRIMARY run and
earlier calibration cell, SUPPLEMENTARY (rank 100) reserves every unfinished PRIMARY and CALIBRATION run.

Official references are pinned by bytes and parsed, not retyped:
  MM-Vet   yuweihao/MM-Vet@acde843e mm-vet_evaluator.py: few-shot `prompt` (module constant, read with ast), one user
           message `prompt + "\\n" + " | ".join([question, answer with spaced <AND>/<OR>, prediction, ""])`,
           max_tokens 3, score = float(first space token) in [0, 1]; on a parse failure the official retry appends
           "\\nPredict the correctness of the answer (digit): " with temperature 0, 0.5, 1.0, 1.5, 2.0 and gives 0.0
           after the fifth retry. Capability scores = 100 * mean over the questions holding the capability; total =
           100 * mean; capability-integration sets as in export_result (one run, so std 0).
  LLaVA-W  LLaVA c121 llava/eval/eval_gpt_review_bench.py + llava/eval/table/rule.json: system message, the
           [Context]/[Question]/[Assistant 1]/[Assistant 2]/[System] user content with the GPT-4 reference answer as
           Assistant 1 and the evaluated model as Assistant 2, max_tokens 1024, parse_score of the first line;
           summarize_gpt_review.py: per category and overall, stats = round(mean(tuple), 3) and relative score
           round(100 * stats[1] / stats[0], 1).

Cost control: every billable call is preceded by an fsynced INTENT line in one append-only ledger (shared by every
BENCH4 judge version and job, exclusive flock) and followed by its RESULT/FAILURE line with actual usage tokens priced
by the pinned table (usage.completion_tokens, which includes reasoning tokens, is priced as output; cached-input
discounts are ignored, so recorded costs are upper bounds). Spend = sum(actual result costs) + conservative maximum for
possibly billed failures and for intents without an outcome. OK results are a durable cache keyed by (model, bank,
cell, unit, call index, request sha256), so a rerun replays them and never pays twice. Concurrent calls of one run
share one lock for every budget check + intent and every outcome line. The API key is read in-process from a 0600
owner file, kept in a redacting holder, set only on the in-memory request header and never written anywhere; API error
bodies are reduced to their type/code/param enums (an error message can echo a masked key, so it is never kept).
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
import re
import socket
import stat
import threading
import time
import urllib.error
import urllib.request


VERSION = "v1.1"
PRIMARY = "gpt-4o-2024-08-06"
CALIBRATION = "gpt-4-0613"
SUPPLEMENTARY = "gpt-5.6-terra"
SUPPLEMENTARY_RETURNED = r"gpt-5\.6-terra(?:-\d{4}-\d{2}-\d{2})?"
ROLES = {PRIMARY: "PRIMARY", CALIBRATION: "CALIBRATION", SUPPLEMENTARY: "SUPPLEMENTARY"}
PRICES_USD_PER_MTOK = {PRIMARY: {"input": 2.50, "output": 10.00}, CALIBRATION: {"input": 30.00, "output": 60.00},
                       SUPPLEMENTARY: {"input": 2.00, "output": 12.00}}
PRICE_POLICY = ("author-pinned web-sourced Nov 2026 prices per 1M tokens; usage.prompt_tokens at the input price, "
                "usage.completion_tokens (including reasoning tokens) at the output price; cached-input discounts "
                "ignored (recorded costs are upper bounds)")
HARD_BUDGET_USD = 81.29
STOP_USD = 80.00
CALIBRATION_CELLS = ("uniform-s17", "coincide-s17", "uniform-s29", "coincide-s29", "uniform-s43", "coincide-s43")
JUDGE_BANKS = ("mm-vet", "llava-wild")
TEMPERATURE = 0.0
ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODELS_ENDPOINT = "https://api.openai.com/v1/models"
KEY_PATH = "/home/luyangz/.config/dsmm_tool1_secrets/openai_api_key"
UID = 2708899
WILD_SYSTEM = "You are a helpful and precise assistant for checking the quality of the answer."
MMVET_RETRY_SUFFIX = "\nPredict the correctness of the answer (digit): "
MMVET_MAX_TOKENS = 3
WILD_MAX_TOKENS = 1024
OFFICIAL_MAX_TOKENS = {"mm-vet": MMVET_MAX_TOKENS, "llava-wild": WILD_MAX_TOKENS}
REASONING_MAX_COMPLETION_TOKENS = {"mm-vet": 4096, "llava-wild": 8192}
MMVET_TRY_LIMIT = 5
RETRYABLE_HTTP = (408, 409, 429, 500, 502, 503, 504)
MAX_TRANSPORT_RETRIES = 8
REQUEST_TIMEOUT = 300
CONCURRENCY = 4
PROJECTED_OUTPUT_TOKENS = {PRIMARY: {"mm-vet": 3, "llava-wild": 400}, CALIBRATION: {"mm-vet": 3, "llava-wild": 400},
                           SUPPLEMENTARY: {"mm-vet": 600, "llava-wild": 1600}}
RESERVE_FLOOR_USD = {(PRIMARY, "mm-vet"): 0.65, (PRIMARY, "llava-wild"): 0.40, (CALIBRATION, "mm-vet"): 7.00,
                     (SUPPLEMENTARY, "mm-vet"): 1.40, (SUPPLEMENTARY, "llava-wild"): 1.10}
RESERVE_OBSERVED_FACTOR = 1.10
SUPPLEMENTARY_RANK = 100
LEDGER_SCHEMA = "tool1-bench4-judge-ledger-v11"
ACCEPTED_LEDGER_SCHEMAS = ("tool1-bench4-judge-ledger-v1", LEDGER_SCHEMA)
REPEAT_POLICY = "ONE_RUN_PER_JUDGE_MODEL_OFFICIAL_MMVET_PARSE_RETRIES_ONLY"
MISSING_POLICY = {
    "PRIMARY": "OFFICIAL: MM-Vet 0.0 after five failed retries; LLaVA-W [-1, -1] kept in the official means",
    "CALIBRATION": "OFFICIAL: MM-Vet 0.0 after five failed retries",
    "SUPPLEMENTARY": "MISSING: a question whose final judge reply is unparsable (MM-Vet after the official five "
                     "retries; LLaVA-W first-line parse_score [-1, -1]) is recorded as missing and excluded from the "
                     "aggregate; never guessed",
}
SAFE_ERROR_TOKEN = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


class JudgeError(ValueError):
    pass


class BudgetStop(JudgeError):
    pass


class JudgeAbort(JudgeError):
    pass


def require(condition, message):
    if not condition:
        raise JudgeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def role(model):
    require(model in ROLES, "Unregistered judge model")
    return ROLES[model]


def missing_mode(model):
    return role(model) == "SUPPLEMENTARY"


# ------------------------------------------------------------------------------------------ pinned references
def mmvet_prompt(evaluator_raw):
    """The official module-level few-shot `prompt` string of mm-vet_evaluator.py (read with ast, never retyped)."""
    found = [node.value.value for node in ast.parse(evaluator_raw).body
             if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "prompt" and isinstance(node.value, ast.Constant)
             and isinstance(node.value.value, str)]
    require(len(found) == 1, "Exactly one official MM-Vet prompt constant required")
    return found[0]


def wild_rules(rule_raw, review_raw):
    """Official rule.json entries for the three LLaVA-Bench categories; the system message must be verbatim."""
    rules = json.loads(rule_raw)
    require(("'content': '%s'" % WILD_SYSTEM).encode() in review_raw, "Official review system message differs")
    selected = {}
    for category in ("conv", "detail", "complex"):
        rule = rules["llava_bench_" + category]
        require(isinstance(rule.get("prompt"), str) and isinstance(rule.get("role"), str), "Official rule fields")
        selected["llava_bench_" + category] = {"prompt": rule["prompt"], "role": rule["role"]}
    return selected


def prompt_hashes(mmvet_text, rules):
    return {"mmvet_fewshot_prompt_sha256": sha(mmvet_text.encode()),
            "mmvet_retry_suffix_sha256": sha(MMVET_RETRY_SUFFIX.encode()),
            "wild_system_sha256": sha(WILD_SYSTEM.encode()),
            "wild_rules_sha256": sha(canonical(rules))}


# ----------------------------------------------------------------------------------------------- MM-Vet port
def mmvet_question(prompt, label, prediction, retry=False):
    text = prompt + "\n" + " | ".join([label["question"],
                                       label["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                                       prediction, ""])
    return text + MMVET_RETRY_SUFFIX if retry else text


def mmvet_parse(content):
    """Official: content.split(' ')[0].strip() -> float within [0, 1]; non-finite values are treated as failures."""
    try:
        token = content.split(" ")[0].strip()
        score = float(token)
    except (AttributeError, ValueError, TypeError):
        return None, content
    if not math.isfinite(score) or score > 1.0 or score < 0.0:
        return None, token
    return score, token


def mmvet_grade(call, prompt, label, prediction, missing_on_failure=False):
    """Official runs() inner loop for one sample and one run. `call(messages, temperature, max_tokens)` returns a
    response record with `content`; returns (score, recorded content, calls). The official result after the fifth
    failed retry is 0.0; with missing_on_failure (SUPPLEMENTARY) the same call schedule ends in None (missing)."""
    messages = [{"role": "user", "content": mmvet_question(prompt, label, prediction)}]
    temperature = TEMPERATURE
    record = call(messages, temperature, MMVET_MAX_TOKENS)
    calls, content, try_time = [record], record["content"], 1
    while True:
        score, content = mmvet_parse(content)
        if score is not None:
            return score, content, calls
        messages = [{"role": "user", "content": mmvet_question(prompt, label, prediction, retry=True)}]
        record = call(messages, temperature, MMVET_MAX_TOKENS)
        calls.append(record)
        content = record["content"]
        try_time += 1
        temperature += 0.5
        if try_time > MMVET_TRY_LIMIT:
            return (None if missing_on_failure else 0.0), content, calls


def mmvet_aggregate(labels, scores, decimal_places=1):
    """export_result with num_run=1 over the full set: capability and capability-integration scores, total."""
    require(len(labels) == len(scores) and len(labels) > 0, "One score per MM-Vet question required")
    counter, sums, sets = {}, {"total": 0.0}, {}
    for label in labels:
        caps = sorted(set(label["capability"]))
        name = "_".join(caps)
        sets.setdefault(name, [0, 0.0])
        sets[name][0] += 1
        for cap in caps:
            counter[cap] = counter.get(cap, 0) + 1
    for label in labels:
        score = scores[label["key"]]
        caps = sorted(set(label["capability"]))
        for cap in caps:
            sums[cap] = sums.get(cap, 0.0) + score
        sums["total"] += score
        sets["_".join(caps)][1] += score
    counter["total"] = len(labels)
    unrounded = {k: v / counter[k] * 100 for k, v in sums.items()}
    integration = {k: v[1] / v[0] * 100 for k, v in sets.items()}
    return {"total": round(unrounded["total"], decimal_places),
            "capabilities": {k: round(v, decimal_places) for k, v in sorted(unrounded.items()) if k != "total"},
            "capability_integration": {k: round(v, decimal_places) for k, v in sorted(integration.items())},
            "unrounded": {"total": unrounded["total"],
                          "capabilities": {k: v for k, v in sorted(unrounded.items()) if k != "total"},
                          "capability_integration": dict(sorted(integration.items()))},
            "counts": dict(sorted(counter.items())), "integration_counts": {k: v[0] for k, v in sorted(sets.items())},
            "std": 0.0, "runs": 1, "decimal_places": decimal_places,
            "integration_name_rule": "sorted capability names joined by '_' (official uses set iteration order)"}


def mmvet_aggregate_present(labels, scores):
    """Official aggregate over the questions with a parsed score; missing questions are counted, never imputed."""
    require(len(labels) == len(scores) and len(labels) > 0, "One score or None per MM-Vet question required")
    present = [label for label in labels if scores[label["key"]] is not None]
    missing = sorted(label["key"] for label in labels if scores[label["key"]] is None)
    if present:
        result = mmvet_aggregate(present, {label["key"]: scores[label["key"]] for label in present})
    else:
        result = {"total": None, "capabilities": {}, "capability_integration": {}, "unrounded": {"total": None},
                  "counts": {"total": 0}, "integration_counts": {}, "std": 0.0, "runs": 1, "decimal_places": 1}
    result.update(questions=len(labels), scored_questions=len(present), missing=len(missing), missing_keys=missing,
                  aggregate_over="parsed questions only (missing excluded)")
    return result


# --------------------------------------------------------------------------------------------- LLaVA-W port
def wild_content(rules, label, prediction):
    rule = rules["llava_bench_" + label["category"]]
    role_name, prompt = rule["role"], rule["prompt"]
    return (f'[Context]\n{label["caption"]}\n\n'
            f'[Question]\n{label["question"]}\n\n'
            f'[{role_name} 1]\n{label["reference_text"]}\n\n[End of {role_name} 1]\n\n'
            f'[{role_name} 2]\n{prediction}\n\n[End of {role_name} 2]\n\n'
            f'[System]\n{prompt}\n\n')


def wild_messages(rules, label, prediction):
    return [{"role": "system", "content": WILD_SYSTEM},
            {"role": "user", "content": wild_content(rules, label, prediction)}]


def wild_parse(review):
    """Official parse_score."""
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        return [-1, -1]
    except Exception:
        return [-1, -1]


def wild_aggregate(reviews):
    """summarize_gpt_review.py for one review file: per category and 'all'."""
    require(reviews, "Reviews required")
    groups = {}
    for review in reviews:
        groups.setdefault(review["category"], []).append(review["tuple"])
        groups.setdefault("all", []).append(review["tuple"])
    table = {}
    for name, tuples in sorted(groups.items()):
        means = [sum(t[i] for t in tuples) / len(tuples) for i in (0, 1)]
        stats = [round(x, 3) for x in means]
        table[name] = {"relative_score": round(stats[1] / stats[0] * 100, 1) if stats[0] else None,
                       "reference_avg_x10": round(stats[0] * 10, 1), "model_avg_x10": round(stats[1] * 10, 1),
                       "unrounded_relative": (means[1] / means[0] * 100) if means[0] else None,
                       "reviews": len(tuples), "unparsed": sum(list(t) == [-1, -1] for t in tuples)}
    return {"overall": table["all"]["relative_score"], "categories": table}


def wild_aggregate_present(reviews):
    """summarize_gpt_review.py over the parsed reviews only; unparsable reviews are counted as missing."""
    require(reviews, "Reviews required")
    present = [review for review in reviews if not review["missing"]]
    result = wild_aggregate(present) if present else {"overall": None, "categories": {}}
    result.update(reviews=len(reviews), scored_reviews=len(present), missing=len(reviews) - len(present),
                  missing_question_ids=[review["question_id"] for review in reviews if review["missing"]],
                  aggregate_over="parsed reviews only (missing excluded)")
    return result


def judge_agreement(primary, other):
    """Per-question agreement of two MM-Vet judges over the same keys; pairs with a missing score are excluded."""
    keys = sorted(primary)
    require(keys == sorted(other) and keys, "Same MM-Vet questions required")
    present = [k for k in keys if primary[k] is not None and other[k] is not None]
    value = {"questions": len(keys), "paired_questions": len(present),
             "missing_primary": sum(primary[k] is None for k in keys), "missing_other": sum(other[k] is None for k in keys)}
    if not present:
        return dict(value, exact_agreement=None, within_0_1=None, mean_absolute_difference=None, pearson=None,
                    primary_total=None, other_total=None, total_difference=None)
    a, b = [primary[k] for k in present], [other[k] for k in present]
    n = len(present)
    ma, mb = sum(a) / n, sum(b) / n
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return dict(value, exact_agreement=sum(x == y for x, y in zip(a, b)) / n,
                within_0_1=sum(abs(x - y) <= 0.1 + 1e-9 for x, y in zip(a, b)) / n,
                mean_absolute_difference=sum(abs(x - y) for x, y in zip(a, b)) / n,
                pearson=cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None,
                primary_total=ma * 100, other_total=mb * 100, total_difference=(ma - mb) * 100)


# --------------------------------------------------------------------------------------------- request bodies
def request_body(model, bank, messages, temperature, max_tokens):
    """Chat judges get the official temperature/max_tokens; the reasoning judge gets only max_completion_tokens."""
    require(model in PRICES_USD_PER_MTOK and bank in JUDGE_BANKS, "Unpriced judge model or unknown bank")
    require(max_tokens == OFFICIAL_MAX_TOKENS[bank], "Official max_tokens of the bank required")
    if role(model) == "SUPPLEMENTARY":
        return {"model": model, "messages": messages, "max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS[bank]}
    return {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}


def request_parameters(model, bank):
    """What is sent for (model, bank) besides model/messages; recorded in plans and SCORE files."""
    if role(model) == "SUPPLEMENTARY":
        return {"sent": {"max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS[bank]},
                "not_sent": ["temperature", "max_tokens", "reasoning_effort"],
                "temperature": "model default (reasoning model; the official MM-Vet retry temperature schedule "
                               "is not sent)",
                "reasoning_effort": "model default (not sent)",
                "official_visible_max_tokens": OFFICIAL_MAX_TOKENS[bank]}
    return {"sent": {"temperature": "0.0 (MM-Vet official retries 0.5..2.0)" if bank == "mm-vet" else 0.0,
                     "max_tokens": OFFICIAL_MAX_TOKENS[bank]}, "not_sent": []}


def output_cap(body):
    return body["max_completion_tokens"] if "max_completion_tokens" in body else body["max_tokens"]


# --------------------------------------------------------------------------------------------- cost & ledger
def call_cost(model, prompt_tokens, completion_tokens):
    price = PRICES_USD_PER_MTOK[model]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1e6


def max_call_cost(model, messages, max_tokens):
    """Upper bound: no more prompt tokens than UTF-8 bytes plus per-message overhead."""
    size = sum(len(m["content"].encode("utf-8")) + 16 for m in messages) + 16
    return call_cost(model, size, max_tokens)


def projected_call_cost(model, bank, messages):
    size = sum(len(m["content"].encode("utf-8")) for m in messages)
    return call_cost(model, size / 3.5 + 20, PROJECTED_OUTPUT_TOKENS[model][bank])


class Ledger:
    """Append-only, fsynced JSONL shared by every BENCH4 judge job; exclusive non-blocking flock while open."""

    def __init__(self, path, job, flock=None):
        import fcntl
        self.path, self.job = str(path), str(job)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077,
                    "Private owned judge ledger required")
            (flock or fcntl.flock)(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
        except BaseException:
            os.close(self.fd)
            raise

    def _load(self):
        self.intents, self.outcomes, self.cache, self.completed, self.returned, self.torn = {}, {}, {}, {}, {}, 0
        raw = b""
        os.lseek(self.fd, 0, os.SEEK_SET)
        while True:
            block = os.read(self.fd, 8 << 20)
            if not block:
                break
            raw += block
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        elif lines:
            self.torn += 1
            lines.pop()
            os.write(self.fd, b"\n")
            os.fsync(self.fd)
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                self.torn += 1
                continue
            self._index(record)

    def _index(self, record):
        require(record.get("schema") in ACCEPTED_LEDGER_SCHEMAS
                and record.get("type") in ("intent", "result", "failure", "run_complete"), "Foreign ledger record")
        if record["type"] == "intent":
            self.intents[record["call_id"]] = record
        elif record["type"] == "run_complete":
            self.completed.setdefault((record["model"], record["bank"], record["cell"]), record)
        else:
            self.outcomes[record["call_id"]] = record
            intent = self.intents.get(record["call_id"])
            if record["type"] == "result" and intent is not None:
                self.cache.setdefault(intent["cache_key"], record)
                self.returned.setdefault(intent["model"], set()).add(record["response"]["model"])

    def _append(self, record):
        record = dict(record, schema=LEDGER_SCHEMA)
        os.write(self.fd, canonical(record))
        os.fsync(self.fd)
        self._index(record)
        return record

    def spend(self):
        total = 0.0
        for call_id, intent in self.intents.items():
            outcome = self.outcomes.get(call_id)
            if outcome is None:
                total += intent["max_cost_usd"]
            elif outcome["type"] == "result":
                total += outcome["cost_usd"]
            else:
                total += outcome["charged_usd"]
        return total

    def by_model(self):
        totals = {}
        for call_id, intent in self.intents.items():
            outcome = self.outcomes.get(call_id)
            value = intent["max_cost_usd"] if outcome is None else outcome.get("cost_usd", outcome.get("charged_usd"))
            totals[intent["model"]] = totals.get(intent["model"], 0.0) + value
        return totals

    def run_cost(self, model, bank, cell):
        total, calls = 0.0, 0
        for call_id, intent in self.intents.items():
            if (intent["model"], intent["bank"], intent["cell"]) != (model, bank, cell):
                continue
            outcome = self.outcomes.get(call_id)
            total += intent["max_cost_usd"] if outcome is None else outcome.get("cost_usd", outcome.get("charged_usd"))
            calls += 1
        return dict(usd=total, ledger_calls=calls)

    def returned_models(self, model):
        """Model ids the API returned for requests of `model` (results only)."""
        return set(self.returned.get(model, ()))

    def intent(self, **fields):
        fields["call_id"] = "%s-%d-%s" % (self.job, len(self.intents), os.urandom(4).hex())
        return self._append(dict(fields, type="intent", job=self.job, time=time.time()))

    def result(self, call_id, response, cost):
        return self._append(dict(type="result", call_id=call_id, response=response, cost_usd=cost, time=time.time()))

    def failure(self, call_id, status, detail, charged):
        return self._append(dict(type="failure", call_id=call_id, status=status, detail=detail[:500],
                                 charged_usd=charged, time=time.time()))

    def run_complete(self, model, bank, cell, score, cost):
        """Idempotent: the first completion of (model, bank, cell) is kept; a replay appends nothing."""
        existing = self.completed.get((model, bank, cell))
        if existing is not None:
            require(existing["score_sha256"] == score["sha256"], "A completed judge run published other bytes")
            return existing
        return self._append(dict(type="run_complete", model=model, bank=bank, cell=cell, job=self.job,
                                 score_sha256=score["sha256"], score_path=score["path"], cost_usd=cost,
                                 time=time.time()))

    def close(self):
        os.close(self.fd)


# ------------------------------------------------------------------------------------- spend-order reservation
def campaign_runs(cells):
    """Whole-campaign roster with priority ranks (lower rank = paid first)."""
    require(len(set(cells)) == len(cells) and set(CALIBRATION_CELLS) <= set(cells), "Campaign roster required")
    runs = [dict(model=PRIMARY, bank=bank, cell=cell, role="PRIMARY", rank=0) for cell in cells for bank in JUDGE_BANKS]
    runs += [dict(model=CALIBRATION, bank="mm-vet", cell=cell, role="CALIBRATION", rank=1 + index)
             for index, cell in enumerate(CALIBRATION_CELLS)]
    runs += [dict(model=SUPPLEMENTARY, bank=bank, cell=cell, role="SUPPLEMENTARY", rank=SUPPLEMENTARY_RANK)
             for cell in cells for bank in JUDGE_BANKS]
    return runs


def batch_runs(cells, campaign):
    """Ordered runs of one batch: PRIMARY (cell order, mm-vet then llava-wild), CALIBRATION (priority order),
    SUPPLEMENTARY (cell order, mm-vet then llava-wild)."""
    require(cells and len(set(cells)) == len(cells), "Unique batch cells required")
    index = {(r["model"], r["bank"], r["cell"]): r for r in campaign}
    require(all((PRIMARY, "mm-vet", cell) in index for cell in cells), "Batch cell outside the campaign roster")
    runs = [index[(PRIMARY, bank, cell)] for cell in cells for bank in JUDGE_BANKS]
    runs += [index[(CALIBRATION, "mm-vet", cell)] for cell in CALIBRATION_CELLS if cell in cells]
    runs += [index[(SUPPLEMENTARY, bank, cell)] for cell in cells for bank in JUDGE_BANKS]
    return [dict(r) for r in runs]


def run_estimate(ledger, model, bank):
    observed = [c["cost_usd"] for c in ledger.completed.values() if (c["model"], c["bank"]) == (model, bank)]
    return max([RESERVE_FLOOR_USD[(model, bank)]] + [RESERVE_OBSERVED_FACTOR * x for x in observed])


def reservation(ledger, campaign, run, unavailable=()):
    """Estimated cost of every unfinished higher-priority campaign run (models unavailable in this job excluded)."""
    total = 0.0
    for other in campaign:
        key = (other["model"], other["bank"], other["cell"])
        if other["rank"] < run["rank"] and key not in ledger.completed and other["model"] not in unavailable:
            total += run_estimate(ledger, other["model"], other["bank"])
    return total


# ------------------------------------------------------------------------------------------------- transport
class Secret:
    """Redacting holder: the key never appears in repr/str, receipts, logs, argv or the environment."""
    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return "<redacted>"

    __str__ = __repr__

    def header(self):
        return "Bearer " + self._value


def read_key(path=KEY_PATH, uid=UID):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and not info.st_mode & 0o077
                and 0 < info.st_size <= 1024, "Owner-only (0600) regular API key file required")
        raw = os.read(fd, 1025)
    finally:
        os.close(fd)
    value = raw.decode("ascii", "replace").strip()
    require(re.fullmatch(r"[A-Za-z0-9_\-]{20,400}", value) is not None, "API key file has an unexpected format")
    return Secret(value)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the Authorization header must reach only the pinned endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def https_transport(secret, body, timeout=REQUEST_TIMEOUT):
    """One POST to the pinned endpoint; returns (http_status, parsed JSON). A redirect surfaces as an HTTPError."""
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": secret.header()})
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        require(response.geturl() == ENDPOINT, "Judge response did not come from the pinned endpoint")
        return response.status, json.loads(response.read(8 << 20))


def https_models(secret, timeout=60):
    """GET the pinned models list (no tokens billed); returns the sorted model ids."""
    request = urllib.request.Request(MODELS_ENDPOINT, method="GET", headers={"Authorization": secret.header()})
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        require(response.geturl() == MODELS_ENDPOINT, "Models list did not come from the pinned endpoint")
        payload = json.loads(response.read(16 << 20))
    ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    require(ids and all(isinstance(x, str) for x in ids), "Models list without ids")
    return sorted(ids)


def model_availability(ids):
    """Which judge ids are listed; the supplementary alias must be listed verbatim (no automatic substitution)."""
    listed = set(ids)
    related = sorted(x for x in ids if x in (PRIMARY, CALIBRATION) or x.startswith(SUPPLEMENTARY))
    return {"available": {model: model in listed for model in ROLES}, "related_ids": related,
            "supplementary_listed_snapshots": sorted(x for x in ids if re.fullmatch(SUPPLEMENTARY_RETURNED, x)),
            "listed_total": len(ids)}


def safe_error(error):
    """type/code/param enums of an API error body; the free-text message is dropped (it can echo a masked key)."""
    try:
        body = json.loads(error.read(65536))
        fields = body.get("error") or {}
    except Exception:
        return {}
    return {k: fields[k] for k in ("type", "code", "param")
            if isinstance(fields.get(k), str) and SAFE_ERROR_TOKEN.fullmatch(fields[k])}


def response_record(model, status, payload):
    choice = payload["choices"][0]
    usage = payload["usage"]
    returned = payload.get("model")
    if role(model) == "SUPPLEMENTARY":
        require(isinstance(returned, str) and re.fullmatch(SUPPLEMENTARY_RETURNED, returned) is not None,
                "Judge returned another model id: %r" % (returned,))
    else:
        require(returned == model, "Judge returned another model id: %r" % (returned,))
    require(type(usage.get("prompt_tokens")) is int and type(usage.get("completion_tokens")) is int,
            "Usage tokens required")
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    content = choice["message"].get("content")
    require(content is None or isinstance(content, str), "Judge content must be text or null")
    return {"id": payload.get("id"), "model": returned, "system_fingerprint": payload.get("system_fingerprint"),
            "created": payload.get("created"), "content": content,
            "finish_reason": choice.get("finish_reason"), "http_status": status,
            "usage": {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                      "reasoning_tokens": details.get("reasoning_tokens") if type(details.get("reasoning_tokens"))
                      is int else None,
                      "cached_prompt_tokens": prompt_details.get("cached_tokens")
                      if type(prompt_details.get("cached_tokens")) is int else None}}


class Client:
    """Ledger-backed, budget- and priority-guarded, cached judge calls; safe for CONCURRENCY worker threads."""

    def __init__(self, ledger, transport, stop_usd=STOP_USD, sleep=time.sleep, deadline=float("inf")):
        self.ledger, self.transport, self.stop_usd, self.sleep, self.deadline = ledger, transport, stop_usd, sleep, deadline
        self.paid_calls, self.cached_calls, self.reserve_usd, self.halted = 0, 0, 0.0, None
        self.lock = threading.Lock()

    def halt(self, reason):
        with self.lock:
            self.halted = self.halted or reason

    def caller(self, model, bank, cell, unit):
        index = [0]

        def call(messages, temperature, max_tokens):
            body = request_body(model, bank, messages, temperature, max_tokens)
            request_sha = sha(canonical(body))
            key = sha(canonical([model, bank, cell, str(unit), index[0], request_sha]))
            index[0] += 1
            with self.lock:
                cached = self.ledger.cache.get(key)
                if cached is not None:
                    self.cached_calls += 1
                    return dict(cached["response"], cached=True, request_sha256=request_sha)
            return self.paid(model, bank, cell, unit, index[0] - 1, body, request_sha, key)
        return call

    def paid(self, model, bank, cell, unit, call_index, body, request_sha, key):
        maximum = max_call_cost(model, body["messages"], output_cap(body))
        parameters = {k: v for k, v in body.items() if k not in ("messages", "model")}
        for attempt in range(MAX_TRANSPORT_RETRIES):
            with self.lock:
                if self.halted:
                    raise BudgetStop("Halted after a sibling call stopped: " + self.halted)
                if time.time() > self.deadline:
                    raise BudgetStop("Allocation deadline reached before the next judge call")
                spend = self.ledger.spend()
                if spend + maximum + self.reserve_usd > self.stop_usd:
                    raise BudgetStop("BUDGET: spend %.4f + call maximum %.4f + higher-priority reservation %.4f "
                                     "would exceed the %.2f USD stop" % (spend, maximum, self.reserve_usd,
                                                                         self.stop_usd))
                intent = self.ledger.intent(model=model, bank=bank, cell=cell, unit=str(unit), call_index=call_index,
                                            request_sha256=request_sha, cache_key=key, max_cost_usd=maximum,
                                            request_parameters=parameters, transport_attempt=attempt)
            try:
                status, payload = self.transport(body)
            except urllib.error.HTTPError as error:
                detail = dict(safe_error(error), http=error.code)
                with self.lock:
                    self.ledger.failure(intent["call_id"], "HTTP_%d" % error.code, json.dumps(detail, sort_keys=True),
                                        0.0)
                if error.code not in RETRYABLE_HTTP:
                    raise JudgeAbort("Non-retryable judge API error for %s: %s" % (model, json.dumps(
                        detail, sort_keys=True)))
            except JudgeError as error:
                with self.lock:
                    self.ledger.failure(intent["call_id"], "FOREIGN_ENDPOINT", repr(error), maximum)
                raise JudgeAbort("Judge response did not come from the pinned endpoint")
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError,
                    http.client.HTTPException, ValueError) as error:
                # Truncated/undecodable bodies (IncompleteRead, JSONDecodeError) are transport failures that may
                # have been billed: charged at the call maximum, then retried within the bounded transport loop.
                with self.lock:
                    self.ledger.failure(intent["call_id"], "TRANSPORT", type(error).__name__, maximum)
            else:
                with self.lock:
                    try:
                        record = response_record(model, status, payload)
                        pinned = self.ledger.returned_models(model)
                        require(not pinned or pinned == {record["model"]},
                                "Judge snapshot changed within the campaign: %r vs %r" % (sorted(pinned),
                                                                                         record["model"]))
                    except (KeyError, IndexError, TypeError, AttributeError, JudgeError) as error:
                        usage = payload.get("usage") if isinstance(payload, dict) else None
                        charged = maximum
                        if isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int \
                                and type(usage.get("completion_tokens")) is int:
                            charged = call_cost(model, usage["prompt_tokens"], usage["completion_tokens"])
                        self.ledger.failure(intent["call_id"], "MALFORMED_RESPONSE", repr(error), charged)
                        raise JudgeAbort("Malformed, foreign or snapshot-changed judge response for %s" % model)
                    cost = call_cost(model, record["usage"]["prompt_tokens"], record["usage"]["completion_tokens"])
                    self.ledger.result(intent["call_id"], record, cost)
                    self.paid_calls += 1
                return dict(record, cached=False, request_sha256=request_sha)
            self.sleep(min(60, 2 ** (attempt + 1)))
        raise JudgeAbort("Judge API unavailable after bounded retries")


def run_parallel(client, items, function, workers=CONCURRENCY):
    """function(item) for every item on a small thread pool; results in item order. The first error halts every
    sibling's next paid call; a JudgeAbort outranks a BudgetStop when re-raised."""
    results, errors = [None] * len(items), []

    def guarded(position):
        try:
            results[position] = function(items[position])
        except BaseException as error:
            errors.append(error)
            client.halt(type(error).__name__)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(guarded, range(len(items))))
    if errors:
        aborts = [e for e in errors if not isinstance(e, BudgetStop)]
        raise (aborts or errors)[0]
    return results


def project(model, bank, units, ledger):
    """Projected cost of the not-yet-cached first calls of `units` [(cell, unit, messages)]."""
    total = 0.0
    for cell, unit, messages in units:
        body = request_body(model, bank, messages, TEMPERATURE, OFFICIAL_MAX_TOKENS[bank])
        key = sha(canonical([model, bank, cell, str(unit), 0, sha(canonical(body))]))
        if key not in ledger.cache:
            total += projected_call_cost(model, bank, messages)
    return total


# --------------------------------------------------------------------------------------------- bank scoring
def _fingerprints(records):
    values = {r.get("system_fingerprint") for r in records}
    return sorted(v for v in values if v is not None) + (["<none>"] if None in values else [])


def _usage(records):
    return {"prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in records),
            "reasoning_tokens": sum(r["usage"].get("reasoning_tokens") or 0 for r in records),
            "calls": len(records)}


def score_mmvet(client, model, cell, prompt, labels, predictions, workers=CONCURRENCY):
    """One complete MM-Vet run; returns the aggregate plus per-question judge evidence."""
    require([l["question_id"] for l in labels] == [p["question_id"] for p in predictions],
            "MM-Vet labels and predictions must align")
    missing = missing_mode(model)

    def grade(pair):
        label, prediction = pair
        return mmvet_grade(client.caller(model, "mm-vet", cell, label["key"]), prompt, label, prediction["text"],
                           missing_on_failure=missing)
    graded = run_parallel(client, list(zip(labels, predictions)), grade, workers)
    scores, evidence, records = {}, [], []
    for label, (score, content, calls) in zip(labels, graded):
        scores[label["key"]] = score
        records.extend(calls)
        evidence.append({"key": label["key"], "score": score, "missing": score is None, "content": content,
                         "calls": len(calls), "response_ids": [c.get("id") for c in calls],
                         "finish_reasons": [c.get("finish_reason") for c in calls],
                         "cached": [c["cached"] for c in calls]})
    result = mmvet_aggregate_present(labels, scores) if missing else mmvet_aggregate(labels, scores)
    result.update(retried_questions=sum(e["calls"] > 1 for e in evidence),
                  exhausted_retries=sum(e["calls"] > MMVET_TRY_LIMIT for e in evidence),
                  system_fingerprints=_fingerprints(records), judge_models_returned=sorted({r["model"] for r in records}),
                  usage=_usage(records))
    return result, scores, evidence


def score_wild(client, model, cell, rules, labels, predictions, workers=CONCURRENCY):
    require([l["question_id"] for l in labels] == [p["question_id"] for p in predictions],
            "LLaVA-W labels and predictions must align")
    missing = missing_mode(model)

    def review(pair):
        label, prediction = pair
        call = client.caller(model, "llava-wild", cell, label["question_id"])
        return call(wild_messages(rules, label, prediction["text"]), TEMPERATURE, WILD_MAX_TOKENS)
    records = run_parallel(client, list(zip(labels, predictions)), review, workers)
    reviews = []
    for index, (label, record) in enumerate(zip(labels, records)):
        content = record["content"] if isinstance(record["content"], str) else ""
        parsed = wild_parse(content)
        reviews.append({"id": index + 1, "question_id": label["question_id"], "category": "llava_bench_" +
                        label["category"], "tuple": parsed, "missing": missing and parsed == [-1, -1],
                        "content": content, "response_id": record.get("id"), "cached": record["cached"],
                        "finish_reason": record.get("finish_reason")})
    result = wild_aggregate_present(reviews) if missing else wild_aggregate(reviews)
    result.update(system_fingerprints=_fingerprints(records), judge_models_returned=sorted({r["model"] for r in records}),
                  usage=_usage(records))
    return result, reviews
