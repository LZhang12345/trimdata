"""Use the pinned VizWiz official accuracy implementation without caption metrics."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import re
import sys
from pathlib import Path


OFFICIAL_SOURCE_SHA256 = 'd26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694'
OFFICIAL_SOURCE_BYTES = 11856


class ScoringError(ValueError):
    pass


class _Input:
    def __init__(self, rows):
        self.imgToQA = {row['image']: deepcopy(row) for row in rows}

    def getImgs(self):
        return list(self.imgToQA)


class _NoCaptionMetrics:
    def __init__(self, images, ground_truth, predictions):
        self.eval = {}

    def evaluate(self):
        pass


def _official_class(source):
    raw = Path(source).read_bytes()
    if len(raw) != OFFICIAL_SOURCE_BYTES or sha256(raw).hexdigest() != OFFICIAL_SOURCE_SHA256:
        raise ScoringError('Official VizWiz scorer source bytes or digest differ')
    tree = ast.parse(raw, filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VQAEval']
    if len(classes) != 1:
        raise ScoringError('Expected exactly one official VQAEval class')
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    namespace = {'__name__': 'pinned_official_vizwiz_accuracy', 're': re, 'sys': sys,
                 'COCOEvalCap': _NoCaptionMetrics}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['VQAEval']


def score(references, predictions, official_source):
    """Score exact matching VizWiz image IDs; caller must separately qualify split and roles."""
    if not references:
        raise ScoringError('A nonempty qualified reference set is required')
    if any(type(row) is not dict for row in references) or any(type(row) is not dict for row in predictions):
        raise ScoringError('Reference and prediction records must be objects')
    try:
        images = [row['image'] for row in references]
        predicted_images = [row['image'] for row in predictions]
    except KeyError as error:
        raise ScoringError('Every reference and prediction needs an image ID') from error
    if (any(type(image) is not str or not image for image in images + predicted_images)
            or len(set(images)) != len(images)
            or len(set(predicted_images)) != len(predicted_images)
            or set(images) != set(predicted_images)):
        raise ScoringError('Missing, extra, duplicate or invalid image IDs')
    for row in references:
        answers = row.get('answers')
        if (type(answers) is not list or len(answers) != 10
                or any(type(answer) is not dict or type(answer.get('answer')) is not str for answer in answers)
                or type(row.get('answer_type')) is not str):
            raise ScoringError('Exactly ten ordered answer strings and an answer type are required')
    if any(type(row.get('answer')) is not str for row in predictions):
        raise ScoringError('Every prediction must contain an answer string, including empty outputs')
    evaluator = _official_class(official_source)(_Input(references), _Input(predictions), n=2)
    with redirect_stdout(StringIO()):
        evaluator.evaluate(images)
    return {
        'policy': 'vizwiz-official-api-vqa-accuracy-class-only-v1',
        'official_source_sha256': OFFICIAL_SOURCE_SHA256,
        'score_percent': evaluator.accuracy['overall'],
        'per_answer_type_percent': evaluator.accuracy['perAnswerType'],
        'per_image_percent': evaluator.evalQA,
        'question_denominator': len(images),
        'caption_metrics_computed': False,
        'unanswerability_metrics_computed': False,
        'split_role_model_and_runtime_qualified': False,
    }
