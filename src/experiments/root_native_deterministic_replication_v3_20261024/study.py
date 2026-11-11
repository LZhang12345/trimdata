"""Adapter for the unchanged strict readout; all real configs remain validated."""
from .protocol import native_identity
from . import protocol as p
def validate_study(study):return p.validate_study(study,gpu=True)
