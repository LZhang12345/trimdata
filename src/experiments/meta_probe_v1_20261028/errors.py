class ProbeError(ValueError):
    """A scientific identity or runtime contract was violated."""


class NotReadyError(ProbeError):
    """A required real provider or qualification is unavailable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)
