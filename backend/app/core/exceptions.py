"""Custom exception hierarchy for Showtime."""


class ShowtimeError(Exception):
    """Base exception for all Showtime errors."""


class SceneDetectionError(ShowtimeError):
    """Error during scene detection."""


class CaptionError(ShowtimeError):
    """Error during frame captioning."""


class TranscriptionError(ShowtimeError):
    """Error during audio transcription."""


class MappingError(ShowtimeError):
    """Error during AI mapping."""


class TimelineError(ShowtimeError):
    """Error during timeline assembly."""


class RenderError(ShowtimeError):
    """Error during video rendering."""
