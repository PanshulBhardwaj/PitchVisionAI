"""Domain-specific exceptions with actionable messages."""


class PitchVisionError(Exception):
    """Base class for all application errors."""


class VideoError(PitchVisionError):
    """Video cannot be opened, is empty, corrupted, or unsupported."""


class ConfigError(PitchVisionError):
    """Invalid configuration value."""


class DetectorError(PitchVisionError):
    """Model failed to load or run."""


class DatasetValidationError(PitchVisionError):
    """Tracking dataset failed schema or sanity validation."""


class OutputWriteError(PitchVisionError):
    """Video writer or output file could not be created."""
