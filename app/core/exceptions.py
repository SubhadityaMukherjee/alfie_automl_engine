class AutoMLError(Exception):
    """Base exception for all AutoML pipeline errors."""


class AutoMLValidationError(AutoMLError, ValueError):
    """Raised when caller-supplied inputs are invalid."""


class AutoMLDataError(AutoMLValidationError):
    """Raised for DataFrame-level problems."""


class AutoMLConfigError(AutoMLValidationError):
    """Raised when a configuration value or constant is out of range."""


class AutoMLRuntimeError(AutoMLError, RuntimeError):
    """Raised when a pipeline step fails at runtime."""


class AutoMLTrainingError(AutoMLRuntimeError):
    """Raised when AutoGluon's .fit() call fails."""


class AutoMLDeploymentError(AutoMLRuntimeError):
    """Raised when cloning or loading the deployment predictor fails."""


class AutoMLLeaderboardError(AutoMLRuntimeError):
    """Raised when leaderboard generation fails."""
