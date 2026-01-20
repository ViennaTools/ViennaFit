"""Custom exceptions for ViennaFit."""


class EarlyStoppingException(Exception):
    """Raised when early stopping criterion is met."""

    def __init__(self, evaluation_number: int, best_score: float):
        self.evaluation_number = evaluation_number
        self.best_score = best_score
        super().__init__(f"Early stopping at evaluation {evaluation_number}")
