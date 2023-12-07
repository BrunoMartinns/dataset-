from __future__ import annotations
import numpy as np
import scipy.stats as st
import abc
import src.models as models

class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""

class NewtonsMethod(OptimizerStrategy):

    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)

    def update_model(self, X, y, model: models.Model):
        gradient = model.gradient(X, y)
        hessian = model.hessian(X, y)
        hessian += self.learning_rate * np.identity(hessian.shape[0])
        inversed_hessian = np.linalg.solve(hessian, np.identity(hessian.shape[0]))
        model.w = model.w - self.learning_rate * (inversed_hessian@gradient)

class SteepestDescentMethod(OptimizerStrategy):

    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)

    def update_model(self, X: np.array, y, model: models.Model):
        gradient = model.gradient(X, y)
        model.w = model.w - self.learning_rate * gradient
        