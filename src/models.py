from __future__ import annotations
import numpy as np
import scipy.stats as st
import abc
import src.optimizers as opt


class Model(abc.ABC):

    def _init_(self) -> None:
        self._w = None
        super()._init_()
        
    
    @abc.abstractmethod
    def predict(self, X: np.array) -> np.ndarray:
        """Implement the predict method"""

    @abc.abstractmethod
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement gradient"""
        
    @abc.abstractmethod
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement hessian"""

    @abc.abstractmethod
    def error(self, X, y):
        """Implement error""" #  def __update_error(self, X, y):
        
        
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

class LinearModel(Model):

    def init(self) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w)

    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X)) * (np.linalg.multi_dot([X.T, X, self.w]) - np.dot(X.T,y))

    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X)) * np.dot(X.T, X)

    def error(self, X, y):
            yhat = self.predict(X)
            errors = yhat - y
            rmse = 1.0/len(X) * np.square(np.linalg.norm(errors))
            return rmse
    
class LogisticModel(Model):

    def _init_(self) -> None:
        super()._init_()

    def predict(self, X: np.array) -> np.ndarray:
        return self.sigmoid(X)
    
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        sigmoid = self.sigmoid(X)
        return X.T@(sigmoid - y.reshape(-1, 1))
    
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        sigmoid = self.sigmoid(X)
        sigmoid_diagonal = np.diag(sigmoid.flatten() * (1 - sigmoid.flatten()))
        return (X.T@sigmoid_diagonal)@X
    
    def error(self, X, y):
        p = self.predict(X)
        return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))

    def sigmoid(self, x):
        return  1 / (1 + np.exp(-x@self._w))