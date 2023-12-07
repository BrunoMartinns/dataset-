from __future__ import annotations
import numpy as np
import abc
import src.optimizers as opt
import src.models as models 
import src.stop_criteria as stop
from src.preprocessing import Preprocessing

class Algorithm(abc.ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        self.algorithm_observers = []
        self.optimizer_strategy = optimizer_strategy
        self.model = model
        
    def add(self, observer):
       if observer not in self.algorithm_observers:
           self.algorithm_observers.append(observer)
       else:
           print('Failed to add: {}'.format(observer))

    def remove(self, observer):
       try:
           self.algorithm_observers.remove(observer)
       except ValueError:
           print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit():
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value    

    @property
    def errors(self):
        return self._errors
    
    @errors.setter
    def errors(self, value):
        self._errors = value

    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, value):
        self._error = value
    
class PLA(Algorithm):

    def fit(self, X : np.ndarray, y: np.ndarray, stop_criteria: stop.StopCriteria):
        self.iteration = 0
        self.model.w = np.zeros((X.shape[1], 1)) 
        while True:
            self.error = self.model.error(X, y)
            self.notify_iteration()
            self.optimizer_strategy.update_model(X, y, self.model)
            self.iteration = self.iteration + 1
            if stop_criteria.isFinished(self):
                self.notify_finished()
                break