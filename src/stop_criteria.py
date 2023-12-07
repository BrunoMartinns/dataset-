from __future__ import annotations
from typing import List
import src.algorithms as al

import abc

class StopCriteria(abc.ABC):

    @abc.abstractmethod
    def isFinished(self, alg: al.Algorithm) -> bool:
        """Implement stop criterium"""

class CompositeStopCriteria(StopCriteria):

    children: List[StopCriteria]
    def __init__(self) -> None:
        super().__init__()
        self.children = [] 

    def add(self, stop_criteria: StopCriteria):
        self.children.append(stop_criteria)

    def isFinished(self, alg: al.Algorithm) -> bool:
        return any([s.isFinished(alg) for s in self.children])


class MaxIterationStopCriteria(StopCriteria):

    def __init__(self, max_iteration) -> None:
        super().__init__()
        self.max_iteration = max_iteration

    def isFinished(self, alg: al.Algorithm) -> bool:
        return alg.iteration >= self.max_iteration


class MinErrorStopCriteria(StopCriteria):

    def __init__(self, min_error) -> None:
        super().__init__()
        self.min_error = min_error

    def isFinished(self, alg: al.Algorithm) -> bool:
        return alg.error < self.min_error