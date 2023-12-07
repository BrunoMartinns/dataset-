from pkgutil import get_data
import numpy as np
import scipy.stats as st
from typing import Tuple
import abc


class DataGenerator(abc.ABC):
    
    @abc.abstractmethod
    def get_data(self):
        """Implement Get Data"""

class DataGeneratorLinearModel(DataGenerator):

    def __init__(self, N, w, x_min, x_max) -> None:
        super().__init__()
        self.N = N
        self.w = w
        self.x_min = x_min
        self.x_max = x_max
 
    def get_data(self):
        x0 = np.ones((self.N, 1))
        x1 = np.array([np.linspace(self.x_min,self.x_max,self.N)]).T
        X = np.column_stack([x0,x1])
        y = X.dot(self.w)
        e = st.norm.rvs(size=self.N, scale=1)
        e.shape = (self.N,1)
        y = y + e
        return X, y

class DataGeneratorLogisticModel(DataGenerator):

    def get_data(self):
        
        return super().get_data()