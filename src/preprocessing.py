import numpy as np
import scipy.stats as st

class Preprocessing:
        
    @staticmethod
    def build_design_matrix(X):
        m,n = X.shape
        ones = np.ones((m, 1))
        return np.column_stack([ones, X]) 

    @staticmethod
    def initilize_weights(X):
        m,n = X.shape
        errors = st.norm.rvs(size=(n+1, 1))
        return np.zeros((n+1, 1)) 
