import src.algorithms as al
import abc 
import matplotlib.pyplot as plt

class AlgorithmAnalyzer(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def notify_started(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_finished(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_iteration(self, alg: al.Algorithm):
        pass

class PlotterAlgorithmObserver(AlgorithmAnalyzer):

    def __init__(self) -> None:
        self.iterations = []
        self.errors = []
        self.weights = []
        super().__init__()  

    def notify_started(self, alg: al.Algorithm):
        pass

    def notify_finished(self, alg: al.Algorithm):
        pass

    def notify_iteration(self, alg: al.Algorithm):
        self.iterations.append(alg.iteration)
        self.errors.append(alg.error)
        self.weights.append(alg.model.w)
    
    def plot(self, **kwargs):
        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.errors, color='red')
        ax.set_ylabel("RMSE(t)")
        ax.set_xlabel("Iterations (t)")
        ax.hlines(0, xmin=min(self.iterations), xmax=max(self.iterations), linestyles="dashed", color="red", label="error", alpha=.6)
        ax2 = ax.twinx()
        if("weights" in kwargs):
            ws = kwargs["weights"]
            ax2.hlines(ws[0], xmin=min(self.iterations), xmax=max(self.iterations), linestyles="dashed", color="green", alpha=.6)
            ax2.hlines(ws[1], xmin=min(self.iterations), xmax=max(self.iterations), linestyles="dashed", color="blue", alpha=.6)
        ax2.plot(self.iterations, [w[0] for w in self.weights], color='green', label="w[0]", alpha=.6)
        ax2.plot(self.iterations, [w[1] for w in self.weights], color='blue', label="w[1]", alpha=.6) 
        ax2.set_ylabel("Weights w(t)")
        ax2.legend()
        ax.legend()
        plt.show()