import numpy as np


class Adam():
    """
    Optimize objective function using Adam

    This is an implementation of the Adam algorithm:
        Kingma, D. P., & Ba, J. L. (2015)
        Adam: a Method for Stochastic Optimization. International Conference on Learning Representations

    Adaptive Moment Estimation (Adam) computes adaptive learning rates for each parameter.
    In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and RMSprop,
    Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum

    """

    def __init__(self, maxiter=100, learning_rate=1e-3, momentum_estimate1=0.9, momentum_estimate2=0.999, noise=1e-7):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.momentum_estimate1 = momentum_estimate1
        self.momentum_estimate2 = momentum_estimate2
        self.noise = noise

    def __repr__(self):
        return "Adam stochastic optimization (learning_rate={0} momentum_estimate1={1} momentum_estimate2={2} noise={3} maxiter={4})".format(
            self.learning_rate, self.momentum_estimate1, self.momentum_estimate2, self.noise, self.maxiter)

    def minimize(self, objfun, x):

        #initialize the moment vectors
        first_moment = np.zeros(objfun.nvar)
        second_moment = np.zeros(objfun.nvar)

        objfun.begin_progress()

        fx, g = objfun.evaluate(x)
        objfun.progress(x, g, fx, 0, 1, 0)

        for i in range(self.maxiter):
            #update moment vectors
            first_moment    = self.momentum_estimate1 * first_moment  +  (1-self.momentum_estimate1)  * (g)
            second_moment   = self.momentum_estimate2 * second_moment +  (1-self.momentum_estimate2)  * (g*g)

            #compute bias corrected moments
            first_moment_corrected  = first_moment  / (1 - np.power(self.momentum_estimate1, i+1))
            second_moment_corrected = second_moment / (1 - np.power(self.momentum_estimate2, i+1))

            #apply update
            x -= self.learning_rate * first_moment_corrected / np.sqrt(second_moment_corrected + self.noise)

            fx, g = objfun.evaluate(x)
            objfun.progress(x, g, fx, i + 1, 1, self.learning_rate)


        ret = {
            "code": 2,
            "message": "Reached number of iterations"
        }

        return fx, x, ret
