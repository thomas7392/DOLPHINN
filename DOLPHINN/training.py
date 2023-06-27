# Thomas Goldman 2023
# DOLPHINN

import numpy as np

from .function import Function

class Scheduler(Function):

    def __init__(self,
                 schedule,
                 loss_weigths=None):

        self.name = "Scheduler"
        self.schedule = schedule
        self.loss_weigths = loss_weigths

        super().__init__({})

    def call(self, DOLPHINN):

        for (lr, iterations) in self.schedule:
            DOLPHINN.model.compile("adam",
                        lr = lr,
                        metrics = DOLPHINN.metrics,
                        loss_weights = self.loss_weigths)

            _, _ = DOLPHINN.model.train(iterations=iterations,
                               callbacks = DOLPHINN.callbacks)


class Restarter(Function):

    def __init__(self,
                 loss_threshold = 10,
                 max_attempts = 50,
                 loss_weigths = None,
                 lr = 1e-2,
                 iterations = 3000):

        self.name = "Restarter"
        self.loss_threshold = loss_threshold
        self.max_attempts = max_attempts
        self.loss_weigths = loss_weigths
        self.lr = lr
        self.iterations = iterations
        super().__init__({})


    def call(self, DOLPHINN):
        attempt = 1
        loss_threshold = 10
        temp_final_test_loss = np.array([np.NAN])

        while np.any(np.isnan(temp_final_test_loss)) or np.sum(temp_final_test_loss) > loss_threshold:

            if DOLPHINN.base_verbose:
                print(f"Initialisation attempt: {attempt}")

            DOLPHINN._create_model(verbose = DOLPHINN.base_verbose)
            DOLPHINN.model.compile("adam", lr=self.lr, loss_weights = self.loss_weigths)
            _, _ = DOLPHINN.model.train(iterations=self.iterations)

            if attempt == self.max_attempts:
                if DOLPHINN.base_verbose:
                    print("Maximum amount of attempts reached, stopping the restarter")
                break

            attempt += 1
            temp_final_test_loss = DOLPHINN.model.losshistory.loss_test[-1]



