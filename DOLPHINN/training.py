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
        temp_final_test_loss = np.array([np.NAN])

        # Keep initalizing network untill condition is met
        while np.any(np.isnan(temp_final_test_loss)) or np.sum(temp_final_test_loss) > self.loss_threshold:

            if DOLPHINN.base_verbose:
                print(f"[RESTARTER] Initialisation attempt: {attempt}")

            # Initialize new network if not the first time
            if not attempt==1:
                DOLPHINN._create_model(verbose = DOLPHINN.base_verbose)

            # Aggresive training
            DOLPHINN.model.compile("adam", lr=self.lr, loss_weights = self.loss_weigths)
            _, _ = DOLPHINN.model.train(iterations=self.iterations)

            # Break if max attempts reached
            if attempt == self.max_attempts:
                if DOLPHINN.base_verbose:
                    print("[RESTARTER] Maximum amount of attempts reached, stopping the restarter")
                break

            # Prepare next attempt
            attempt += 1
            temp_final_test_loss = DOLPHINN.model.losshistory.loss_test[-1]



