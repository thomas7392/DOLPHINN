# Thomas Goldman 2023
# DOLPHINN

import numpy as np

from .function import Function
from .callbacks import SaveBest

class Scheduler(Function):

    def __init__(self,
                 schedule,
                 loss_weigths=None):

        self.name = "Scheduler"
        self.schedule = schedule
        self.loss_weigths = loss_weigths

        super().__init__({})

    def call(self,
             DOLPHINN,
             additional_callbacks = []):

        for (lr, iterations) in self.schedule:

            DOLPHINN.current_lr = lr
            DOLPHINN.model.compile("adam",
                        lr = lr,
                        metrics = DOLPHINN.metrics,
                        loss_weights = self.loss_weigths)

            _, _ = DOLPHINN.model.train(iterations=iterations,
                                        callbacks = [*DOLPHINN.callbacks, *additional_callbacks],
                                        display_every = DOLPHINN.display_every)


class Restarter(Function):

    def __init__(self,
                 schedule,
                 loss_threshold = 10,
                 max_attempts = 50,
                 loss_weigths = None):

        self.name = "Restarter"
        self.schedule = schedule
        self.loss_threshold = loss_threshold
        self.max_attempts = max_attempts
        self.loss_weigths = loss_weigths

        super().__init__({})


    def call(self,
             DOLPHINN,
             additional_callbacks = []):

        attempt = 1
        temp_final_test_loss = np.array([np.NAN])

        # Keep initalizing network untill condition is met
        while np.any(np.isnan(temp_final_test_loss)) or np.sum(temp_final_test_loss) > self.loss_threshold:

            if DOLPHINN.base_verbose:
                print(f"[RESTARTER] Initialisation attempt: {attempt}")

            # Initialize new network if not the first time
            if not attempt==1:
                DOLPHINN._create_model(verbose = DOLPHINN.base_verbose)

            # Perform schedule
            training_schedule = Scheduler(self.schedule, self.loss_weigths)
            training_schedule.call(DOLPHINN, additional_callbacks=additional_callbacks)

            # Break if max attempts reached
            if attempt == self.max_attempts:
                if DOLPHINN.base_verbose:
                    print("[RESTARTER] Maximum amount of attempts reached, stopping the restarter")
                break

            # Prepare next attempt
            attempt += 1
            temp_final_test_loss = DOLPHINN.model.losshistory.loss_test[-1]



class Restorer(Function):

    def __init__(self,
                 schedule,
                 loss_weigths = None):

        self.name = "Restorer"
        self.schedule = schedule
        self.loss_weigths = loss_weigths
        super().__init__({})


    def call(self,
             DOLPHINN,
             additional_callbacks = []):

        # Create savebest callback
        total_iterations = np.sum(np.array([x for _, x in self.schedule]))
        savebest_callback = SaveBest(total_iterations, verbose = True)

        # Call the scheduler
        training_schedule = Scheduler(self.schedule, self.loss_weigths)
        training_schedule.call(DOLPHINN, additional_callbacks=[savebest_callback, *additional_callbacks])

        # Restore best
        DOLPHINN.restore(savebest_callback.weigths_path)

        if DOLPHINN.base_verbose:
            print(f"[DOLPHINN] Restored model from: {savebest_callback.weigths_path}")

        savebest_callback.delete_directory_contents()

        print("[DOLPHINN] Finished Restorer training algorithm: best ")


