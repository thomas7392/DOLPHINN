# Thomas Goldman 2023
# DOLPHINN

import numpy as np
import tensorflow as tf

from .function import Function
from .callbacks import SaveBest

class Scheduler(Function):

    def __init__(self,
                 schedule,
                 loss_weigths=None,
                 beta_1 = 0.9,
                 beta_2 = 0.999):

        self.name = "Scheduler"
        self.schedule = schedule
        self.loss_weigths = loss_weigths
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        super().__init__({})

    def call(self,
             DOLPHINN,
             additional_callbacks = []):

        for (lr, iterations) in self.schedule:

            DOLPHINN.current_lr = lr

            optimisation_alg = tf.keras.optimizers.Adam(learning_rate=lr,
                                               beta_1 = self.beta_1,
                                               beta_2 = self.beta_2)

            DOLPHINN.model.compile(optimisation_alg,
                        lr = lr,
                        metrics = DOLPHINN.metrics,
                        loss_weights = self.loss_weigths)

            _, _ = DOLPHINN.model.train(iterations=iterations,
                                        callbacks = [*DOLPHINN.callbacks, *additional_callbacks],
                                        display_every = DOLPHINN.display_every)


class DoubleRestarter(Function):

    def __init__(self,
                 schedule1,
                 schedule2,
                 loss_threshold = 10,
                 loss_threshold2 = 0.2,
                 max_attempts = 50,
                 loss_weigths = None,
                 attempts1 = None,
                 attempts2 = None):

        self.name = "DoubleRestarter"
        self.schedule1 = schedule1
        self.schedule2 = schedule2
        self.loss_threshold = loss_threshold
        self.loss_threshold2 = loss_threshold2
        self.max_attempts = max_attempts
        self.loss_weigths = loss_weigths
        self.attempts1 = attempts1
        self.attempts2 = attempts2

        super().__init__({})


    def call(self,
             DOLPHINN,
             additional_callbacks = []):

        temp_final_test_loss2 = np.array([np.NAN])
        temp_final_train_loss2 = np.array([np.NAN])

        attempt2 = 1
        while np.any(np.isnan(temp_final_train_loss2)) or np.abs(np.log10(np.sum(temp_final_train_loss2)) - np.log10(np.sum(temp_final_test_loss2))) > self.loss_threshold2:

            attempt1 = 1
            temp_final_test_loss1 = np.array([np.NAN])
            # Keep initalizing network untill condition is met
            while np.any(np.isnan(temp_final_test_loss1)) or np.sum(temp_final_test_loss1) > self.loss_threshold:

                if DOLPHINN.base_verbose:
                    print(f"[RESTARTER] Initialisation attempt: {attempt1}")

                # Initialize new network if not the first time
                if attempt1==1 and attempt2==1:
                    pass
                else:
                    DOLPHINN._create_model(verbose = DOLPHINN.base_verbose)

                # Perform schedule
                training_schedule = Scheduler(self.schedule1, self.loss_weigths)
                training_schedule.call(DOLPHINN, additional_callbacks=additional_callbacks)

                attempt1 += 1
                # Break if max attempts reached
                if attempt1 == self.max_attempts+1:
                    if DOLPHINN.base_verbose:
                        print("[RESTARTER] Maximum amount of attempts reached, stopping the restarter")
                    break

                # Prepare next attempt
                temp_final_test_loss1 = DOLPHINN.model.losshistory.loss_test[-1]


            self.attempts1 = attempt1 - 1

            # Perform schedule
            training_schedule = Scheduler(self.schedule2, self.loss_weigths)
            training_schedule.call(DOLPHINN, additional_callbacks=additional_callbacks)

            attempt2 += 1
            if attempt2 == 4:
                break

            # Prepare next attempt
            temp_final_test_loss2 = DOLPHINN.model.losshistory.loss_test[-1]
            temp_final_train_loss2 = DOLPHINN.model.losshistory.loss_train[-1]

            if DOLPHINN.base_verbose:
                print("[DOLPHINN] Test-Train Difference (log) = ", np.abs(np.log10(np.sum(temp_final_train_loss2)) - np.log10(np.sum(temp_final_test_loss2))))
        self.attempts2 = attempt2 - 1


class Restarter(Function):

    def __init__(self,
                 schedule,
                 loss_threshold = 10,
                 max_attempts = 50,
                 loss_weigths = None,
                 attempts = None):

        self.name = "Restarter"
        self.schedule = schedule
        self.loss_threshold = loss_threshold
        self.max_attempts = max_attempts
        self.loss_weigths = loss_weigths
        self.attempts = attempts

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


        self.attempts = attempt - 1

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


class LBFGS(Function):

    def __init__(self,
                 schedule,
                 loss_weigths = None):

        self.name = "LBFGS"
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