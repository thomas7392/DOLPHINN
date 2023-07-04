## Thomas Goldman 2023
# DOLPHINN

import deepxde as dde
import datetime
import numpy as np
import os
import shutil
import pickle


class Callback(dde.callbacks.Callback):

    def __init__(self):
        super().__init__()

    def add_dolphinn(self, DOLPHINN):
        self.DOLPHINN = DOLPHINN

class SaveBest(Callback):

    def __init__(self, iterations,
                       base_path = None,
                       monitor="train loss",
                       test_period = 1000,
                       verbose = 0):

        super().__init__()

        self.test_period = test_period
        self.monitor = monitor
        self.base_path = base_path
        self.best_value = np.inf
        self.epoch = 0
        self.iterations = iterations
        self.verbose = verbose

        # If no weigth_path provided, store inside cache folder
        if not base_path:
            current_time = datetime.datetime.now().strftime("case_%Y_%m_%d_%H_%M_%S")
            self.base_path = "../cache/" + current_time + "/"

        self.train_path = self.base_path + "train.dat"
        self.test_path = self.base_path + "test.dat"

        # Create directory
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        else:
            raise Exception(f"Path already exists: {self.base_path}, choose non-existing directory")

    def on_epoch_end(self):

        self.epoch += 1
        current_value = self.get_monitor_value()

        if self.epoch % self.test_period == 0 or self.epoch + 1 == self.iterations:
            if current_value < self.best_value:

                # Delete previous best
                self.best_value = current_value
                self.delete_directory_contents()

                # Save new best
                self.weigths_path = self.model.save(self.base_path)

                if self.verbose > 0:
                    print(f"[DOLPHINN] [Epoch {self.epoch}] New best stored at: {self.weigths_path}")

                #dde.utils.save_best_state(self.model.train_state, self.train_path, self.test_path, verbose = False)
                #dde.utils.save_loss_history(self.model.losshistory, loss_fname, verbose = False)

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        elif self.monitor == "position metric":
            result = self.model.train_state.metrics_test[0]
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result

    def delete_directory_contents(self):

        #Iterate over folder content and delete everything
        for filename in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")



class StoreAnimationData(Callback):

    def __init__(self,
                 path=None,
                 period=100,
                 verbose=1):

        super().__init__()

        if not path:
            current_time = datetime.datetime.now().strftime("case_%Y_%m_%d_%H_%M_%S")
            self.base_path = "animation_data/" + current_time + "/"

        else:
            if path[-1] != "/":
                raise Exception("Final character of path should be '/'")
            if os.path.exists(path):
                raise Exception(f"Path already exists: {path}")
            self.base_path = path

        os.makedirs(self.base_path)

        self.epoch = 0
        self.period = period
        self.verbose = verbose

        # Inputs/Outputs data
        self.X_train = {}
        self.X_test = None
        self.y_pred_test = {}
        self.y_pred_train = {}

        # Loss/Metrics data
        self.loss_train = {}
        self.loss_test = {}
        self.metrics = {}
        self.lr = {}

    def on_train_begin(self):
        self.X_test = self.model.train_state.X_test

    def on_epoch_end(self):
        '''
        Store the test runs evever {period} epochs
        '''

        self.epoch += 1

        if self.epoch % self.period == 0 or self.epoch == 1:

            # Include training data and loss
            self.X_train[self.epoch] = self.model.train_state.X_train

            # Include testing data and loss
            self.y_pred_train[self.epoch], self.loss_train[self.epoch] = self.model._outputs_losses(True,
                                                                      self.model.train_state.X_train,
                                                                      self.model.train_state.y_train,
                                                                      self.model.train_state.test_aux_vars)

            # Include testing data and loss
            self.y_pred_test[self.epoch], self.loss_test[self.epoch] = self.model._outputs_losses(False,
                                                                      self.model.train_state.X_test,
                                                                      self.model.train_state.y_test,
                                                                      self.model.train_state.test_aux_vars)

            self.metrics[self.epoch] = [m(self.model.train_state) for m in self.model.metrics]

            self.lr[self.epoch] = self.DOLPHINN.current_lr

    def on_train_end(self):
        '''
        Save the data
        '''

        data = {"x_train": self.X_train,
                "x_test": self.X_test,
                "y_pred_test": self.y_pred_test,
                "y_pred_train": self.y_pred_train,
                "loss_train": self.loss_train,
                "loss_test": self.loss_test,
                "loss_metrics": self.metrics,
                "lr": self.lr}

        with open(self.base_path + 'data.pickle', 'wb') as handle:
            pickle.dump(data, handle)

        if self.verbose > 0:
            print(f"[DOLPHINN] Saved animation data at {self.base_path + 'data.pickle'}")
