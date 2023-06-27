# Thomas Goldman 2023
# DOLPHINN

# DeepXDE Imports
import deepxde as dde
from deepxde.backend import tf
from deepxde import config, optimizers

class ObjectivePINN(dde.Model):
    '''
    Model that includes an objective as a regularization term
    '''

    def __init__(self, data, net):
        super().__init__(data, net)

    def _compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""

        @tf.function(jit_compile=config.xla_jit)
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses += [self.net.regularizer(inputs, outputs_, losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
            return outputs_, losses

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

        def train_step_tfp(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )