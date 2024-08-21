import os

os.environ['DDE_BACKEND'] = 'tensorflow'
import deepxde as dde
from deepxde.backend import tf

class PINN():
    
    def input_transform(self, t: tf.Tensor):
        """
        Applies a transformation to the input tensor.

        Parameters
        ----------
        t : tf.Tensor
            The input tensor.

        Returns
        -------
        transformed_t : tf.Tensor
            The transformed tensor.

        """
        return tf.concat(
            (
                t,
                tf.sin(t),
                tf.sin(2 * t)
            ),
            axis=1,
        )


    def output_transform(self, t: tf.Tensor, y: tf.Tensor):
        """
        Transforms the output of the model.

        Parameters
        ----------
        t : tf.Tensor
            The input tensor.
        y : tf.Tensor
            The output tensor.

        Returns
        -------
        transformed_t : tf.Tensor
            The transformed output tensor.

        """
        return tf.concat([y * tf.tanh(t)], axis=1)
    
    
    def get_network(self, input_transform: bool = True, output_transform: bool = True) -> dde.nn.FNN:
        """
        Returns a fully connected neural network (FNN) model.

        Parameters
        ----------
        input_transform : bool, optional
            Whether to apply input transformation to the network. The default is True.
        output_transform : bool, optional
            Whether to apply output transformation to the network. The default is True.

        Returns
        -------
        net : dde.nn.FNN
            The fully connected neural network model.

        """
        layer_size = [1] + [32] * 2 + [1]
        activation = 'tanh'
        initializer = 'Glorot normal'
        net = dde.nn.FNN(layer_size, activation, initializer)
    
        if input_transform:
            net.apply_feature_transform(self.input_transform)
        if output_transform:
            net.apply_output_transform(self.output_transform)
    
        return net
    

class PairPINN():
    
    def input_transform(self, t: tf.Tensor):
        """
        Applies a transformation to the input tensor.

        Parameters
        ----------
        t : tf.Tensor
            The input tensor.

        Returns
        -------
        transformed_t : tf.Tensor
            The transformed tensor.

        """
        return tf.concat(
            (
                t,
                tf.sin(t),
                tf.sin(2 * t)
            ),
            axis=1,
        )


    def output_transform(self, t: tf.Tensor, y: tf.Tensor):
        """
        Transforms the output of the model.

        Parameters
        ----------
        t : tf.Tensor
            The input tensor.
        y : tf.Tensor
            The output tensor.

        Returns
        -------
        transformed_t : tf.Tensor
            The transformed output tensor.

        """
        return tf.concat([y * tf.tanh(t)], axis=1)
    
    
    def get_network(self, pair: int, input_transform: bool = True, output_transform: bool = True) -> dde.nn.PFNN:
        """
        Get the network model for the given pair.

        Parameters
        ----------
        pair : int
            The pair value.
        input_transform : bool, optional
            Whether to apply input transformation. The default is True.
        output_transform : bool, optional
            Whether to apply output transformation. The default is True.

        Returns
        -------
        net : dde.nn.PFNN
            The network model.

        """
        pair_size = pair + 1
        layer_size = [1] + [[32] * pair_size, [32] * pair_size] + [pair_size]
        activation = 'tanh'
        initializer = 'Glorot normal'
        net = dde.nn.PFNN(layer_size, activation, initializer)

        if input_transform:
            net.apply_feature_transform(self.input_transform)
        if output_transform:
            net.apply_output_transform(self.output_transform)

        return net