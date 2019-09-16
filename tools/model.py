import tensorflow as tf
from tools.model_component import Layers, ModelComponent


class Model(Layers, ModelComponent):
    def unet(self, x, num_layers, feature_, filter, pool, loss, optimizer):
        self.model_name = 'U-Net'

        # contracting_path
        for layer in range(num_layers):
            with tf.name_scope(f'contracting_path_{layer}'):