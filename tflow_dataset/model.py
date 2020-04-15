import tensorflow as tf

from tensorflow import Tensor


#Model inputs
class Inputs(object):
    def __init__(self, img1: Tensor, img2 : Tensor, label: Tensor):
        self.img1 = img1
        self.img2 = img2
        self.label = label
        
class Model(object):
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
    def predict(self, inputs: Inputs):
        with tf.name_scope("image subtraction"):
            img_diff = (inputs.img1 - inputs.img2)
            x = img_diff
        
        with tf.name_scope("conv_relu_maxpool"):
            for conv_layer_i in range(5):
                x = tf.keras.layers.Conv2D(x,
                                           filters= 20 * (conv_layer_i + 1),
                                           kernel_size = 3,
                                           activation = tf.nn.relu)
                x = tf.keras.layers.MaxPool2D(x, pool_size=3, strides=2)
                
        with tf.name_scope('fully_connected'):
            for conv_layer_i in range(1):
                x = tf.keras.layers.Dense(x, units = 200, activations=tf.nn.relu)
        
        with tf.name_scope('linear_predict'):
            predicted_logits = tf.keras.layers.Dense(x, 1, activation=None)
            
        return tf.squeeze(predicted_logits)
            
    def calculate_loss(self, inputs:Inputs, prediction_logits: Tensor):
        with tf.name_scope('calculate_loss'):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs.label, logits=prediction_logits))
        
        