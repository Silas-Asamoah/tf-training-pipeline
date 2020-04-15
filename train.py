from tflow_dataset.model import Model
from tflow_dataset.tf_dataset import Dataset
from tflow_dataset.pair_generator import PairGenerator

import tensorflow as tf
import pylab as plt
import numpy as np



def main():
    generator = PairGenerator()
    iter = generator.get_next_pair()
    for i in range(2):
        print(next(iter))
        
    ds = Dataset(generator)
    model_input = ds.next_element
    model = Model(model_input)
    
    with tf.compat.v1.Session() as sess:
        (img1, img2, label) = sess.run([model_input.img1, model_input.img2, model_input.label])
        plt.subplot(2, 1, 1)
        plt.imshow(img1[0].astype(np.uint8))
        plt.subplot(2, 1, 1)
        plt.imshow(img2[0].astype(np.uint8))
        plt.title(f'label {label[0]}')
        plt.show()
        
        sess.run(tf.compat.v1.global_variables_initializer)
        
        for step in range(100):
            (_, current_loss) = sess.run([model.opt_step, model.loss])
            print(f"step {step} log loss {current_loss}")
            
if __name__ == '__main__':
    main()