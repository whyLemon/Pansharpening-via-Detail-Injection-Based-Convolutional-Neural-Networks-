
"""
# This is a re-implementation of training code of this paper:
# Pansharpening via Detail Injection Based Convolutional Neural Networks
"""

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def PanNet(ms, pan, num_spectral=8, num_res=4, num_fm=64, reuse=False):
    weight_decay = 1e-5

    with tf.variable_scope('net'):
        if reuse:  # reuse the PanNet or not
            tf.get_variable_scope().reuse_variables()

        ms_1 = tf.concat([ms, pan], axis=3)

        rs = ly.conv2d(ms_1, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(),
                       activation_fn=tf.nn.relu)

        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(),
                       activation_fn=tf.nn.relu)

        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(),
                       activation_fn=None)

        rs = tf.add(rs, ms)

        return rs


#################################################################
################# Main fucntion ##################################
if __name__=='__main__':

    test_data = 'new_data.mat'#read your own data

    model_directory = './models/'#keep your models

    tf.reset_default_graph()
    
    data = sio.loadmat(test_data)

    lms = data['lms'][...]
    lms = np.array(lms, dtype = np.float32) /2047.

    pan  = data['pan'][...]
    pan  = np.array(pan,dtype = np.float32) /2047.


    pan = pan[np.newaxis,:,:,np.newaxis]

    h = pan.shape[1] # height
    w = pan.shape[2] # width

    lms   = lms[np.newaxis,:,:,:]
    
##### placeholder for testing#######
    p_hp = tf.placeholder(shape=[1,h,w,1],dtype=tf.float32)
    lms_p = tf.placeholder(shape=[1,h,w,8],dtype=tf.float32)


    rs = PanNet(lms_p,p_hp) # output high-frequency parts
    
    output = tf.clip_by_value(rs,0,1) # final output，



################################################################
##################Session Run ##################################
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:  
        sess.run(init)
        
        # loading  model       
        if tf.train.get_checkpoint_state(model_directory):
           ckpt = tf.train.latest_checkpoint(model_directory)
           saver.restore(sess, ckpt)
           print ("load new model")

        else:
           ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
           saver.restore(sess,ckpt.model_checkpoint_path)
           print ("load pre-trained model")                            
        


        final_output = sess.run(output,feed_dict = {p_hp:pan, lms_p:lms})

        sio.savemat('./result/output.mat', {'output':final_output[0,:,:,:]})#keep your output
