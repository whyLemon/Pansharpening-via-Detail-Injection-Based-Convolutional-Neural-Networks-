
"""
# This is a re-implementation of training code of this paper:
# Pansharpening via Detail Injection Based Convolutional Neural Networks 
"""

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import h5py
import scipy.io as sio
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_batch(train_data,bs): 
    
    gt = train_data['gt'][...]    #read your trainning data
    pan = train_data['pan'][...]
    ms_lr = train_data['ms'][...]
    lms   = train_data['lms'][...]
    
    gt = np.array(gt,dtype = np.float32) / 2047.  ### normalization
    pan = np.array(pan, dtype = np.float32) /2047.
    ms_lr = np.array(ms_lr, dtype = np.float32) / 2047.
    lms  = np.array(lms, dtype = np.float32) /2047.

    
    N = gt.shape[0]
    batch_index = np.random.randint(0,N,size = bs)
    
    gt_batch = gt[batch_index,:,:,:]
    pan_batch = pan[batch_index,:,:]
    ms_lr_batch = ms_lr[batch_index,:,:,:]
    lms_batch  = lms[batch_index,:,:,:]

    pan_batch = pan_batch[:,:,:,np.newaxis]
    
    return gt_batch, lms_batch, pan_batch, ms_lr_batch


def vis_ms(data):
    _,b,g,_,r,_,_,_ = tf.split(data,8,axis = 3)
    vis = tf.concat([r,g,b],axis = 3)
    return vis


########## PanNet structures ################
def PanNet(ms, pan, num_spectral = 8, num_res = 4, num_fm = 64, reuse=False):
    
    weight_decay = 1e-5

    with tf.variable_scope('net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()


        ms_1 = tf.concat([ms,pan],axis=3)

        rs = ly.conv2d(ms_1, num_outputs = num_fm, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),
                          activation_fn = tf.nn.relu)

        rs = ly.conv2d(rs, num_outputs = num_fm, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay),
                          weights_initializer = ly.variance_scaling_initializer(),
                          activation_fn = tf.nn.relu)

        rs = ly.conv2d(rs, num_outputs = num_spectral, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay),
                          weights_initializer = ly.variance_scaling_initializer(),
                          activation_fn = None)
            
        rs = tf.add(rs,ms)

        return rs

 ###########################################################################
 ###########################################################################
 ########### Main Function: input data from here! (likes sub-funs in matlab before) ######

if __name__ =='__main__':

    tf.reset_default_graph()   

    train_batch_size = 32 # training batch size
    validation_batch_size = 32  # validation batch size
    image_size = 64      # patch size
    iterations = 100100 # total number of iterations to use
    model_directory = './models' # directory to save trained model to
    train_data_name = './training_data/train.mat'  # training data
    validation_data_name  = './training_data/validation.mat'   # validation data
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD
    
############## loading data
    train_data = sio.loadmat(train_data_name)   # for small data (not v7.3 data)
    validation_data = sio.loadmat(validation_data_name)

############## placeholder for training ###########
    gt = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    lms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    pan = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,1])

############# placeholder for validationing ##############
    validation_gt = tf.placeholder(dtype = tf.float32,shape = [validation_batch_size,image_size,image_size,8])
    validation_lms = tf.placeholder(dtype = tf.float32,shape = [validation_batch_size,image_size,image_size,8])
    validation_pan = tf.placeholder(dtype = tf.float32,shape = [validation_batch_size,image_size,image_size,1])

######## network architecture (call: PanNet constructed before!) ######################调用网络架构
    mrs = PanNet(lms,pan)    # call pannet
    
    validation_rs = PanNet(validation_lms,validation_pan,reuse = True)

######## loss function ################
    mse = tf.reduce_mean(tf.square(mrs - gt))  # compute cost : loss
    validation_mse = tf.reduce_mean(tf.square(validation_rs - validation_gt))

##### Loss summary (for observation) ################
    mse_loss_sum = tf.summary.scalar("mse_loss",mse)

    validation_mse_sum = tf.summary.scalar("validation_loss",validation_mse)

    lms_sum = tf.summary.image("lms",tf.clip_by_value(vis_ms(lms),0,1))
    mrs_sum = tf.summary.image("rs",tf.clip_by_value(vis_ms(mrs),0,1))

    label_sum = tf.summary.image("label",tf.clip_by_value(vis_ms(gt),0,1))
    
    all_sum = tf.summary.merge([mse_loss_sum,mrs_sum,label_sum,lms_sum])

############ optimizer: Adam or SGD ##################
    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'net')    

    if method == 'Adam':#
        g_optim = tf.train.AdamOptimizer(0.001, beta1 = 0.9) \
                          .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(0.1,global_steps,decay_steps = 50000, decay_rate = 0.1)
        clip_value = 0.1/lr
        optim = tf.train.MomentumOptimizer(lr,0.9)
        gradient, var   = zip(*optim.compute_gradients(mse,var_list = t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient,clip_value)
        g_optim = optim.apply_gradients(zip(gradient,var),global_step = global_steps)
        
##### GPU setting，
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

###########################################################################
###########################################################################
#### Run the above (take real data into the Net, for training) ############

    init = tf.global_variables_initializer()  # initialization: must done!

    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(init)
 
        if restore:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess,ckpt.model_checkpoint_path)

        #### read training data #####
        gt1 = train_data['gt'][...]
        pan1 = train_data['pan'][...]
        lms1 = train_data['lms'][...]

        gt1 = np.array(gt1, dtype=np.float32) / 2047.  ### [0, 1] normalization, WorldView L = 11，
        pan1 = np.array(pan1, dtype=np.float32) / 2047.
        lms1 = np.array(lms1, dtype=np.float32) / 2047.

        N = gt1.shape[0]

        #### read validation data #####
        gt2 = validation_data['gt'][...]
        pan2 = validation_data['pan'][...]
        lms2 = validation_data['lms'][...]

        gt2 = np.array(gt2, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
        pan2 = np.array(pan2, dtype=np.float32) / 2047.
        lms2 = np.array(lms2, dtype=np.float32) / 2047.

        N2 = gt2.shape[0]

        mse_train = []#keep the loss
        mse_valid = []
        
        for i in range(iterations):
            ###################################################################
            #### training phase! ###########################

            bs = train_batch_size
            batch_index = np.random.randint(0, N, size=bs)

            train_gt = gt1[batch_index, :, :, :]
            pan_batch = pan1[batch_index, :, :]
            train_lms = lms1[batch_index, :, :, :]

            train_pan = pan_batch[:, :, :, np.newaxis]  # expand to N*H*W*1


            _,mse_loss,merged = sess.run([g_optim,mse,all_sum],feed_dict = {gt: train_gt, lms: train_lms,
                                         pan: train_pan})

            mse_train.append(mse_loss)   # record the mse of trainning，

            if i % 100 == 0:

                print ("Iter: " + str(i) + " MSE: " + str(mse_loss))

            if i % 5000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
                print ("Save Model")

            ###################################################################
            #### validation phase! ###########################

            bs_validation = validation_batch_size
            batch_index2 = np.random.randint(0, N, size=bs_validation)

            validation_gt_batch = gt2[batch_index2, :, :, :]

            validation_lms_batch = lms2[batch_index2, :, :, :]

            pan_batch = pan2[batch_index2, :, :]
            validation_pan_hp_batch = pan_batch[:, :, :, np.newaxis]

            validation_mse_loss,merged = sess.run([validation_mse,validation_mse_sum],
                                               feed_dict = {validation_gt : validation_gt_batch, validation_lms : validation_lms_batch,
                                                            validation_pan : validation_pan_hp_batch})

            mse_valid.append(validation_mse_loss)  # record the mse of trainning

            if i % 1000 == 0 and i != 0:
                print("Iter: " + str(i) + " Valid MSE: " + str(validation_mse_loss))  # print
                
        ## finally write the mse info ##
        file = open('train_mse.txt','w')  # write the training error into train_mse.txt
        file.write(str(mse_train))
        file.close()

        file = open('valid_mse.txt','w')  # write the valid error into valid_mse.txt
        file.write(str(mse_valid))
        file.close()





