# -*- coding: utf-8 -*-
"""
DeNA提出用
Cycle GAN

photo-現実の写真
monet-画家の書いた絵

photoをmonet風に画風変換

"""

import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_addons as tfa
import keras
from keras.layers import Input, Conv2D, add, Conv2DTranspose, Activation, LeakyReLU
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
import numpy as np
import glob
import cv2
#from IPython.display import clear_output


input_size = (256,256,3)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

#===========================================================
#setup img

files_monet = glob.glob("C:/Users/user01/Desktop/kaggle/monet_jpg/*")
monet = np.zeros((len(files_monet),256,256,3))
i = 0
for f in files_monet:
  monet[i] = cv2.imread(f)
  i+=1
  

#まずは練習用でデータ数をmonetに合わせる
files_photo = glob.glob("C:/Users/user01/Desktop/kaggle/photo_jpg/*")
#img = np.zeros((len(files),256,256,3))
photo = np.zeros((300,256,256,3))
i = 0
for f in files_photo:
  photo[i] = cv2.imread(f)
  i+=1
  if i ==300:
      break;



def normalizing(image):
    #image = tf.cast(image, tf.float32)
    #image = (image / 127.5) - 1
    image = image/255
    return image


monet = normalizing(monet)
photo = normalizing(photo)

train_monet = monet[:250]
train_photo = photo[:250]
test_monet = monet[250:]
test_photo = photo[250:]


# plt.subplot(121)
# plt.title('monet')
# plt.imshow(train_monet[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('photo')
# plt.imshow(train_photo[0] * 0.5 + 0.5)





#================================================================
#build gan model






def downsample(x, filters, activation, kernel_initializer=kernel_init,
               kernel_size=(3,3), strides=(2,2), padding="same",
               gamma_initializer=gamma_init, use_bias=False,):
    
    x = Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer,
               padding=padding, use_bias=use_bias,)(x)
    #x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def get_u_net(img_size):
    num_classes = 3
    inputs = Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="tanh", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model




def get_discriminator(filters=64, kernel_initializer=kernel_init, num_downsampling=3):
    
    inputs = Input(shape=input_size)
    x = Conv2D(filters, (4,4), strides=(2,2), padding="same", kernel_initializer=kernel_init)(inputs)
    x = LeakyReLU(0.2)(x)
    
    for i in range(num_downsampling):
        filters *= 2
        if i < 2:
            x = downsample(x, filters=filters, activation=LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=filters, activation=LeakyReLU(0.2), kernel_size=(4, 4), strides=(1,1))
    
    x = Conv2D(1, (3,3), strides=(1,1), padding="valid", kernel_initializer=kernel_init)(x)
    x = layers.Activation("sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model



#======================
gene_monet = get_u_net(input_size)
gene_photo = get_u_net(input_size)
disc_monet = get_discriminator()
disc_photo = get_discriminator()
#=======================






#=======================================================
#compile

optimizer = Adam(2e-4, beta_1=0.5)
optimizer1 = Adam(5e-4, beta_1=0.5)


z_monet = Input(shape=input_size)
z_photo = Input(shape=input_size)

fake_photo = gene_photo(z_monet)
cycle_monet = gene_monet(fake_photo)

fake_monet = gene_monet(z_photo)
cycle_photo = gene_photo(fake_monet)

same_monet = gene_monet(z_monet)
same_photo = gene_photo(z_photo)

disc_monet_real = disc_monet(z_monet)
disc_photo_real = disc_photo(z_photo)

# disc_monet_fake = disc_monet(fake_monet)
# disc_photo_fake = disc_photo(fake_photo)

    

#gene_photo->gene_monet (cycle)
cycle1 = Model(z_monet, cycle_monet)
cycle1.compile(loss="binary_crossentropy", optimizer=optimizer1)
#gene_monet->gene_photo (cycle)
cycle2 = Model(z_photo, cycle_photo)
cycle2.compile(loss='binary_crossentropy', optimizer=optimizer1)

#gene_photo->gene_monet (same)
same1 = Model(z_monet, same_monet)
same1.compile(loss='binary_crossentropy', optimizer=optimizer1)
#gene_monet->gene_photo (same)
same2 = Model(z_photo, same_photo)
same2.compile(loss='binary_crossentropy', optimizer=optimizer)

disc_monet.compile(loss='binary_crossentropy', optimizer=optimizer)
disc_photo.compile(loss='binary_crossentropy', optimizer=optimizer)
disc_monet.trainable = False
disc_photo.trainable = False

disc_monet_fake = disc_monet(fake_monet)
disc_photo_fake = disc_photo(fake_photo)


#gene_photo->disc_photo (gene_loss)
gene_loss1 = Model(z_monet, disc_photo_fake)
gene_loss1.compile(loss='binary_crossentropy', optimizer=optimizer)
#gene_monet->disc_monet (gene_loss)
gene_loss2 = Model(z_photo, disc_monet_fake)
gene_loss2.compile(loss='binary_crossentropy', optimizer=optimizer)

# #real_photo->disc_photo (disc_real)
# disc_real1 = Model(z_photo, disc_photo_real)
# disc_real1.compile(loss='binary_crossentropy', optimizer=optimizer)
# #real_monet->disc_monet (disc_real)
# disc_real2 = Model(z_monet, disc_monet_real)
# disc_real2.compile(loss='binary_crossentropy', optimizer=optimizer)

# #gene_photo->disc_photo (disc_fake)
# disc_fake1 = Model(z_monet, disc_photo_fake)
# disc_fake1.compile(loss='binary_crossentropy', optimizer=optimizer)
# #gene_monet->disc_monet (disc_fake)
# disc_fake2 = Model(z_photo, disc_monet_fake)
# disc_fake2.compile(loss='binary_crossentropy', optimizer=optimizer)








#========================================================================
#train

EPOCH = 100
step = 0
i = 0
if files_monet > files_photo:
    step = len(files_photo)
else:
    step = len(files_monet)

#test用があるので今回は
step = 250

for epoch in range(EPOCH):
    print("epoch:",epoch+1)
    
    start = time.time()
    n = 0
    
    g_monet_loss = 0
    g_photo_loss = 0
    d_monet_loss = 0
    d_photo_loss = 0
    
    for i in range(step):
        
        target_monet = train_monet[i].reshape(-1,256,256,3)
        target_photo = train_photo[i].reshape(-1,256,256,3)
        ones = np.ones((1,30,30,1))
        zero = np.zeros((1,30,30,1))
        
        #train_on_batch
        fake_photo = gene_photo.predict(target_monet)
        fake_monet = gene_monet.predict(target_photo)
        c1 = cycle1.train_on_batch(target_monet, target_monet)
        c2 = cycle2.train_on_batch(target_photo, target_photo)
        s1 = same1.train_on_batch(target_monet, target_monet)
        s2 = same2.train_on_batch(target_photo, target_photo)
        dr1 = disc_photo.train_on_batch(target_photo, ones)
        dr2 = disc_monet.train_on_batch(target_monet, ones)
        df1 = disc_photo.train_on_batch(fake_photo, zero)
        df2 = disc_monet.train_on_batch(fake_monet, zero)
        g1 = gene_loss1.train_on_batch(target_monet, ones)
        g2 = gene_loss2.train_on_batch(target_photo, ones)
        
        g_monet_loss += g2
        g_photo_loss += g1
        d_monet_loss += (dr2 + df1)/2
        d_photo_loss += (dr1 + df2)/2
        
        
        if n % 10 == 0:
            print(".", end="")
        n += 1
    
    
    #choice random number
    rand = random.randrange(step)
    
    #predict monet
    target_photo = train_photo[rand].reshape(-1,256,256,3)
    predict = gene_monet.predict(target_photo)
    predict = predict.reshape(256,256,3)
    predict = (predict * 255)
    t_photo = (train_photo[rand] * 255)
    # predict = (predict + 1) * 127.5
    # t_photo = (train_photo[rand] + 1) * 127.5
    cv2.imwrite("train/train{}_a_gene_.jpg".format(epoch), predict)
    cv2.imwrite("train/train{}_a_origin.jpg".format(epoch), t_photo)
    
    #predict photo
    target_monet = train_monet[rand].reshape(-1,256,256,3)
    predict = gene_photo.predict(target_monet)
    predict = predict.reshape(256,256,3)
    predict = (predict * 255)
    t_monet = (train_monet[rand] * 255)
    # predict = (predict + 1) * 127.5
    # t_monet = (train_monet[rand] + 1) * 127.5
    cv2.imwrite("train/train{}_b_gene.jpg".format(epoch), predict)
    cv2.imwrite("train/train{}_b_origin.jpg".format(epoch), t_monet)
    
    g_monet_loss /= step
    g_photo_loss /= step
    d_monet_loss /= step
    d_photo_loss /= step
    step_time = time.time() - start
    print('Time taken for epoch {} is %f sec'.format(epoch+1) % step_time)
    print("gene_monet_loss:", g_monet_loss)
    print("gene_photo_loss:", g_photo_loss)
    print("disc_monet_loss:", d_monet_loss)
    print("disc_photo_loss:", d_photo_loss)
    print("")

gene_monet.save('C:/Users/user01/Desktop/AI/cnn/gene_monet_model.h5')


#============================================================
#test

test = 50
for i in range(50):
    testphoto = test_photo[i].reshape(-1,256,256,3)
    predict = gene_monet.predict([testphoto])
    predict = predict.reshape(256,256,3)
    #predict = (predict + 1) * 127.5
    predict = predict * 255
    cv2.imwrite("test/test{}.jpg".format(i), predict)