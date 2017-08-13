import numpy as np
import tensorflow as tf
import tflearn
import cv2
from sklearn.cross_validation import train_test_split
import os
from skimage import color, io
from scipy.misc import imresize
from glob import glob
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
with tf.device('/gpu:0'):
    tflearn.init_graph(2,gpu_memory_fraction=0.8)
    files_path1='D:\\fire_smoke_photos\\fire'
    files_path2='D:\\fire_smoke_photos\\smoke'
    files_path3='D:\\fire_smoke_photos\\other'
    fire_files_path=os.path.join(files_path1,'*.jpg')
    smoke_files_path=os.path.join(files_path2,'*.jpg')
    other_files_path=os.path.join(files_path3,'*.jpg')
    fire_files=sorted(glob(fire_files_path))
    smoke_files=sorted(glob(smoke_files_path))
    other_files=sorted(glob(other_files_path))
    n_files=len(fire_files)+len(smoke_files)+len(other_files)
    print(n_files)
    size_image=64
    allX=np.zeros((n_files,size_image,size_image,3),dtype='float64')
    allY=np.zeros(n_files)
    count=0
    for f in fire_files:
        try:
            img = cv2.imread(f,1)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            allY[count] = 0
            count += 1

        except:

            continue



    for f in smoke_files:

        try:

            img = cv2.imread(f,1)

            new_img = imresize(img, (size_image, size_image, 3))

            allX[count] = np.array(new_img)

            allY[count] = 1

            count += 1

        except:

            continue
    for f in other_files:
        try:
            img=cv2.imread(f,1)
            new_img=imresize(img,(size_image,size_image,3))
            allX[count]=np.array(new_img)
            allY[count]=2
            count+=1
        except:
            continue
    X, X_test, Y, Y_test = train_test_split(allX, allY, test_size=0.1, random_state=42)
    Y = to_categorical(Y, 3)
    Y_test = to_categorical(Y_test, 3)
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    network = input_data(shape=[None, 64, 64, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
    conv_1 = conv_2d(network, 16, 3,1,'same', activation='relu', name='conv_1')
    conv_2 = conv_2d(conv_1, 16, 3,1,'same', activation='relu', name='conv_2')
    network = max_pool_2d(conv_2,2,2,'same')
    conv_3 = conv_2d(network, 16, 3,1,'same', activation='relu', name='conv_3')
    conv_4 = conv_2d(conv_3, 1, 3,1,'same', activation='relu', name='conv_4')
    network = max_pool_2d(conv_4,2,2,'same')
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0005, metric=acc)
    model = tflearn.DNN(network, checkpoint_path='model_fire_smoke_6.tflearn', max_checkpoints = 3,
                        tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
    model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
          n_epoch=100, run_id='model_fire_smoke_6', show_metric=True)
    model.save('model_fire_smoke_6_final.tflearn')


