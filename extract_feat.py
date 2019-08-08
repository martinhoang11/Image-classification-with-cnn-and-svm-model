import numpy as np
import tensorflow as tf
import vgg16
import cv2
import os

# Change this path
DATASET_DIR = './Images/run/'

if not os.path.exists('./npydataset'):
    os.mkdir('./npydataset')

def load_image(PATH):
    batch = []
    folder_name = []
    img = [cv2.imread(PATH + file) for file in os.listdir(PATH)]
    img = [cv2.resize(file,(224,224),3) for file in img]
    folder_name = [file.replace(file.split('.')[1],"npy") for file in os.listdir(PATH)]
    batch = [file.reshape((224, 224, 3)) for file in img] # 1 image resized in 224x224x3
    return batch, folder_name

# Neu batch nay ma truyen vao mot luc 20 tam hinh
# thi no se tra ve fc6 kich thuoc 20

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
def extract_data_feature(path):
    batch, folder_name = load_image(path)
    batch_size = len(batch)
    i = 1
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            print('Loading model...')
            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            print("Extracting feature...")
            fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
            print('FC6 feature: ', fc6)
            for x in fc6:
                np.save('./npydataset' + folder_name[i],x)
                print('Saved ' , i )
                i = i + 1
            print('Number of input: ', len(fc6))
            print('Feature length of FC6: ', len(fc6[0]))

extract_data_feature(DATASET_DIR)
