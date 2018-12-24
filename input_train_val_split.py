
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import math


# In[2]:


# you need to change this to your data directory

def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(i) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(i) for i in val_labels]
#     print('tra_labels')
#     print(type(tra_labels[0]))
#     print (isinstance(tra_labels[0],int))
#     print('tra_images')
#     print(type(tra_images[0]))
#     print (isinstance(tra_images[0],int))
    
    return tra_images,tra_labels,val_images,val_labels


# In[3]:


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
#     print (isinstance(label[0],int))
#     print(type(label[0]))
#     print(label[0].dtype)
#     print('type(label[0])')
    
#     print (isinstance(image[0],int))
#     print(type(image[0]))
#     print(image[0].dtype)
#     print('type(image[0])')
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    
    # if you want to test the generated batches of images, you might want to comment the following line.
    

    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    print(label_batch)
    print('before label_batch')
    label_batch = tf.reshape(label_batch, [batch_size])
    label_batch = tf.cast(label_batch, tf.int32)
    image_batch = tf.cast(image_batch, tf.float32)
#     print (isinstance(label_batch[0],int))
#     print(type(label_batch[0]))
#     print(label_batch[0].dtype)
    print(label_batch)
    print('after label_batch')
#     print('type(label_batch[0])')
#     print (isinstance(image_batch[0],int))
#     print(type(image_batch[0]))
#     print(image_batch[0].dtype)
#     print('type(image_batch[0])')
#     print(image_batch.dtype)
#     print('type(image_batch)')
    
    return image_batch, label_batch

