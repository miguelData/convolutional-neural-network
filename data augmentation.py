
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.preprocessing import image as im
import keras.backend as K
import numpy as np
import os


# In[2]:


# you need to change this to your data directory
train_dir = 'C:/MIGUEL/ML/dogs and cats/train/'
save_dir = 'C:/MIGUEL/ML/dogs and cats/traingen/'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats_list = []
    label_cats = []
    dogs_list = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats_list.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs_list.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats_list), len(dogs_list)))

    tempDogs = np.array([dogs_list, label_dogs])
    tempCats = np.array([cats_list, label_cats])

    tempDogs = tempDogs.transpose()
    tempCats = tempCats.transpose()

    np.random.shuffle(tempDogs)
    np.random.shuffle(tempCats)
    
    dogs_list = list(tempDogs[:, 0])
    cats_list = list(tempCats[:, 0])

    label_dogs = list(tempDogs[:, 1])
    label_cats = list(tempCats[:, 1])

    label_dogs = [int(i) for i in label_dogs]
    label_cats = [int(i) for i in label_cats]

    
    
    return dogs_list,label_dogs,cats_list,label_cats


# In[3]:


def dataAugmente(dogOrCat,image, label, image_W, image_H, batch_size, capacity):
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

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label],num_epochs=1)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])#read_fine need 1 dimention string
    image = tf.image.decode_jpeg(image_contents, channels=3)

    
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, [image_W, image_H])


    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    
    datagen = im.ImageDataGenerator(
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            horizontal_flip=True,
                            fill_mode='nearest')
    with tf.Session() as sess:
    # Required to get the filename matching to run.
    

    # Coordinate the loading of image files.
#     sess.run(tf.initialize_all_variables())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        fileNum=0
        try:
            while 1:
                if coord.should_stop():
                    break

                imageBatch,labelBatch = sess.run([image_batch, label_batch])
                datagen.fit(imageBatch)
                data=datagen.flow(imageBatch, batch_size=batch_size
                             ,save_to_dir=save_dir,save_prefix=dogOrCat+'.gen',save_format='jpeg')
                for i in range(1):#generate pics, range(num) 1 pic generae num of times
                    data.next()

                fileNum=fileNum+1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            print(fileNum)
            coord.request_stop()
 

        coord.join(threads)
        sess.close()



    return 


# In[4]:


IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 224
BATCH_SIZE = 16
CAPACITY = 64


# In[5]:


dogs_list,label_dogs,cats_list,label_cats=get_files(train_dir)
dataAugmente('dog',dogs_list, label_dogs, IMG_W, IMG_H, BATCH_SIZE, CAPACITY )
dataAugmente('cat',cats_list, label_cats, IMG_W, IMG_H, BATCH_SIZE, CAPACITY )

