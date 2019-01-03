
# coding: utf-8

# In[1]:


import tensorflow as tf
import import_ipynb
import os
import numpy as np
import input_train_val_split
import model
import numpy as np


# In[2]:


saver=tf.train.import_meta_graph("C:/MIGUEL/ML/dogs and cats/logs/train/model.ckpt-40.meta")
bn_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
x,y_,training,train_op, loss, acc,summary_op,logits=tf.get_collection("retrain_ops")#,bn_update_ops


# In[3]:


N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
RATIO = 0.2
MAX_STEP = 1000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


# In[4]:


# you need to change the directories to yours.
train_dir = 'C:/MIGUEL/ML/dogs and cats/train/'
logs_train_dir = 'C:/MIGUEL/ML/dogs and cats/logs/train/retrain/saveretrain'
logs_val_dir = 'C:/MIGUEL/ML/dogs and cats/logs/val/'
    
train, train_label, val, val_label = input_train_val_split.get_files(train_dir, RATIO)
train_batch, train_label_batch = input_train_val_split.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
val_batch, val_label_batch = input_train_val_split.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)


# In[5]:


train_writer = tf.summary.FileWriter(logs_train_dir, tf.get_default_graph())
val_writer = tf.summary.FileWriter(logs_val_dir, tf.get_default_graph())


# In[6]:


with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess,"C:/MIGUEL/ML/dogs and cats/logs/train/model.ckpt-40")
        

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
          
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])

                if step % 50 != 0:
                    _, lossValue, accValue,bnUpdateOps = sess.run([train_op, loss, acc,bn_update_ops]
                                                                  ,feed_dict={x:tra_images, y_:tra_labels,training:True})

               
                else : # means step % 50 == 0 
                    _, lossValue, accValue,bnUpdateOps,summary_str = sess.run([train_op, loss, acc,bn_update_ops,summary_op]
                                                                              ,feed_dict={x:tra_images, y_:tra_labels,training:True})

                    train_writer.add_summary(summary_str, step)
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, lossValue, accValue*100.0))
                
                if step % 200 == 0 or (step) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc ,summary_str= sess.run([loss, acc,summary_op], 
                                                 feed_dict={x:val_images, y_:val_labels})
                    
                    val_writer.add_summary(summary_str, step)  
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                
                
            
                if step % 20 == 0 or (step) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
        coord.join(threads)
        sess.close()
    

