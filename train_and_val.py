
# coding: utf-8

# In[1]:


#%%
import import_ipynb
import os
import numpy as np
import tensorflow as tf
import input_train_val_split
import model


# In[2]:


#%%

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2 # take 20% of dataset as validation data 
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 25000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001



# In[3]:


#%%
def run_training():
    
    # you need to change the directories to yours.
    train_dir = 'C:/MIGUEL/ML/dogs and cats/train/'
    logs_train_dir = 'C:/MIGUEL/ML/dogs and cats/logs/train/'
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
    

    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3],name='x')
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE],name='y')
    training = tf.placeholder_with_default(False, shape=(), name='training')
    
#     print("y_")
#     print(y_)
#     print(y_.dtype)
#     print(type(y_))
#     print(y_.shape)
    logits = model.inference(x, BATCH_SIZE, N_CLASSES,training)
    loss = model.losses(logits, y_)  
    acc = model.evaluation(logits, y_)
    train_op = model.trainning(loss, learning_rate)
    

             
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
        bn_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
       
        for op in (x,y_,training,train_op, loss, acc,summary_op,logits):#,bn_update_ops
            tf.add_to_collection("retrain_ops",op)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
#                 print('train_label_batch')
#                 print(train_label_batch)
#                 print(train_label_batch.dtype)
#                 tra_images,tra_labels = sess.run([train_batch, train_label_batch])
#                 print('tra_labels')
#                 print(tra_labels)
#                 print(tra_labels.dtype)
#                 print('tra_labels.shape')
#                 print(tra_labels.shape)
#                 print('type(tra_labels)')
#                 print(type(tra_labels))
#                 print(tra_labels.dtype)
#                 print(tra_labels is y_)
#                 print(tra_labels is train_label_batch)
#                 print(isinstance(tra_labels, tf.int32))
#                 print(isinstance(y_, tf.int32))
#                 print('tra_images')
#                 print(tra_images)
#                 print(tra_images.dtype)

                if step % 50 != 0:
                    _, lossValue, accValue,bnUpdateOps = sess.run([train_op, loss, acc,bn_update_ops]
                                                                  ,feed_dict={x:tra_images, y_:tra_labels,training:True})

#             print('Step %d' %(step))
            
#             print('tra_labels')
#             print(tra_labels)
#             print(tra_labels.ndim)
#             print(tra_labels.dtype)
#             print('tra_labels.shape')
#             print(tra_labels.shape)
#             print('type(tra_labels)')
#             print(type(tra_labels))
#             print(tra_labels.dtype)
#             print(tra_labels is y_)
#             print(tra_labels is train_label_batch)
               
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

                if step % 2000 == 0 or (step ) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)


# In[ ]:



run_training()

