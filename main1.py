import sys
import inception
from inception import transfer_values_cache
import tumordata
import prettytensor as pt
import os
import tensorflow as tf
import numpy as np

from transfer_learning2 import main

num_classes = 2

def predict_avg(imgpath):
    s1 = main("split1", imgpath)
    # s2, pred2 = main("split2", imgpath)
    # s3, pred3 = main("split3", imgpath)
    # s4, pred4 = main("split4", imgpath)
    # s5, pred5 = main("split5", imgpath)

    print(s1)

    # sumsplit = s1+s2+s3+s4+s5
    # meansplit = round(sumsplit/5, 2)

    # print("Total accuracy: ", meansplit)

    # print("The predictions are: ", pred1, pred2, pred3, pred4, pred5)

# def predict_cls(session, x, y_true, y_pred_cls, transfer_values, labels, cls_true):
#     batch_size = 256
#     # Number of images.
#     num_images = len(transfer_values)

#     # Allocate an array for the predicted classes which
#     # will be calculated in batches and filled into this array.
#     cls_pred = np.zeros(shape=num_images, dtype=np.int)

#     # Now calculate the predicted classes for the batches.
#     # We will just iterate through all the batches.

#     # The starting index for the next batch is denoted i.
#     i = 0

#     while i < num_images:
#         # The ending index for the next batch is denoted j.
#         j = min(i + batch_size, num_images)

#         # Create a feed-dict with the images and labels
#         # between index i and j.
#         feed_dict = {x: transfer_values[i:j],
#                      y_true: labels[i:j]}

#         # Calculate the predicted class using TensorFlow.
#         cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

#         # Set the start-index for the next batch to the
#         # end-index of the current batch.
#         i = j
        
#     # Create a boolean array whether each image is correctly classified.
#     correct = (cls_true == cls_pred)

#     return correct, cls_pred


# def classification_accuracy(correct):
#     # When averaging a boolean array, False means 0 and True means 1.
#     # So we are calculating: number of True / len(correct) which is
#     # the same as the classification accuracy.

#     # Return the classification accuracy
#     # and the number of correct classifications.
#     return correct.mean(), correct.sum()


# def predict(splitnum, imgpath):  


#     tumordata.add_data_path(splitnum)

#     tumordata.start()

#     print(tumordata.data_path)

#     class_names = tumordata.load_class_names()
#     images_train, cls_train, labels_train = tumordata.load_training_data()
#     images_test, cls_test, labels_test = tumordata.load_testing_data()

#     print("Size of:")
#     print("- Training-set:\t\t{}".format(len(images_train)))
#     print("- Test-set:\t\t{}".format(len(images_test)))

#     # inception dir
#     inception.data_dir = 'inception/'

#     # load model
#     model = inception.Inception()

#     # caches for training and test sets
#     file_path_cache_train = os.path.join(tumordata.data_path, 'inception_tumordata_train.pkl')
#     file_path_cache_test = os.path.join(tumordata.data_path, 'inception_tumordata_test.pkl')

#     print("Processing Inception transfer-values for training-images ...")

#     # If transfer-values have already been calculated then reload them,
#     # otherwise calculate them and save them to a cache-file.
#     transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
#                                                   images=images_train,
#                                                   model=model)

#     print("Processing Inception transfer-values for test-images ...")

#     # If transfer-values have already been calculated then reload them,
#     # otherwise calculate them and save them to a cache-file.
#     transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
#                                                  images=images_test,
#                                                  model=model)

#     transfer_len = model.transfer_len

#     x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

#     y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

#     y_true_cls = tf.argmax(y_true, dimension=1)

#     # Wrap the transfer-values as a Pretty Tensor object.
#     x_pretty = pt.wrap(x)

#     with pt.defaults_scope(activation_fn=tf.nn.relu):
#         y_pred, loss = x_pretty.\
#             fully_connected(size=1024, name='layer_fc1').\
#             softmax_classifier(num_classes=num_classes, labels=y_true)


#     y_pred_cls = tf.argmax(y_pred, dimension=1)

#     correct_prediction = tf.equal(y_pred_cls, y_true_cls)

#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#     sess = tf.Session('', tf.Graph())
#     with sess.graph.as_default():
#         # Read meta graph and checkpoint to restore tf session
#         saver = tf.train.import_meta_graph("checkpoints\\split1\\my-model-200.meta")
#         saver.restore(sess, "checkpoints\\split1\\my-model-200")

#         sess.run(tf.global_variables_initializer())

#         correct, cls_pred = predict_cls(sess, x, y_true, y_pred_cls, transfer_values_test, labels_test, cls_test)
#         acc, num_correct = classification_accuracy(correct) 
#         print("num correct: ",num_correct)
        
#         prediction = model.classify(image_path=imgpath)
#         print("PREDICTION: ", prediction)
#         # model.print_scores(pred=prediction, k=2)

#         print("ACCURACY: ", acc)



if __name__ == '__main__':
    imgpath = "D:\\AI stuff\\aiproject-inception-master\\breakhis\\BreaKHis_data\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\fibroadenoma\\SOB_B_F_14-23222AB\\40X\\SOB_B_F-14-23222AB-40-001.png"
    predict_avg(imgpath)

