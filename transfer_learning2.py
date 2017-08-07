import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import sys
from PIL import Image

# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache

# We use Pretty Tensor to define the new classifier.
import prettytensor as pt

# train and test data
import tumordata
from tumordata import num_classes

# load split data
import transfer

# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

# ========================================================================

def main(splitnum, finalimgpath):

    # print stuff 
    print("=========================================")
    print(splitnum)

    train_batch_size = 64   


    def random_batch():
        # Number of images (transfer-values) in the training-set.
        num_images = len(transfer_values_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)

        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = transfer_values_train[idx]
        y_batch = labels_train[idx]

        return x_batch, y_batch



    def optimize(num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        # Else, save for the first time
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
        session_list = []
        test_acc = []

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images (transfer-values) and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = random_batch()

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = session.run([global_step, optimizer],
                                      feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                savepath = saver.save(session, 'checkpoints\\split1\\model', global_step=i_global)
                session_list.append(savepath)

                # Calculate the accuracy on the training-batch.
                batch_acc = session.run(accuracy,
                                        feed_dict=feed_dict_train)

                # Test accuracy with session 
                correct, cls_pred = predict_cls_test()
                acc, num_correct = classification_accuracy(correct)

                # Print status.
                # msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                # print(msg.format(i_global, batch_acc))
                # print("Testing Accuracy: ", round(acc*100, 2))

                # save test accuracy
                test_acc.append(round(acc*100, 2))
                # print("========================================================")


        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        max_acc = max(test_acc)

        print("MAX: ", max_acc)
        # print(list(test_acc.values()).index(max_acc))
        print(test_acc.index(max_acc))
        # print(mydict.values().index(max(test_acc.values()))) #
        pth = session_list[test_acc.index(max_acc)]

        saver.restore(session, pth)
        return

    def plot_example_errors(cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)
        
        # Get the images from the test-set that have been
        # incorrectly classified.
        images = images_test[incorrect]
        
        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = cls_test[incorrect]

        n = min(9, len(images))
        
        # Plot the first n images.
        plot_images(images=images[0:n],
                    cls_true=cls_true[0:n],
                    cls_pred=cls_pred[0:n])


    def plot_confusion_matrix(cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.

        # Print the confusion matrix as text.
        for i in range(num_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)

        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(num_classes)]
        print("".join(class_numbers))


    # Split the data-set in batches of this size to limit RAM usage.
    batch_size = 256


    def predict_cls(transfer_values, labels, cls_true):
        # Number of images.
        num_images = len(transfer_values)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: transfer_values[i:j],
                         y_true: labels[i:j]}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
            
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct, cls_pred

    def predict_one_image(imgarr):
        # Number of images.
        num_images = 1

        label = np.zeros(shape=[0, 2], dtype=np.int)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        feed_dict = {x: imgarr, y_true: label}
        cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

        return cls_pred

    def predict_cls_test():
        return predict_cls(transfer_values = transfer_values_test,
                           labels = labels_test,
                           cls_true = cls_test)


    def classification_accuracy(correct):
        # When averaging a boolean array, False means 0 and True means 1.
        # So we are calculating: number of True / len(correct) which is
        # the same as the classification accuracy.

        # Return the classification accuracy
        # and the number of correct classifications.
        return correct.mean(), correct.sum()


    def print_test_accuracy(show_example_errors=False,
                            show_confusion_matrix=False):

        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        correct, cls_pred = predict_cls_test()
        
        # Classification accuracy and the number of correct classifications.
        acc, num_correct = classification_accuracy(correct)
        
        # Number of images being classified.
        num_images = len(correct)

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, num_correct, num_images))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)

    # =========================================================================================
    # THIS IS WHERE EVERYTHING COMES TOGETHER
    # =========================================================================================

    test, train, val = transfer.preprocessing("D:\\AI Stuff\\aiproject-inception-master\\breakhissplits_v2\\train_val_test_60_12_28\\")
    transfer.split(test, train, val, str(splitnum))

    tumordata.add_data_path(splitnum)

    tumordata.start()

    print(tumordata.data_path)

    class_names = tumordata.load_class_names()
    images_train, cls_train, labels_train = tumordata.load_training_data()
    images_test, cls_test, labels_test = tumordata.load_testing_data()

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))

    # Image to predict on
    img = Image.open(finalimgpath)
    imgarr = []
    imgarr.append(np.array(img))

    # inception dir
    inception.data_dir = 'inception/'

    # download the model
    inception.maybe_download()

    # load model
    model = inception.Inception()

    # caches for training and test sets
    file_path_cache_train = os.path.join(tumordata.data_path, 'inception_tumordata_train.pkl')
    file_path_cache_test = os.path.join(tumordata.data_path, 'inception_tumordata_test.pkl')

    file_path_cache_single_test = os.path.join(tumordata.data_path, 'inception_tumordata_single_test.pkl')

    print("Processing Inception transfer-values for training-images ...")

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                  images=images_train,
                                                  model=model)

    print("Processing Inception transfer-values for test-images ...")

    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                 images=images_test,
                                                 model=model)

    transfer_values_single_test = transfer_values_cache(cache_path=file_path_cache_single_test, images=imgarr, model=model)

    # print("TRANSFER VALUES TEST: ", transfer_values_test)

    transfer_len = model.transfer_len

    x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # x_one = tf.placeholder(tf.float32, shape=[len(imgarr), len(imgarr[0]), 3], name='x_one')
    # y_true_one = tf.placeholder(tf.float32, shape=1, name='y_true_one')    
    # y_true_cls_one = tf.argmax(y_true_one, dimension=1)

    # Wrap the transfer-values as a Pretty Tensor object.
    x_pretty = pt.wrap(x)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            fully_connected(size=1024, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

    y_pred_cls = tf.argmax(y_pred, dimension=1)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    session = tf.Session()
    session.run(tf.global_variables_initializer())


    print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)

    optimize(1000)

    print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)
    correct, cls_pred = predict_cls_test()
    acc, num_correct = classification_accuracy(correct)
    # print("acc, num correct: ", acc, num_correct)


    cls_pred = predict_one_image(transfer_values_single_test)
    # print("PREDICTION")
    # print(cls_pred)
    print(">>>>>>>>>>>><<<<<<<<<<<<<<<<")


    # prediction = model.classify(finalimgpath)

    return acc, cls_pred
