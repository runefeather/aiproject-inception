from transfer_learning import main
import sys
import inception
from inception import transfer_values_cache
import tumordata
import prettytensor as pt


NUM_CLASSES = 2

def predict_avg(imgpath):
	s1, pred1 = main("split1", imgpath)
	s2, pred2 = main("split2", imgpath)
	s3, pred3 = main("split3", imgpath)
	s4, pred4 = main("split4", imgpath)
	s5, pred5 = main("split5", imgpath)

	sumsplit = s1+s2+s3+s4+s5
	meansplit = round(sumsplit/5, 2)

	print("Total accuracy: ", meansplit)

	print("The predictions are: ", pred1, pred2, pred3, pred4, pred5)


def predict(splitnum, imgpath):  

    tumordata.add_data_path(splitnum)

    tumordata.start()

    print(tumordata.data_path)

    class_names = tumordata.load_class_names()
    images_train, cls_train, labels_train = tumordata.load_training_data()
    images_test, cls_test, labels_test = tumordata.load_testing_data()

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))

    # inception dir
    inception.data_dir = 'inception/'

	# load model
    model = inception.Inception()

    # caches for training and test sets
    file_path_cache_train = os.path.join(tumordata.data_path, 'inception_tumordata_train.pkl')
    file_path_cache_test = os.path.join(tumordata.data_path, 'inception_tumordata_test.pkl')

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

    transfer_len = model.transfer_len

    x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    y_true_cls = tf.argmax(y_true, dimension=1)

    # Wrap the transfer-values as a Pretty Tensor object.
    x_pretty = pt.wrap(x)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            fully_connected(size=1024, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)


    y_pred_cls = tf.argmax(y_pred, dimension=1)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())

	saver = tf.train.import_meta_graph('my-model-200.meta')
	saver.restore(session, 'my-model-200')




if __name__ == '__main__':
	imgpath = ""
	predict("split1", imgpath)