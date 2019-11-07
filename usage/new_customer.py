from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import src.facenet as facenet
import os
import argparse
import math
import pickle
from sklearn.naive_bayes import GaussianNB

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)

def retrain(args):
    args=parse_arguments(args)
    try:
        with tf.Graph().as_default():

            with tf.Session() as sess:

                np.random.seed(seed=666)

                dataset = facenet.get_dataset(args.data_dir)

                # Check that there are at least one training image per class
                for cls in dataset:
                    assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(args.model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * args.batch_size
                    end_index = min((i + 1) * args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    print("paths_: ", paths_batch)
                    images = facenet.load_data_npy(paths_batch, False, False, 160)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                    # 여기서 임베딩값 저장
                    print(emb_array.shape)
                    np.save("emb.npy", emb_array)
                    print("emb.npy saved.")

                classifier_filename_exp = os.path.expanduser(args.classifier_filename)  # 절대경로로 바꿔줌
        # Train classifier
        print('Training classifier')
        model = GaussianNB()
        # model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    except ValueError:
        print("reload fail!")
    return True
