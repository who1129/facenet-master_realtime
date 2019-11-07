import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import src.align.detect_face as align_detect_face
import src.facenet as facenet


gpu_memory_fraction = 0.3
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self, facenet_model_checkpoint, classifier_model, min_face_size=20):
        self.detect = Detection(min_face_size=min_face_size)
        self.encoder = Encoder(facenet_model_checkpoint)
        self.identifier = Identifier(classifier_model)
    #얼굴 찾기
    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        #얼굴 하나인거 확인~!
        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces
        else:
            print("여기 처리해야함")
            #얼굴 2개일때 화면에 한 사람만 나오도록 출력

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            #여기서 임베딩값 뽑아냄
            face.embedding = self.encoder.generate_embedding(face)
            #여기서 누구인지 식별함
            #face.name = self.identifier.identify(face)
            #여기서 누구인지 확인 안되면 등록하는 함수로 넘어가게
            #if unkown?
            data = np.load("emb.npy")
            distances=[]
            #거리 뽑아내서
            print("emb data: ", data.shape)

            for i in range(len(data)):
                distances.append(np.sum(np.square(np.subtract(data[i],face.embedding))))#유클리디안 거리
                print(str(i)+'번째: '+str(distances[i]))
                #print(np.subtract(data[i],face.embedding))
            print("제일 가까운 이미지: ", np.argmin(distances))
            print("distances[np.argmin(distances)]: ", distances[np.argmin(distances)])

            if distances[np.argmin(distances)] < 0.5:
                face.name = self.identifier.identify(face)

            else:
                face.name = 'Unknown'
            print(data.shape)
            '''
            for i in range(0, data.shape[0]):
                print(str(i)+"번째 사진")
                print(distances[i])
            '''
        return faces


class Identifier:
    def __init__(self, classifier_model):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)
            print(self.class_names)

    def identify(self, face):
        if face.embedding is not None:
            #임베딩값으로 예측
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]#argmax가 두개일 수도...


class Encoder:
    def __init__(self, facenet_model_checkpoint):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        #얼굴 픽셀값 정규화
        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

#[얼굴+바운딩박스]리스트 반환
class Detection:
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32, min_face_size=20):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = min_face_size

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align_detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align_detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces
