import argparse
import sys
import time

import cv2
import os
import usage.face
import datetime
import src.facenet
import argparse
import numpy as np
from threading import Thread
from PIL import ImageFont, ImageDraw, Image
import new_customer
import random

window_name = "Real_Time_Facial_Recognition"


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.stream=Thread(target=self.show_frame(), args=())
        self.stream.start()
    def run(self):
        print("start T")
        webcam()
        print("end T")
    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:  # 연결안됨
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                cv2.waitKey(2)

    def stop(self):
        self.stopped = True
    def show_frame(self):
        cv2.imshow(window_name, self.frame)



#############################
# frame이라는 이미지에 글씨 넣는 함수
# frame : 카메라 이미지
# str : 문자열 변수
# (0, 100) : 문자열이 표시될 좌표 x = 0, y = 100
# cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 폰트 형태
# 1 : 문자열 크기(scale) 소수점 사용가능
# (0, 255, 0) : 문자열 색상 (r,g,b)


def add_overlays(frame, faces, frame_rate):
    if faces is not None:  # 객체가 들어오는?
        if faces == []:
            cv2.putText(frame, "Look Screen.", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=2,
                        lineType=2)
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 153, 255), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (84, 112, 9),
                            thickness=2, lineType=2)

    '''
    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                thickness=2, lineType=2)
    '''
    return frame


'''
        1. 얼굴 없는 사람 들어오면 Unkown 뜨면서 등록하시겠습니까 
        2. 고객 영어 이름 입력받고 폴더 생성 - join_customer
        3. 다음 누르면 puttext로 화면 봐달라고 하고
        4. 캡쳐 저장
        5. 다시 임베딩 로드해서 결제 진행 
'''

def webcam():
    video_capture = VideoGet().start()
    print("T start!")

    while True:
        frame = video_capture.frame
        print(frame)
        cv2.imshow("Video", frame)
        cv2.waitKey(30)

def main(args):
    frame_interval = 30  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    capture_count = 10

    if args.debug:
        print("Debug enabled")
        usage.face.debug = True
    # webcam
    '''
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    '''
    video_capture = VideoGet()
    print("T start!")
    '''
    while True:
        frame = video_capture.frame
        print(frame)
        cv2.imshow("Video", frame)
        cv2.waitKey(30)
    '''

    facenet_model_checkpoint = os.path.dirname(__file__) + args.model
    classifier_model = os.path.dirname(__file__) + args.classifier
    face_recognition = usage.face.Recognition(facenet_model_checkpoint, classifier_model, min_face_size=20)
    start_time = time.time()
    '''
    ret, frame = video_capture.read()
    name = "no"
    cap_and_save = threading.Thread(target=capture_customer,
                                    args=(video_capture, window_name, frame, capture_count, name))
    '''
    unknown_count = 0

    while True:
        # Capture frame-by-frame
        # ret, frame = video_capture.read()
        video_capture.show_frame()
        print("???")
        frame = video_capture.frame
        ret = video_capture.grabbed
        if ret == 0:
            print("Error: check if webcam is connected.o")
            return
        # 식별
        faces = face_recognition.identify(frame)

        # unknown
        if faces != []:
            if len(faces) > 1:
                print("2명이상 등장")
                cv2.putText(frame, "Only one face.", (20, 200),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                            thickness=2, lineType=2)
                cv2.imshow(window_name, frame)  # 업데이트
                continue

            if faces[0].name == 'Unknown':
                print("unknown!", unknown_count)

                unknown_count += 1
                if unknown_count == 5:  # 50프레임동안 unknown이면 고객 등록
                    print("[얼굴 unknown]")
                    # 고객 이름 입력받아서 폴더 생성
                    name = join_customer()
                    print("이름 : ", name)

                    # 화면 보도록 텍스트 띄움
                    f1 = frame.copy()
                    cv2.putText(f1, "Start facial recognition registration.", (20, 200),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                thickness=2, lineType=2)
                    cv2.putText(f1, "Look Screen!", (20, 250),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                thickness=2, lineType=2)
                    cv2.imshow(window_name, f1)  # 업데이트
                    cv2.waitKey(3000)

                    # 고객 얼굴 이미지 캡쳐->처리함수에서 저장
                    capture_customer(video_capture, window_name, frame, capture_count, name)
                    # print("쓰레드 호출")
                    # cap_and_save.start()
                    # print("쓰레드 종료??")
                    cv2.imshow(window_name, frame)

                    # 저장 완료 안내
                    f2 = frame.copy()
                    cv2.putText(f2, "Facial recognition registration completed.", (20, 200),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                thickness=2, lineType=2)
                    cv2.putText(f2, "wait a moment...", (20, 250),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                thickness=2, lineType=2)
                    cv2.imshow(window_name, f2)  # 업데이트
                    cv2.waitKey(5000)

                    # 다시 로딩.
                    arguments = ['datasets\\Customer', 'models\\20180402-114759\\20180402-114759.pb',
                                 'models\datasets_classifier.pkl', '--batch_size', '1000']
                    flag = new_customer.retrain(arguments)
                    face_recognition = usage.face.Recognition(facenet_model_checkpoint, classifier_model,
                                                              min_face_size=20)
                    if flag:
                        print("reload success!")

                    unknown_count = 0

        if (frame_count % frame_interval) == 0:
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        new_frame = add_overlays(frame.copy(), faces, frame_rate)

        frame_count += 1
        cv2.imshow(window_name, new_frame)  # 업데이트
        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27:  # ESC key
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


# brief 고객 얼굴 이미지 캡쳐->처리함수에서 저장
# details
# author yj
# date 10/31
def capture_customer(video_capture, window_name, frame, capture_count, name):
    # print("start thread!!!")
    capture_img = []
    customer_path = './datasets/Customer/' + name
    for i in range(0, capture_count):
        cv2.waitKey(500)
        # _, only_cam=video_capture.read()
        only_cam = video_capture.frame
        capture_img.append(only_cam)
        nf = frame.copy()
        cv2.putText(nf, "During shooting..." + "( " + str(i + 1) + " )", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), thickness=1, lineType=2)
        cv2.imshow(window_name, nf)  # 업데이트
    src.facenet.captureimg_save_npy(capture_img, False, False, customer_path)
    print('ALL Screenshot saved!')


# brief 고객 영어 이름 입력받고 폴더 생성 - join_customer
# details
# author yj
# date 11/03
def join_customer():
    try:
        # 아래 input으로 받으면 웹캠 화면 응답없음.
        name = input("input name: ")
        # name=str(random.randint(0, 100))
        if not (os.path.isdir('.\\datasets\\Customer\\' + name)):
            os.makedirs(os.path.join('.\\datasets\\Customer\\' + name))
            print('.\\datasets\\Customer\\' + name)
    except OSError as e:
        print("Failed to create directory : " + '.\\datasets\\Customer\\' + name)
    return name


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    parser.add_argument('--model', help='Model to use.', required=True)
    parser.add_argument('--classifier', help='Classifier to use.', required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(['--model', '/models/20180402-114759', '--classifier', '/models/datasets_classifier.pkl']))
    # main(parse_arguments(sys.argv[1:]))
