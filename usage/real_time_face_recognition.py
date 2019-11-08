import time

import cv2
import os
import usage.face

import src.facenet
import argparse
import numpy as np
import random
import pickle
from sklearn.naive_bayes import GaussianNB

window_name = "Real_Time_Facial_Recognition"


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

    return frame
'''
        main함수 설명
        0. 얼굴 없음 / 얼굴 두명 이상 안내문 처리
        1. 새 고객 들어오면 Unkown 뜨고 50프레임동안 등록 검사. 
        2. 고객 이름 입력받음
        3. (capture_customer)화면 봐달라고 하고 얼굴 등장 사진 10장 캡쳐한 후,
         기존 임베딩 값에 추가해서 emb.npy 저장.
         classifier model train 하고 클래스 추가해서 pickle 파일 저장. 
        4. 다시 로드 
'''
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
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    facenet_model_checkpoint = os.path.dirname(__file__) + args.model
    classifier_model = os.path.dirname(__file__) + args.classifier
    face_recognition = usage.face.Recognition(facenet_model_checkpoint, classifier_model, min_face_size=20)
    start_time = time.time()

    unknown_count = 0
    encoder = usage.face.Encoder(facenet_model_checkpoint)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret == 0:
            print("Error: check if webcam is connected.o")
            return
        # 식별 #
        faces = face_recognition.identify(frame)

        # unknown #
        #얼굴 있을 때,
        if faces != []:
            #얼굴 2명 이상일 때 안내하고 넘어감.
            if len(faces) > 1:
                print("2명이상 등장")
                delay = 1
                close_time = time.time() + delay
                while True:
                    cv2.waitKey(3)
                    _, frame2 = video_capture.read()
                    cv2.imshow(window_name, frame2)
                    cv2.putText(frame2, "Only one person face!", (20, 250),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                thickness=2, lineType=2)
                    cv2.imshow(window_name, frame2)  # 업데이트
                    if time.time() > close_time:
                        break
                unknown_count = 0
                continue
            #한명&unknown이면 일정 프레임 이상 검사 후 등록 처리
            if faces[0].name == 'Unknown':
                print("unknown!", unknown_count)
                unknown_count += 1
                if unknown_count == 50:  # 50프레임동안 unknown이면 고객 등록
                    print("[얼굴 unknown]")
                    # 고객 이름 입력받아서 폴더 생성
                    #name = input("input name: ")
                    name = str(random.randint(0, 100))
                    print("이름 : ", name)

                    # 화면 보도록 텍스트 띄움
                    delay = 3
                    close_time = time.time() + delay
                    while True:
                        cv2.waitKey(3)
                        ret2, frame2 = video_capture.read()
                        cv2.imshow(window_name, frame2)
                        f1 = frame2.copy()
                        cv2.putText(f1, "Start facial recognition registration.", (20, 200),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                    thickness=2, lineType=2)
                        cv2.putText(f1, "Look Screen!", (20, 250),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                                    thickness=2, lineType=2)
                        cv2.imshow(window_name, f1)  # 업데이트
                        if time.time() > close_time:
                            break

                    # 고객 얼굴 이미지 캡쳐 ->embading 값 append하고 저장
                    capture_customer(video_capture, window_name, capture_count, name, args.classifier, encoder, face_recognition)
                    cv2.imshow(window_name, frame)

                    # 등록 완료 & 대기 안내
                    delay = 2
                    close_time = time.time() + delay
                    while True:
                        cv2.waitKey(3)
                        ret2, frame2 = video_capture.read()
                        cv2.imshow(window_name, frame2)
                        f1 = frame2.copy()
                        cv2.putText(f1, "Facial recognition registration completed!", (20, 200),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                    thickness=2, lineType=2)
                        cv2.putText(f1, "ReLoading... wait a moment...", (20, 250),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                    thickness=2, lineType=2)
                        cv2.imshow(window_name, f1)  # 업데이트
                        if time.time() > close_time:
                            break

                    #다시 로딩
                    face_recognition = usage.face.Recognition(facenet_model_checkpoint, classifier_model, min_face_size=20)


                    unknown_count = 0
        #프레임 계산
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
def capture_customer(video_capture, window_name, capture_count, name, classifier, encoder, face_recognition):
    capture_img = []

    #10장 캡쳐
    for i in range(0, capture_count):
        delay = 0.5
        close_time = time.time() + delay
        while True:
            cv2.waitKey(3)
            _, only_cam = video_capture.read()
            cv2.putText(only_cam, "During shooting..." + "( " + str(i + 1) + " )", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255), thickness=1, lineType=2)

            cv2.imshow(window_name, only_cam)  # 업데이트
            if time.time() > close_time:
                break
        faces = face_recognition.identify(only_cam)
        #얼굴 하나 있을 때 캡쳐하기
        while len(faces) != 1:
            cv2.waitKey(5)
            _, only_cam = video_capture.read()
            cv2.imshow(window_name, only_cam)  # 업데이트
            faces = face_recognition.identify(only_cam)
        capture_img.append(only_cam)

    #캡쳐 사진 당 얼굴 추출
    faces = src.facenet.captureimg_face(capture_img)

    #얼굴들 임베딩 추출해서 face_emb 넘파이 배열에 저장
    face_emb=np.ndarray([capture_count, 512])
    for i, face in enumerate(faces):
        face.embedding = encoder.generate_embedding(face)
        face_emb[i, :] = face.embedding
    print("face_emb.shape",len(face_emb))

    #기존 임베딩에 추가
    data = np.load("emb.npy")
    print("data.shape",data.shape)
    data=np.vstack([data, face_emb])
    print("app_data.shpae", data.shape)

    #추가된 임베딩 값 저장
    np.save("emb.npy", data)

    #새 고객 이름 추가.
    with open('./models/datasets_classifier.pkl', 'rb') as infile:
        model, class_names = pickle.load(infile)
        print(class_names)
        print("name:",name)
        class_names.append(name)
        print("class_names",class_names)
        classifier_filename_exp = os.path.expanduser(classifier)

    #클래스명*10 만큼 label생성
    label = []
    print("len(class_names): ", len(class_names))
    for i in range(len(class_names)):
        for j in range(10):
            label.append(i)
    print("label len: ",len(label))
    print("label: ", label)
    print("data: ",data.shape)

    #classifier model 생성 - 추가된 고객 포함해 train
    model = GaussianNB()
    model.fit(data, label)

    #classifier model과 고객이름 클래스명 저장
    with open('./models/datasets_classifier.pkl', 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)

#사용x
# brief 고객 영어 이름 입력받고 폴더 생성 - join_customer
# details
# author yj
# date 11/03
def join_customer():
    try:
        # 아래 input으로 받으면 웹캠 화면 응답없음.
        #name = input("input name: ")
        name=str(random.randint(0, 100))
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
