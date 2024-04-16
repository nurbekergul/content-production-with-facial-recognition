import cv2
import math
import time
import argparse

def getFaceBox(net, frame, conf_threshold=0.80):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

def showContentBasedOnAgeAndGender(age, gender):
    if gender == 'Male':
        if age == '(0-2)':
            print("Erkek çocuk içeriği gösteriliyor: Oyuncak")
        elif age == '(4-6)':
            print("Erkek çocuk içeriği gösteriliyor: Oyuncak")
        elif age == '(8-12)':
            print("Erkek çocuk içeriği gösteriliyor: Oyuncak")
        elif age == '(15-20)':
            print("Genç erkek içeriği gösteriliyor: Araba")
        elif age == '(25-32)':
            print("Genç erkek içeriği gösteriliyor: Araba")
        elif age == '(38-43)':
            print("Orta yaşlı erkek içeriği gösteriliyor: Ev")
        elif age == '(48-53)':
            print("Orta yaşlı erkek içeriği gösteriliyor: Ev")
        elif age == '(60-100)':
            print("Yaşlı erkek içeriği gösteriliyor: Ev")
    elif gender == 'Female':
        if age == '(0-2)':
            print("Kız çocuk içeriği gösteriliyor: Barbie")
        elif age == '(4-6)':
            print("Kız çocuk içeriği gösteriliyor: Barbie")
        elif age == '(8-12)':
            print("Kız çocuk içeriği gösteriliyor: Barbie")
        elif age == '(15-20)':
            print("Genç kadın içeriği gösteriliyor: Gratis")
        elif age == '(25-32)':
            print("Genç kadın içeriği gösteriliyor: Gratis")
        elif age == '(38-43)':
            print("Orta yaşlı kadın içeriği gösteriliyor: Estetik")
        elif age == '(48-53)':
            print("Orta yaşlı kadın içeriği gösteriliyor: Estetik")
        elif age == '(60-100)':
            print("Yaşlı kadın içeriği gösteriliyor: Estetik")

def showCustomContent(age, gender):
    img_path = ""
    
    if gender == 'Male':
        if age == '(15-20)' or age == '(25-32)':
            img_path = "araba.jpg"
        elif age == '(38-43)' or age == '(48-53)' or age == '(60-100)':
            img_path = "ev.jpg"
    elif gender == 'Female':
        if age == '(15-20)' or age == '(25-32)':
            img_path = "gratis.jpg"
        elif age == '(38-43)' or age == '(48-53)' or age == '(60-100)':
            img_path = "estetik.jpg"
    elif age == '(0-2)':
        img_path = "barbie.jpg"
    elif age == '(4-6)' or age == '(8-12)':
        img_path = "oyuncak.jpg"

    if img_path:
        img = cv2.imread(img_path)
        cv2.imshow("Özel İçerik", img)

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Ağları yükle, DNN
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(0)
padding = 20

genç_erkek_içerikleri = ["araba.jpg", "uçak.jpg"]
geçiş_süresi = 5  # İçerikler arasındaki geçiş süresi (saniye)
içerik_sayacı = 0
içerik_zamanı = 0

while cv2.waitKey(1) < 0:
    # Kareyi oku
    t = time.time()
    hasFrame, frame = cap.read()

    if not hasFrame:
        cv2.waitKey()
        break

    # Daha iyi optimizasyon için daha küçük bir kare oluştur
    small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

    frameFace, bboxes = getFaceBox(faceNet, small_frame)

    if not bboxes:
        print("Yüz bulunamadı, bir sonraki kareye geçiliyor.")
        continue

    for bbox in bboxes:
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Cinsiyet Tahmini
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Yaş tahmini
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Yaş ve Cinsiyeti göster
        label = "{}, {}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
        
        # Yaş ve cinsiyete göre içerikleri göster

        showContentBasedOnAgeAndGender(age, gender)

        # Özel içeriği göster
        current_time = time.time()
        
        if current_time - içerik_zamanı >= geçiş_süresi:
            içerik_sayacı = (içerik_sayacı + 1) % 2  # İçerikler arasında döngü yapar (0, 1, 0, 1...)
            içerik_zamanı = current_time

            if içerik_sayacı == 0:
                img_path = "araba.jpg"
            else:
                img_path = "ucak.jpg"

            img = cv2.imread(img_path)
            cv2.imshow("Kisiye Ozel Icerik", img)

    cv2.imshow("YuzAnaliz", frameFace)
    print("FPS: {:.2f}".format(1 / (time.time() - t)))

cv2.destroyAllWindows()
cap.release()
