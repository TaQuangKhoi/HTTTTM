import PIL
import cv2, numpy, os
from PIL import Image
from cv2 import face

path="Datasets"

recognizer= face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:
        try:
            PIL_img=Image.open(imagePath).convert('L')
            img_numpy=numpy.array(PIL_img,"uint8")

            id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces=detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        except PIL.UnidentifiedImageError:
            print(f"Unable to identify image: {imagePath}")
            continue

    return faceSamples,ids

print("\n INFO Đang trainning dữ liệu...")
faces,ids=getImagesAndLabels(path)
recognizer.train(faces,numpy.array(ids))

recognizer.write("trainer/trainer.yml")

print("\n INFO {0} khuôn mặt đã được trainning. Thoát".format(len(numpy.unique(ids))))