import cv2, sys, numpy, os, PIL


haar_file='haarcascade_frontalface_default.xml'

datasets='Datasets'

path='Datasets'

path=os.path.join(datasets)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height)=(500, 500)

imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

face_cascade=cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
tam=0
for imagePath in imagePaths:
    try:
        id=int(os.path.split(imagePath)[-1].split(".")[1])
        print("id",id)
        if tam<=id: 
            tam = id + 1
        print("3",tam)

    except PIL.UnidentifiedImageError:
        print(f"Unable to identify image: {imagePath}")
        continue
    except ValueError:
        print(f"Skipping invalid image: {imagePath}")
        continue

print("1",tam)
face_id=tam
print("\n Nhập id khuôn mặt <return> ==> ", face_id)
print("\n Khởi tạo camera...")

count=0
while count<30:
    (_, im) = webcam.read()
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        count+=1
        face_resize=cv2.resize(face,(width,height))

        cv2.imwrite(f"Datasets/User.{str(face_id)}.{count}.png", face_resize)

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)& 0xff
    if key == 27:
        break

print("\n Thoát")
webcam.release()
cv2.destroyAllWindows()