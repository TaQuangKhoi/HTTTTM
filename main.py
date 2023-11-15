import PIL
from flask import Flask, render_template, request, redirect
import cv2, numpy, os
from cv2 import face
from PIL import Image
import base64


app = Flask(__name__)

def rtmpServer = "rtmp://35.185.190.46/live/dangbaokhang"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture') 
def capture():
    haar_file='haarcascade_frontalface_default.xml'

    datasets='Datasets'

    path=os.path.join(datasets)
    if not os.path.isdir(path):
        os.mkdir(path)
        
    (width, height)=(130, 100)

    face_cascade=cv2.CascadeClassifier(haar_file)
    webcam=cv2.VideoCapture(rtmpServer)

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    tam=0
    for imagePath in imagePaths:
        try:
            id=int(os.path.split(imagePath)[-1].split(".")[1])
            if tam<=id: 
                tam=id
                tam+=1
        except PIL.UnidentifiedImageError:
            print(f"Unable to identify image: {imagePath}")
            continue
        except ValueError:
            print(f"Skipping invalid image: {imagePath}")
            continue

    
    face_id=tam
    print("\n Nhập id khuôn mặt <return> ==> ",tam)

    print("\n Khởi tạo camera...")

    count=0
    captured_images = []

    while count<30:
        (_, im) = webcam.read()
        gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,4)

        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face=gray[y:y+h,x:x+w]
            count+=1
            face_resize=cv2.resize(face,(width,height))

            cv2.imwrite("Datasets/User."+str(face_id)+"."+str(count)+".png",face_resize)
            print(count)
            # _, buffer = cv2.imencode('.png', im)
            # img_str = base64.b64encode(buffer).decode('utf-8')
            # captured_images.append(img_str)
            
        # cv2.imshow('OpenCV', im)
        # key = cv2.waitKey(10)& 0xff
        # if key == 27:
        #     break
    print("\n Thoát")
    webcam.release()
    # cv2.destroyAllWindows() 
    return redirect('/train')

@app.route('/train')
def train():
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
    return redirect('/recognize')  

@app.route('/recognize')
def recognize():
    recognizer= cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    font=cv2.FONT_HERSHEY_SIMPLEX

    id=0

    names=['Nguyen Phuong Thanh', '1','2',"Dang Bao Khang"]
    webcam=cv2.VideoCapture(rtmpServer)

    while True:
        ret, img= webcam.read()
        # img=cv2.flip(img,-1)

        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces=detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence<100):
                id=names[id]
                confidence="{0}%".format(round(100-confidence))
            else:
                id="unknown"
                confidence="{0}%".format(round(100-confidence))
            cv2.putText(img, str(id),(x+5,y-5), font,1,(255,255,255),2)
            cv2.putText(img, str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)

        # cv2.imshow("Nhận diện khuôn mặt",img)

        k=cv2.waitKey(10)& 0xff
        if k==27:
            break

    print("\n Thoát")
    webcam.release()
    # cv2.destroyAllWindows()
    return render_template('recognize.html') 

if __name__ == '__main__':
    app.run(debug=True)