import cv2

recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font=cv2.FONT_HERSHEY_SIMPLEX

id=0

names=['Dang Bao Khang', 'Vo Pham Thuan Khang','thien',"Dang Bao Khang"]
webcam=cv2.VideoCapture(0)

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

    cv2.imshow("Nhan dien khuon mat",img)

    k=cv2.waitKey(10)& 0xff
    if k==27:
        break

print("\n Thoát")
webcam.release()
cv2.destroyAllWindows()
