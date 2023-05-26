import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face.yml')
cascade_path = "/Users/xupeihan/PycharmProjects/LBPH20230522/haarcascade_frontalface_default.xml"
face_casade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("cannot receive frame")
        break
    img = cv2.resize(img, (540,300))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_casade.detectMultiScale(gray)

    #建立姓名和id的對照表
    name = {
        '1':'Tsai',
        '2':'Trump',
        '3':'oxxostudio'
    }
    #依序判斷每張臉屬於哪個id
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,0),2) #標記人臉外框
        idum, confidence = recognizer.predict(gray[y:y+h,x:x+w]) #取出id號碼以及信箱指數confidence
        if confidence < 60:
            text = name[str(idum)] #如果信心指數小於60 取得對應的名字
        else:
            text = '???' #不然名字就是？？？
        #在人臉外框加上名字
        cv2.putText(img, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,225,0), 2, cv2.LINE_AA)

    cv2.imshow('oxoxstudio', img)
    if cv2.waitKey(5) == ord('q'):
        break #按下q鍵停止
cap.release()
cv2.destroyAllWindows()