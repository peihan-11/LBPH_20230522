import cv2
import numpy as np

detector = cv2.CascadeClassifier('/Users/xupeihan/PycharmProjects/LBPH20230522/haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer_create() # 啟用訓練人臉模型方法
faces = [] # 儲存人臉位置大小的串列
ids = [] # 記錄該人臉 id 的串列
for i in range(1,10):
    img = cv2.imread(f'/Users/xupeihan/PycharmProjects/LBPH20230522/pic_TIW/{i}.jpg') #依序開啟每一張蔡英文的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #色彩轉換成黑白
    img_np = np.array(gray,'uint8') #轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray) #擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(1)

for i in range(1,10):
    img = cv2.imread(f'/Users/xupeihan/PycharmProjects/LBPH20230522/pic_trump/{i}.jpg') #依序開啟每一張川普的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #色彩轉換成黑白
    img_np = np.array(gray,'uint8') #轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray) #擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(2)

print('camera...') #提示啟用相機
cap = cv2.VideoCapture(0) #啟用相機
if not cap.isOpened():
    print("cannnot open camera")
    exit()
while True:
    ret, img = cap.read() #讀取影片的每一幀
    if not ret:
        print("cannot receive frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #色彩轉換成黑白
    img_np = np.array(gray,'uint8') #轉換成指定編碼的numpy陣列
    face = detector.detectMultiScale(gray) #擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w]) #紀錄自己人臉的位置和大小內像素的數值
        ids.append(3) #紀錄自己人臉對應的id，只能是整數，都是1表示川普的id
    cv2.imshow('oxxostudio', img) #顯示攝影機畫面
    if cv2.waitKey(10000) == ord('q'): #每一毫秒更新一次，直到按下q結束
        break

print('training...') #提示開始訓練
recog.train(faces,np.array(ids)) #開始訓練
recog.save('face.yml') #訓練完成儲存為face.yml
print('ok!')



"""
img = cv2.imread('/Users/xupeihan/PycharmProjects/LBPH20230522/pic_TEW/01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 將圖片轉成灰階
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # 載入人臉模型
faces = face_cascade.detectMultiScale(gray)    # 偵測人臉
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 利用 for 迴圈，抓取每個人臉屬性，繪製方框
cv2.imshow('oxxostudio', img)
cv2.waitKey(0) # 按下任意鍵停止
cv2.destroyAllWindows()


img = cv2.imread('/Users/xupeihan/PycharmProjects/LBPH20230522/pic_trump/1.jpg')
cv2.imshow('dek',img)
cv2.waitKey(0)
"""