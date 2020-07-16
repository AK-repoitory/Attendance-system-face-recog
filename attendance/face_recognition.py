import cv2
import numpy as np
import face_recognition
import os
def face_recogn():
	path = '/Users/DELL/Desktop/attendance/train'
	images = []
	classNames = []
	myList = os.listdir(path)
	print(myList)
	count=[]
	i=0
	no=0
	for cl in myList:
	    curImg = cv2.imread(f'{path}/{cl}')
	    images.append(curImg)
	    classNames.append(os.path.splitext(cl)[0])
	print(classNames)

	def findEncodings(images):
	    encodeList = []
	    for img in images:
	        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	        encode = face_recognition.face_encodings(img)[0]
	        encodeList.append(encode)
	    return encodeList


	encodeListKnown = findEncodings(images)
	cap = cv2.VideoCapture(0)
	 
	while True:
	    success, img = cap.read()
	    #img = captureScreen()
	    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
	    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
	 
	    facesCurFrame = face_recognition.face_locations(imgS)
	    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
	 
	    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
	        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
	        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
	        #print(faceDis)
	        matchIndex = np.argmin(faceDis)
	 
	        if matches[matchIndex]:
	            name = classNames[matchIndex].	
	            # print(name)
	            count.append(name)
	            i=i+1
	    if i == 10:
	    	break
	    no = no +1
	    if no == 100:
	    	break
	# print(count)
	try:
		m = max(count)
		return(m)
	except:
		return None


def face_Detection(test_img):
    gray_img =cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade =cv2.CascadeClassifier('/Users/DELL/Desktop/attendance/cascades/data/haarcascade_frontalface_default.xml')
    face = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=5)
    return face, gray_img

def entry(name):
    cap = cv2.VideoCapture(0)
    count = 0
    i = 0
    no = 0
    f = 0

    os.chdir('/Users/DELL/Desktop/attendance/train')
    # os.chdir('/Users/DELL/Desktop/attendance/lol')
    # os.mkdir(name)
    os.chdir('/Users/DELL/Desktop/attendance')

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        face_detected, gray_img = face_Detection(test_img)
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        face_haar_cascade = cv2.CascadeClassifier(
            '/Users/DELL/Desktop/attendance/cascades/data/haarcascade_frontalface_default.xml')
        face = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)


        for face in face_detected:
            (x, y, w, h) = face

            if face.all() != 0:
                f = 1
                no = no + 1
                print(no)
            else:
                f = 0

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('testing', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break
        if f == 1:
            cv2.imwrite("/Users/DELL/Desktop/attendance/train"+ "/%s.jpg" % name, test_img)
            count += 1
            break
            f = 0
        if no == 2:
            break

    cap.release()
    cv2.destroyAllWindows()
    # os.chdir('/Users/DELL/Desktop/university_portal')

# print(face_recogn())
# entry("Name of person")