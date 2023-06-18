import cv2

img  = cv2.imread("walking.avi")
# Create our body classifier
body_classifier =  cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    faces = face_cascade.detectMultiScale(gray)
    print(faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+y,y+h),(255,0,0),2)
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite("face.jpg",roi_color)
    cv2.imshow('img',img)
    cv2.waitkey(0)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
