# Module 1 Facial Recognition
# Import Libraries
import cv2
#Load Cascades
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
#Function for detections
def detect(greyimage, image):
    faces = face.detectMultiScale(greyimage, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        faceRegionGrey = greyimage[y:y+h, x:x+w]
        faceRegion = image[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(faceRegionGrey, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(faceRegion, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    return image

#turn on webcam, apply function, and then return video
video = cv2.VideoCapture(0)
while True:
    _, image = video.read()
    greyimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas = detect(greyimage, image)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

