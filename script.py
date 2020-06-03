# Import OpenCV library
import cv2

# Load a cascade file for detecting faces
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# open capture device 
cap = cv2.VideoCapture(0) # internal camera

#infinite capturing of video
while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look for faces in the image using the loaded cascade file
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
            # Create rectangle around faces
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

    # Create the resizeable window
    cv2.namedWindow('FaceDetection', cv2.WINDOW_NORMAL)

    # Display the image
    cv2.imshow('FaceDetection', image)


    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
