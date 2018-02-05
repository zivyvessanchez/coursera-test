import numpy as np
import cv2
import random

def read(source=0):
    cap = cv2.VideoCapture(source)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Frame operations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def readFile(filename):
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def write(filename):
    cap = cv2.VideoCapture(0)

    # Define codec and VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, 8.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #frame = cv2.flip(frame,0)
            # Write frame
            out.write(frame)
            # Show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    read()
