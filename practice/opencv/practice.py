import numpy as np
import cv2
import win32Util
from mss import mss

def getWindowRect(wildcard):
    w = win32Util.Window(wildcard)
    rect = w.getWindowRect()
    return rect

def getScreen(rect):
    sct = mss()
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    monitor = {'top': y, 'left': x, 'width': w, 'height': h}
    grab = sct.grab(monitor)
    return grab

def getVSTM(frameArray):
    out = None
    weights = []
    '''for i in range(len(frameArray)):
        weights.append(i+1)
    weights[:] = [x / sum(weights) for x in weights]
    '''
    
    for i in range(len(frameArray)):
        if i == 0:
            out = frameArray[i]
        else:
            #out = cv2.addWeighted(out,0.8,frameArray[i],0.2,0)
            out = cv2.bitwise_and(out,frameArray[i])
    
    return out

def readGauss(source=0):
    cap = cv2.VideoCapture(source)

    i = 3
    i_itr = 2
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame2 = cv2.GaussianBlur(frame,(i,i),0)
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = max(3, (i+i_itr) % 100)
        if i_itr > 0 and (i+i_itr) % 100 < i:
            i_itr *= -1
        elif i_itr < 0 and (i+i_itr) < 3:
            i_itr *= -1

    # When done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def read(source=0):
    cap = cv2.VideoCapture(source)
    frames = []
    max_frames = 20
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if len(frames) >= max_frames:
            frames.pop(0)
        frames.append(frame)
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main(wildcard):
    window_name = '2hu'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    sct = mss()
    rect = getWindowRect(wildcard)
    while(True):
        
        grab = getScreen(rect)
        #grab = sct.grab(sct.monitors[0])
        img = np.array(grab)
        cv2.imshow(window_name,img)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('.*Touhou.*')
