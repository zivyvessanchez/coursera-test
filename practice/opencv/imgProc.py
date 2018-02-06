import numpy as np
import cv2
from mss import mss

def rgbToHsv(rgbArray):
    assert len(rgbArray) == 3
    # Convert RGB to BGR
    rgb = np.uint8([[[rgbArray[2], rgbArray[1], rgbArray[0]]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

def colorTrack(colorArray=(-1,-1,-1)):
    assert len(colorArray) == 3
    
    cap = cv2.VideoCapture('vtest.mkv')
    i = 0
    while(True):
        i = (i+1) % 180
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define range of color in HSV
        lower = []
        upper = []
        if colorArray[0] == -1:
            lower = np.array([i,0,0])
            upper = np.array([i,255,255])
        else:
            rgbToHsvVal = rgbToHsv(colorArray)
            lower = np.array([max(0, rgbToHsvVal[0]-10),0,0])
            upper = np.array([min(179, rgbToHsvVal[0]+10),255,255])
        # Apply mask of colors to HSV image        
        mask = cv2.inRange(hsv, lower, upper)
        # Apply bitwise-AND mask to original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Show
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def screenRead():
    sct = mss()
    imgCount = 0
    firstFrame = True
    # Camshift vars
    r,h,c,w = 250,90,400,125
    track_window = (c,r,w,h)
    roi, bgr, hsv_roi, mask, roi_hist = None, None, None, None, None
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while(True):
        grab = sct.grab(sct.monitors[0])
        img = np.array(grab)
        edges = cv2.Canny(img, 100, 200)
        img2 = None

        if firstFrame:
            roi = edges[r:r+h, c:c+w]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            #hsv_roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hsv_roi = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(bgr,[pts],True, 255,2)
        
        cv2.imshow('image',img2)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(25) & 0xFF == ord('w'):
            cv2.imwrite('cap{}.jpg'.format(imgCount),frame)
            imgCount += 1

        firstFrame = False
    
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture('vtest.mkv')
    imgCount = 0

    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(25) & 0xFF == ord('w'):
            cv2.imwrite('cap{}.jpg'.format(imgCount),frame)
            imgCount += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    colorTrack()
