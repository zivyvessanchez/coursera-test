import numpy as np
import cv2

def main(filename):
    cap = cv2.VideoCapture(filename)

    # Take first frame of the video
    ret, frame = cap.read()

    # Setup initial location of window
    r,h,c,w = 250,90,400,125    # Hardcoded. Row, Height, Column, Width
    track_window = (c,r,w,h)

    # Set up the ROI (Region of Interest) for tracking
    roi = frame[r:r+h, c:c+w]
    # Hue-Saturation-Value ROI
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Discard low light values
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
    # Histogram using HSV ROI and Mask
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    # Normalize values of histogram
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # Set up termination criteria: either 10 iterations or if ROI moves only by 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while(1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # Appy meanshift to get new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw it an image
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
            cv2.imshow('img2',img2)

            k = cv2.waitKey(60) & 0xff
            if k == ord('q'):
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
