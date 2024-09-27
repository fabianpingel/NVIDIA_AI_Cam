import numpy as np
import cv2
#import av
import torch
import torchvision

if __name__ == '__main__':
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'

        while cap.isOpened():

            ret, frame = cap.read()

            frame = cv2.flip(frame,1)
            cv2.imshow('mycreame',frame)
            if ( cv2.waitKey(30)  == 27 ):
                break
        cv2.destroyAllWindows()