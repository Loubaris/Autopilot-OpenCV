import cv2 as OpenCV
import numpy as np 


car_cascade = OpenCV.CascadeClassifier('Cascades/cars.xml')
stop = OpenCV.CascadeClassifier('Cascades/cascade_stop_3_15.xml')

cap = OpenCV.VideoCapture("Videos/Canada.mp4")
autoroute = True
detection = {'voiture': True, 'stop': True}

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = OpenCV.cvtColor(frame, OpenCV.COLOR_BGR2GRAY)
        gray = OpenCV.blur(gray,(8,8))

        edges = OpenCV.Canny(gray, 50, 200)

        # Détection des lignes de la route avec la transformée de Hough
        if autoroute:
            lines = OpenCV.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    OpenCV.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if detection['voiture']:
            cars, _, car_confidences = car_cascade.detectMultiScale3(gray, 1.2, 6, outputRejectLevels=True)
            cars_filtered = []
            for i, (x, y, w, h) in enumerate(cars):
                if car_confidences[i] > 1:
                    cars_filtered.append((x, y, w, h, car_confidences[i]))
            for (x, y, w, h, conf) in cars_filtered:
                text = 'Vehicule ' + str(round(conf, 2))
                (tw, th), _ = OpenCV.getTextSize(text, OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                OpenCV.putText(frame, text, (x, y-th-5), OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                OpenCV.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
         
        if detection['stop']:       
            stop_signs, _, stop_confidences = stop.detectMultiScale3(gray, 1.5, 6, outputRejectLevels=True)
            stop_signs_filtered = []
            for i, (x, y, w, h) in enumerate(stop_signs):
                if stop_confidences[i] > 1.3:
                    stop_signs_filtered.append((x, y, w, h, stop_confidences[i]))
            for (x, y, w, h, conf) in stop_signs_filtered:
                text = 'STOP ' + str(round(conf, 2))
                (tw, th), _ = OpenCV.getTextSize(text, OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                OpenCV.putText(frame, text, (x, y-th-5), OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                OpenCV.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    


        OpenCV.imshow('Detection',frame)
     
        if OpenCV.waitKey(25) & 0xFF == ord('q'):
            break
     
    else: 
        break

cap.release()
OpenCV.destroyAllWindows()