import cv2 as OpenCV
import numpy as np
from laneDetection import canny, region_of_interest, houghLines, average_slope_intercept, display_lines, addWeighted

# Load the cascade classifiers
car_cascade = OpenCV.CascadeClassifier('Cascades/cars.xml')
stop = OpenCV.CascadeClassifier('Cascades/cascade_stop_3_15.xml')

# Open video capture
cap = OpenCV.VideoCapture("Videos/lanes.mp4")
detection = {'autoroute': {'lanesv1': True, 'lanesv2': True}, 'voiture': True, 'stop': True, 'fastmode': False}

# Pre-process the video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Reduce the image size for faster processing (optional)
resize_width = int(frame_width / 2)
resize_height = int(frame_height / 2)

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        # Resize the frame (optional)
        if detection['fastmode']:
            frame = OpenCV.resize(frame, (resize_width, resize_height))

        gray = OpenCV.cvtColor(frame, OpenCV.COLOR_BGR2GRAY)
        gray = OpenCV.blur(gray, (8, 8))
        edges = OpenCV.Canny(gray, 50, 200)
        if detection['autoroute']['lanesv2']:
            canny_image = canny(frame)
            cropped_canny = region_of_interest(canny_image)
            # cv2.imshow("cropped_canny",cropped_canny)

            lines = houghLines(cropped_canny)
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            combo_image = addWeighted(frame, line_image)
            frame = combo_image

        elif not detection['autoroute']['lanesv2'] and detection['autoroute']['lanesv1']:
            lines = OpenCV.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    OpenCV.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if detection['voiture']:
            if detection['fastmode']:
                cars, _, car_confidences = car_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
            else:
                cars, _, car_confidences = car_cascade.detectMultiScale3(gray, 1.5, 6, outputRejectLevels=True)
            cars_filtered = [(x, y, w, h, conf) for (x, y, w, h), conf in zip(cars, car_confidences) if conf > 1]
            for (x, y, w, h, conf) in cars_filtered:
                text = 'Vehicule ' + str(round(conf, 2))
                if not detection['fastmode']:
                    (tw, th), _ = OpenCV.getTextSize(text, OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    OpenCV.putText(frame, text, (x, y - th - 5), OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                OpenCV.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if detection['stop']:
            stop_signs, _, stop_confidences = stop.detectMultiScale3(gray, 1.5, 6, outputRejectLevels=True)
            stop_signs_filtered = [(x, y, w, h, conf) for (x, y, w, h), conf in zip(stop_signs, stop_confidences) if conf > 1.3]
            for (x, y, w, h, conf) in stop_signs_filtered:
                text = 'STOP ' + str(round(conf, 2))
                if not detection['fastmode']:
                    (tw, th), _ = OpenCV.getTextSize(text, OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    OpenCV.putText(frame, text, (x, y - th - 5), OpenCV.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                OpenCV.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        OpenCV.imshow('Detection', frame)

        if OpenCV.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
OpenCV.destroyAllWindows()
