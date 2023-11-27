import cv2
import numpy as np
import math

cap = cv2.VideoCapture("Files/Videos/10.mp4")

X_pos = []
Y_pos = []

X_plot = [item for item in range(0, 1200)]

def get_centers(mask, minarea=20):
    center_points = []
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minarea:
            arc = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*arc, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + (w//2), y + (h//2)
            center_points.append({"area":area, "center":[cx,cy]})
    
    center_points = sorted(center_points, key=lambda x:x["area"], reverse=True)
    return center_points

while True:
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_range = (6, 174, 78)
        upper_range = (10, 203, 99)

        mask = cv2.inRange(frameHSV, lower_range, upper_range)

        ball_detection = cv2.bitwise_and(frame, frame, mask=mask)

        center_points = get_centers(mask, minarea=20)

        if center_points:
            X_pos.append(center_points[0]["center"][0])
            Y_pos.append(center_points[0]["center"][1])

        if X_pos:
            A, B, C = np.polyfit(X_pos, Y_pos, 2)
            for i, (posX, posY) in enumerate(zip(X_pos, Y_pos)):
                pos=(posX, posY)
                cv2.circle(frame_copy, pos, 8, (0,255,0), cv2.FILLED)
            for x in X_plot:
                y = int(A*x**2+B*x+C)
                cv2.circle(frame_copy, (x,y), 2, (0,0,0), cv2.FILLED)
            if len(X_pos) < 13:
                a = A
                b = B
                c = C - 180
                x = int((-b + math.sqrt(b ** 2 - (4*a*c))) / (2 * a))
                prediction = 970 < x < 1050

            if prediction:
                IN = "BASKET"
                pos = (50, 150)
                offset = 10
                (w, h), _ = cv2.getTextSize(IN, cv2.FONT_HERSHEY_PLAIN, 3, thickness=3)
                x1, y1, x2, y2 = pos[0] - offset, pos[1] + offset, pos[0] + w + offset, pos[1] - h - offset
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), -1)
                cv2.putText(frame_copy, IN, (pos[0], pos[1]), cv2.FONT_HERSHEY_PLAIN, 3, [255, 255, 255], thickness=3)
            else:
                OUT = "NO BASKET"
                pos = (50, 150)
                offset = 10
                (w, h), _ = cv2.getTextSize(OUT, cv2.FONT_HERSHEY_PLAIN, 3, thickness=3)
                x1, y1, x2, y2 = pos[0] - offset, pos[1] + offset, pos[0] + w + offset, pos[1] - h - offset
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), -1)
                cv2.putText(frame_copy, OUT, (pos[0], pos[1]), cv2.FONT_HERSHEY_PLAIN, 3, [255, 255, 255], thickness=3)


        frame_copy = cv2.resize(frame_copy, (0,0), None, 0.6, 0.61)
        cv2.imshow("Video", frame_copy)
        if cv2.waitKey(500) & 0xFF == ord('1'):
            break

    else:
        break