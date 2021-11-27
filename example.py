import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'depthai_hand_tracker'))
from HandTrackerEdge import HandTracker
from HandTrackerRenderer import HandTrackerRenderer
import cv2
import handmotion

tracker = HandTracker()
renderer = HandTrackerRenderer(tracker)
hands = []
class_names = ['BYE', 'NON', 'CIRCLE', 'POINTCIRCLE', 'PAPER']
modelfile = 'handmotionXY.tflite'
handmotions = handmotion.HandMotion(class_names, modelfile)

while True:
    # Run hand tracker on next frame
    frame, hands, bag = tracker.next_frame()
    if frame is None: break
    # Render frame
    frame = renderer.draw(frame, hands, bag)
    motionsclass = handmotions.run(hands)
    cv2.putText(frame, motionsclass, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    cv2.imshow("HandTracker", frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()