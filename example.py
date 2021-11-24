import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'depthai_hand_tracker'))
from HandTrackerEdge import HandTracker
from HandTrackerRenderer import HandTrackerRenderer
import cv2

tracker = HandTracker()
renderer = HandTrackerRenderer(tracker)
hands = []

while True:
    # Run hand tracker on next frame
    frame, hands, bag = tracker.next_frame()
    if frame is None: break
    # Render frame
    frame = renderer.draw(frame, hands, bag)
    cv2.imshow("HandTracker", frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()