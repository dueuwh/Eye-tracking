
"""
    Kwangwoon university Bio Computing & Machine learning lab present
    developers:
        - Nam Yu Sang
"""


import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import EyeExtractionConvexHull, display_frames, Pupil
import multiprocessing

        
def EyeTracking_pipeline(output_queue=None, duration=10, gamma=1.0):

    save_dir = "C:/Users/U/Desktop/BCML/IITP/eye/test_set/raw/"

    cap = cv2.VideoCapture(0)
    
    eech = EyeExtractionConvexHull()
    
    old_ldmks = np.zeros((468, 5), dtype=np.float32)
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    channel = 3
    
    start_time = time.time()

    pupil_detector = Pupil(gamma=gamma)
    
    not_detected = True

    # frame_count = 10000000
    
    while time.time() - start_time < duration:
        
        ret, frame = cap.read()
        
        if not ret:
            print("\nwebcam status is strange...\nMaybe the previous process is still running.")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5
        
        ldmks = np.zeros((468, 2), dtype=np.float32)
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [l for l in face_landmarks.landmark]
                for idx in range(len(landmarks)):
                    landmark = landmarks[idx]
                    if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                            or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                        coords = mp_drawing._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, width, height)
                    
                    if coords:
                        ldmks[idx, 0] = coords[1]
                        ldmks[idx, 1] = coords[0]
                
                left_eye, right_eye, box_left, box_right = eech.extract_eye(frame, ldmks)
                old_ldmks = ldmks
            
            else:
                left_eye, right_eye, box_left, box_right = eech.extract_eye(frame, old_ldmks)
        try:
            right_eye = cv2.resize(right_eye, (0, 0),  fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
            left_eye = cv2.resize(left_eye, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
        except:
            print("continue")
            continue
        left_coord, image_left = pupil_detector.detect(left_eye)
        right_coord, image_right = pupil_detector.detect(right_eye)

        # cv2.imwrite(save_dir+"left_eye/" + str(frame_count)+".jpg", cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(save_dir+"right_eye/" + str(frame_count)+".jpg", cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(save_dir + "frames/" + str(frame_count) + ".jpg", frame)

        # frame_count += 1
        
        if output_queue is not None:
            try:
                output_queue.put(right_eye)
                output_queue.put(left_eye)
                output_queue.put(image_left)
                output_queue.put(image_right)
                detected = True
            except:
                empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
                for _ in range(4):
                    output_queue.put(empty_image)
                detected = False
            output_queue.put(frame)

    if output_queue is not None:
        for _ in range(5):
            output_queue.put(None)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visual = True
    duration = 10  # seconds
    gamma = 0.1
    
    if visual:
        output_queue = multiprocessing.Queue()
        eye_tracking = multiprocessing.Process(target=EyeTracking_pipeline, 
                                               args=(output_queue, duration, gamma))
        display_process = multiprocessing.Process(target=display_frames, args=(output_queue,))
    
        eye_tracking.start()
        display_process.start()
    
        eye_tracking.join()
        display_process.join()

    else:
        EyeTracking_pipeline(duration=duration, gamma=gamma)

    print("End eye tracking")
# =============================================================================
#     current error
#     
#     Process Process-1:
#     Traceback (most recent call last):
#       File "C:\Users\U\anaconda3\envs\gen\Lib\multiprocessing\process.py", line 314, in _bootstrap
#         self.run()
#       File "C:\Users\U\anaconda3\envs\gen\Lib\multiprocessing\process.py", line 108, in run
#         self._target(*self._args, **self._kwargs)
#       File "C:\Users\U\Desktop\BCML\IITP\eye\GazeTracking_new\eye_tracking.py", line 80, in EyeTracking_pipline
#         output_queue.put(cv2.resize(right_eye, (0,0),  fx=5, fy=5, interpolation=cv2.INTER_CUBIC))
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     cv2.error: OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4152: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
#     
# =============================================================================
    