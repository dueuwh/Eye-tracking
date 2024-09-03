  
"""
    Kwangwoon university Bio Computing & Machine learning lab present
    developers:
        - Nam Yu Sang
"""

from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
import tkinter as tk
import queue


def display_frames(fq):
    window = tk.Tk()
    window.title("Eye tracking")
    
    canvas1 = tk.Canvas(window, width=640, height=480)
    canvas1.grid(row=0, column=0)
    canvas2 = tk.Canvas(window, width=640, height=480)
    canvas2.grid(row=0, column=1)
    canvas3 = tk.Canvas(window, width=640, height=480)
    canvas3.grid(row=0, column=2)
    
    canvas4 = tk.Canvas(window, width=640, height=480)
    canvas4.grid(row=1, column=0)
    canvas5 = tk.Canvas(window, width=640, height=480)
    canvas5.grid(row=1, column=1)
    
    def update_canvas():
        try:
            frame1 = fq.get(timeout=15)
            frame2 = fq.get(timeout=15)
            frame5 = fq.get(timeout=15)
            
            frame4 = fq.get(timeout=15)
            frame3 = fq.get(timeout=15)

            if frame1 is None or frame2 is None or frame3 is None:
                window.quit()
                return
            # print(f"frame1.shape: {frame1.shape}\nframe2.shape: {frame2.shape}\nframe3.shape: {frame3.shape}")
            img1 = ImageTk.PhotoImage(image=Image.fromarray(frame1))
            img2 = ImageTk.PhotoImage(image=Image.fromarray(frame2))
            img3 = ImageTk.PhotoImage(image=Image.fromarray(frame3))

            img4 = ImageTk.PhotoImage(image=Image.fromarray(frame4))
            img5 = ImageTk.PhotoImage(image=Image.fromarray(frame5))

            canvas1.create_image(0, 0, anchor=tk.NW, image=img1)
            canvas1.image = img1
            canvas2.create_image(0, 0, anchor=tk.NW, image=img2)
            canvas2.image = img2
            canvas3.create_image(0, 0, anchor=tk.NW, image=img3)
            canvas3.image = img3
            
            canvas4.create_image(0, 0, anchor=tk.NW, image=img4)
            canvas4.image = img4
            canvas5.create_image(0, 0, anchor=tk.NW, image=img5)
            canvas5.image = img5

            window.after(10, update_canvas)
        except queue.Empty:
            print("Frame queue is empty. Exiting...")
            window.quit()

    window.after(10, update_canvas)
    window.mainloop()


class RemoveSpecularReflection:
    def __init__(self,):
        self.initialization = 1


class Pupil:
    def __init__(self, gamma):
        self.gamma = gamma
        self.threshold = 80
        self.scaling_x = 3
        self.scaling_slope = -30

    def image_processing_gaze_tracking(self, eye_frame):

        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, self.threshold, 255, cv2.THRESH_BINARY)[1]
        # new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        return new_frame

    def threshold_calibration(self, input_frame):
        self.threshold = 29
        for i in range(51):
            self.threshold += 1
            
            sample = self.image_processing_gaze_tracking(input_frame)
            sample = sample.astype(np.uint8)
            
            contours, _ = cv2.findContours(sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            print(f"contours: {contours}")
            

    def circle_approximation(self, ):
        pass
        
        

    def detect(self, input_frame):

        output_frame = self.image_processing_gaze_tracking(input_frame)
        self.threshold_calibration(input_frame)
        # output_frame = self.scale_transform(output_frame)
        return 1, output_frame
    
        # contours, _ = cv2.findContours(gamma_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        # contours = sorted(contours, key=cv2.contourArea)
        #
        # try:
        #     moments = cv2.moments(contours[-2])
        #     x = int(moments['m10'] / moments['m00'])
        #     y = int(moments['m01'] / moments['m00'])
        #     return [x, y], gamma_frame
        # except (IndexError, ZeroDivisionError):
        #     return [None, None], None, None


def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class EyeExtractionConvexHull:
    """
        This class performs eye extration only.
    """
    
    def __init__(self, device='CPU'):
        self.device = device
        self.eye_l = [465, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
        self.eye_r = [245, 189, 221, 222, 223, 224, 225, 113, 226, 31, 228, 229, 230, 231, 232, 233]
        self.left = 1
        self.right = 2
        self.old_l = []
        self.old_r = []


    def __creat_mask(self, image, ldmks_input, eye, which_eye):
        ldmks = np.array([ldmks_input[i] for i in eye])
        ldmks = ldmks[ldmks[:, 0] >= 0][:, :2]
        
        try:
            hull = ConvexHull(ldmks)
            if which_eye == self.left:
                self.old_l = hull
            else:
                self.old_r = hull
        except:
            if which_eye == self.left:
                hull = self.old_l
            else:
                hull = self.old_r
        
        verts = [(ldmks[v, 0], ldmks[v, 1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask, axis=0).T
        mask_inv = (np.ones_like(mask) - mask) * 255
        return mask, mask_inv


    def __crop_image(self, image, mask, mask_inv):
        eye = image * mask
        
        if self.device == 'CPU':
            rmin, rmax, cmin, cmax = bbox2_CPU(eye)
        
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            eye = eye[int(rmin):int(rmax), int(cmin):int(cmax)]
            mask_inv = mask_inv[int(rmin):int(rmax), int(cmin):int(cmax)]
        mask_inv = np.repeat(mask_inv, repeats=3, axis=2)
        eye = eye + mask_inv
        return eye, rmin, cmin


    def extract_eye(self, image, ldmks):
        
        mask_left, left_inv = self.__creat_mask(image, ldmks, self.eye_l, self.left)
        mask_right, right_inv = self.__creat_mask(image, ldmks, self.eye_r, self.right)
        
        left_eye, rmin_left, cmin_left = self.__crop_image(image, mask_left, left_inv)
        right_eye, rmin_right, cmin_right = self.__crop_image(image, mask_right, right_inv)
        
        if sum(left_eye.shape[:2]) <= 4:
            left_eye = None
        if sum(right_eye.shape[:2]) <= 2:
            right_eye = None
            
        return left_eye, right_eye, [rmin_left, cmin_left], [rmin_right, cmin_right]
    

if __name__ == "__main__":
    image = cv2.imread('C:/Users/U/Downloads/eyes_1.png')
    # image = cv2.resize(image, dsize=(0,0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Original Image', image)
    cv2.imshow('Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()