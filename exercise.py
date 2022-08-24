from ast import While
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout,QPushButton
from PyQt5.QtGui import QPixmap
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def convert_to_point(landmark,image_width,image_height):
    return tuple(np.multiply(landmark, [image_width, image_height]).astype(int))


def render_extremities(image,color,point_1,point_2):
    cv2.line(image, point_1,point_2, color, thickness=10, lineType=8)


def render_point(image,color1,color2,point):
    cv2.circle(image,point,10,color1,cv2.FILLED)
    cv2.circle(image,point,12,color2,2)


def correct_toe_left(foot,knee,image_height,image_width):
    correct = None
    a=np.multiply(foot, [image_width, image_height]).astype(int)
    b=np.multiply(knee, [image_width, image_height]).astype(int)
    if a[0]>b[0] :
        correct ="BAD"
    else :
        correct="GOOD"

    return correct


def correct_toe_right(foot,knee,image_height,image_width):
    correct = None
    a=np.multiply(foot, [image_width, image_height]).astype(int)
    b=np.multiply(knee, [image_width, image_height]).astype(int)
    if a[0]>b[0] :
        correct ="GOOD"
    else :
        correct="BAD"

    return correct


def true_position_down_right(image,color,point): 
    one=list(point)
                
    one[0]+=200
    two=list(point)
    two[0]+=200
    two[1]+=200
    cv2.line(image, point,one, color, thickness=4, lineType=8)
    cv2.line(image, one,two, color, thickness=4, lineType=8)


def true_position_up_right(image,color,point):
    one=list(point)            
    one[1]+=200

    two=list(point)

    two[0]-=200
    two[1]+=200
    cv2.line(image, point,one, color, thickness=4, lineType=8)
    cv2.line(image, one,two, color, thickness=4, lineType=8)


def true_position_down_left(image,color,point) :
    one=list(point)            
    one[1]+=200

    two=list(point)

    two[0]+=200
    two[1]+=200
    cv2.line(image, point,one, color, thickness=4, lineType=8)
    cv2.line(image, one,two, color, thickness=4, lineType=8)
       

def true_position_up_left(image,color,point):
    one=list(point)            
    one[0]-=200

    two=list(point)

    two[0]-=200
    two[1]+=200
    cv2.line(image, point,one, color, thickness=4, lineType=8)
    cv2.line(image, one,two, color, thickness=4, lineType=8)


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0,255,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE=(255,0,0)
ORANGE=(0,165,255)


class Good_Morning(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
        upper_left = (image_width  // 10 , image_height  // 10)
        bottom_right = (image_width * 9 // 10 , image_height * 9 // 10 )
        stage = "Start"
        stage_info=(image_width  // 4 , image_height  // 10)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, image = cap.read()
                image = cv2.flip(image,1)
                if ret:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    try:
                        landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
                        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]                    
                        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                        left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                        left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                        left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                        l_knee=convert_to_point(knee_left,image_width,image_height)    
                        r_knee=convert_to_point(knee_right,image_width,image_height)
                
                        l_ankle=convert_to_point(ankle_left,image_width,image_height)
                        r_ankle=convert_to_point(ankle_right,image_width,image_height)

                        l_hip=convert_to_point(hip_left,image_width,image_height)
                        r_hip=convert_to_point(hip_right,image_width,image_height)                

                        l_shoulder=convert_to_point(left_shoulder,image_width,image_height)
                        r_shoulder=convert_to_point(right_shoulder,image_width,image_height)

                        l_elbow=convert_to_point(left_elbow,image_width,image_height)
                        r_elbow=convert_to_point(right_elbow,image_width,image_height)

                        l_wrist=convert_to_point(left_wrist,image_width,image_height)
                        r_wrist=convert_to_point(right_wrist,image_width,image_height)


                        angle_elbow_left=round(calculate_angle(left_wrist,left_elbow,left_shoulder),2)  
                        angle_elbow_right=round(calculate_angle(right_wrist,right_elbow, right_shoulder),2)  
                        
                        angle_back_right=round(calculate_angle(right_shoulder,hip_right,knee_right),2)
                        
                        angle_leg_left=round(calculate_angle(hip_left,knee_left,ankle_left),2)
                        angle_leg_right=round(calculate_angle(hip_right,knee_right,ankle_right),2)


                        render_point(image,BLACK,RED,l_ankle)
                        render_point(image,BLACK,RED,l_knee)
                        render_point(image,BLACK,RED,l_hip)
                        render_point(image,BLACK,RED,l_shoulder)
                        render_point(image,BLACK,RED,l_elbow)
                        render_point(image,BLACK,RED,l_wrist)
                        
                        render_point(image,BLACK,RED,r_ankle)
                        render_point(image,BLACK,RED,r_knee)
                        render_point(image,BLACK,RED,r_hip)
                        render_point(image,BLACK,RED,r_shoulder)
                        render_point(image,BLACK,RED,r_elbow)
                        render_point(image,BLACK,RED,r_wrist)

                        render_extremities(image,WHITE, l_ankle,l_knee)
                        render_extremities(image,WHITE, l_knee,l_hip)
                        render_extremities(image,WHITE, l_hip,l_shoulder)
                        render_extremities(image,WHITE, l_shoulder,l_elbow)
                        render_extremities(image,WHITE, l_elbow,l_wrist)

                        render_extremities(image,WHITE, r_ankle,r_knee)
                        render_extremities(image,WHITE, r_knee,r_hip)
                        render_extremities(image,WHITE, r_hip,r_shoulder)
                        render_extremities(image,WHITE, r_shoulder,r_elbow)
                        render_extremities(image,WHITE, r_elbow,r_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_elbow_left), 
                                tuple(np.multiply(left_elbow, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_elbow_right), 
                                tuple(np.multiply(right_elbow, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

                        cv2.putText(image, str(angle_back_right), 
                                tuple(np.multiply(hip_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)

                        cv2.putText(image, str(angle_leg_left), 
                                tuple(np.multiply(knee_left, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_leg_right), 
                                tuple(np.multiply(knee_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)


                        if angle_elbow_left>=40 and angle_elbow_right>=40 and angle_elbow_left<=80 and angle_elbow_right<=80:
                            render_extremities(image,GREEN,l_shoulder,l_elbow)
                            render_extremities(image,GREEN,l_elbow,l_wrist)
                            render_extremities(image,GREEN,r_shoulder,r_elbow)
                            render_extremities(image,GREEN,r_elbow,r_wrist)
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            stage ="KNEE"

                        if stage=="KNEE" and angle_leg_right>=120 and angle_leg_right<=160:
                            render_extremities(image,GREEN,r_hip,r_knee)
                            render_extremities(image,GREEN,r_knee,r_ankle)
                            render_extremities(image,GREEN,l_hip,l_knee)
                            render_extremities(image,GREEN,l_knee,l_ankle)
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            stage="DOWN"

                        if stage=="DOWN" and angle_back_right>=90 and angle_back_right<=130:
                            render_extremities(image,GREEN,r_shoulder,r_hip)
                            render_extremities(image,GREEN,l_shoulder,l_hip)
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            stage="UP"

                    except:
                        pass
                    
        # Stage data
                    cv2.putText(image, str(stage), 
                            stage_info, 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    self.change_pixmap_signal.emit(image)
        # shut down capture system
            cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Run_Good_Morning(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GOOD MORNING")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        
        self.back_btn = QPushButton(self)
        self.back_btn.setObjectName("back_btn")
        self.back_btn.setText("Back")

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.back_btn)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = Good_Morning()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.back_btn.clicked.connect(self.close_window)

    def close_window(self):
        self.thread.stop()
        self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
class Cabaret(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.stage = "START"

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
        upper_left = (image_width  // 10 , image_height  // 10)
        bottom_right = (image_width * 9 // 10 , image_height * 9 // 10 )
        counter = 0 
        side= None
        stage_info=(image_width  // 4 , image_height  // 10)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, image = cap.read()
                image = cv2.flip(image,1)
                if ret:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]                    
                        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                        left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                        left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                        l_knee=convert_to_point(knee_left,image_width,image_height)
                        r_knee=convert_to_point(knee_right,image_width,image_height)
                        
                        l_ankle=convert_to_point(ankle_left,image_width,image_height)
                        r_ankle=convert_to_point(ankle_right,image_width,image_height)

                        l_hip=convert_to_point(hip_left,image_width,image_height)
                        r_hip=convert_to_point(hip_right,image_width,image_height)                

                        l_shoulder=convert_to_point(left_shoulder,image_width,image_height)
                        r_shoulder=convert_to_point(right_shoulder,image_width,image_height)

                        l_elbow=convert_to_point(left_elbow,image_width,image_height)
                        r_elbow=convert_to_point(right_elbow,image_width,image_height)

                        l_wrist=convert_to_point(left_wrist,image_width,image_height)
                        r_wrist=convert_to_point(right_wrist,image_width,image_height)

                        
                        angle_shoulder_left=round(calculate_angle(left_elbow,left_shoulder,hip_left),2)  
                        angle_shoulder_right=round(calculate_angle(right_elbow,right_shoulder, hip_right),2)  
                        
                        angle_knee_right=round(calculate_angle(hip_right,knee_right,ankle_right),2)
                        angle_knee_left=round(calculate_angle(hip_left,knee_left,ankle_left),2)
                    
                        angle_leg_right=round(calculate_angle(right_shoulder,hip_right,ankle_right),2)
                        angle_leg_left=round(calculate_angle(left_shoulder,hip_left,ankle_left),2)


                        render_point(image,BLACK,RED,l_ankle)
                        render_point(image,BLACK,RED,l_knee)
                        render_point(image,BLACK,RED,l_hip)
                        render_point(image,BLACK,RED,l_shoulder)
                        render_point(image,BLACK,RED,l_elbow)
                        render_point(image,BLACK,RED,l_wrist)
                        
                        render_point(image,BLACK,RED,r_ankle)
                        render_point(image,BLACK,RED,r_knee)
                        render_point(image,BLACK,RED,r_hip)
                        render_point(image,BLACK,RED,r_shoulder)
                        render_point(image,BLACK,RED,r_elbow)
                        render_point(image,BLACK,RED,r_wrist)

                        render_extremities(image,WHITE, l_ankle,l_knee)
                        render_extremities(image,WHITE, l_knee,l_hip)
                        render_extremities(image,WHITE, l_hip,l_shoulder)
                        render_extremities(image,WHITE, l_shoulder,l_elbow)
                        render_extremities(image,WHITE, l_elbow,l_wrist)

                        render_extremities(image,WHITE, r_ankle,r_knee)
                        render_extremities(image,WHITE, r_knee,r_hip)
                        render_extremities(image,WHITE, r_hip,r_shoulder)
                        render_extremities(image,WHITE, r_shoulder,r_elbow)
                        render_extremities(image,WHITE, r_elbow,r_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_shoulder_left), 
                                tuple(np.multiply(left_shoulder, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_shoulder_left), 
                                tuple(np.multiply(right_shoulder, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)

                        cv2.putText(image, str(angle_knee_left), 
                                tuple(np.multiply(knee_left, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_knee_right), 
                                tuple(np.multiply(knee_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)

                        cv2.putText(image, str(angle_leg_right), 
                                tuple(np.multiply(ankle_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_leg_left), 
                                tuple(np.multiply(ankle_left, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)

                        if (counter % 2) == 0 :
                            side="LEFT"
                            render_extremities(image,YELLOW,l_shoulder,l_elbow)
                            render_extremities(image,YELLOW,l_elbow,l_wrist)
                            render_extremities(image,YELLOW,r_hip,r_knee)
                            render_extremities(image,YELLOW,r_knee,r_ankle)
                        else:
                            side="RIGHT"
                            render_extremities(image,YELLOW,r_shoulder,r_elbow)
                            render_extremities(image,YELLOW,r_elbow,r_wrist)
                            render_extremities(image,YELLOW,l_hip,l_knee)
                            render_extremities(image,YELLOW,l_knee,l_ankle)

                        if angle_shoulder_left>=80  and angle_shoulder_right>=80:
                            render_extremities(image,GREEN,l_shoulder,l_elbow)
                            render_extremities(image,GREEN,l_elbow,l_wrist)
                            render_extremities(image,GREEN,r_shoulder,r_elbow)
                            render_extremities(image,GREEN,r_elbow,r_wrist)
                            self.stage="KNEE"
                        else :
                            render_extremities(image,RED,l_shoulder,l_elbow)
                            render_extremities(image,RED,l_elbow,l_wrist)
                            render_extremities(image,RED,r_shoulder,r_elbow)
                            render_extremities(image,RED,r_elbow,r_wrist)
                        
                        if side=="LEFT":
                            if self.stage=="KNEE" and angle_knee_right>=80 and angle_knee_right<=100:
                                self.stage="LEG"
                                render_extremities(image,GREEN,r_hip,r_knee)

                            if self.stage=="LEG" and angle_leg_right>=90 and angle_leg_right<=110:
                                self.stage="STAND"
                                counter+=1
                                render_extremities(image,GREEN,r_knee,r_ankle)
                        
                        if side=="RIGHT":
                            if self.stage=="KNEE" and angle_knee_left>=80 and angle_knee_left<=100:
                                self.stage="LEG"
                                render_extremities(image,GREEN,r_hip,r_knee)

                            if self.stage=="LEG" and angle_leg_left>=90 and angle_leg_left<=110:
                                self.stage="STAND"
                                counter+=1
                                render_extremities(image,GREEN,r_knee,r_ankle)
                        
                    except:
                        pass

                # Stage data
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    cv2.putText(image, 'FEET', (125,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(side), 
                                (100,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                                
                    cv2.putText(image, str(self.stage), 
                                stage_info, 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)  
                        

                    self.change_pixmap_signal.emit(image)
        # shut down capture system
            cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Run_Cabaret(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CABARET")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        
        self.back_btn = QPushButton(self)
        self.back_btn.setObjectName("back_btn")
        self.back_btn.setText("Back")
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.back_btn)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = Cabaret()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.back_btn.clicked.connect(self.close_window)

    def close_window(self):
        self.thread.stop()
        self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class March(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.stage = "START"

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
        upper_left = (image_width  // 10 , image_height  // 10)
        bottom_right = (image_width * 9 // 10 , image_height * 9 // 10 )
        counter = 0 
        side= None
        stage_info=(image_width  // 4 , image_height  // 10)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, image = cap.read()
                image = cv2.flip(image,1)
                if ret:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]                    
                        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                        left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                        left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                        l_knee=convert_to_point(knee_left,image_width,image_height)
                        r_knee=convert_to_point(knee_right,image_width,image_height)
                        
                        l_ankle=convert_to_point(ankle_left,image_width,image_height)
                        r_ankle=convert_to_point(ankle_right,image_width,image_height)

                        l_hip=convert_to_point(hip_left,image_width,image_height)
                        r_hip=convert_to_point(hip_right,image_width,image_height)                

                        l_shoulder=convert_to_point(left_shoulder,image_width,image_height)
                        r_shoulder=convert_to_point(right_shoulder,image_width,image_height)

                        l_elbow=convert_to_point(left_elbow,image_width,image_height)
                        r_elbow=convert_to_point(right_elbow,image_width,image_height)

                        l_wrist=convert_to_point(left_wrist,image_width,image_height)
                        r_wrist=convert_to_point(right_wrist,image_width,image_height)
                

                        angle_arm_left=round(calculate_angle(left_shoulder,left_elbow,left_wrist),2)  
                        angle_arm_right=round(calculate_angle(right_shoulder,right_elbow, right_wrist),2)  
                                                
                        angle_leg_left=round(calculate_angle(hip_left,knee_left,ankle_left),2)
                        angle_leg_right=round(calculate_angle(hip_right,knee_right,ankle_right),2)
                    

                        render_point(image,BLACK,RED,l_ankle)
                        render_point(image,BLACK,RED,l_knee)
                        render_point(image,BLACK,RED,l_hip)
                        render_point(image,BLACK,RED,l_shoulder)
                        render_point(image,BLACK,RED,l_elbow)
                        render_point(image,BLACK,RED,l_wrist)
                        
                        render_point(image,BLACK,RED,r_ankle)
                        render_point(image,BLACK,RED,r_knee)
                        render_point(image,BLACK,RED,r_hip)
                        render_point(image,BLACK,RED,r_shoulder)
                        render_point(image,BLACK,RED,r_elbow)
                        render_point(image,BLACK,RED,r_wrist)

                        render_extremities(image,WHITE, l_ankle,l_knee)
                        render_extremities(image,WHITE, l_knee,l_hip)
                        render_extremities(image,WHITE, l_hip,l_shoulder)
                        render_extremities(image,WHITE, l_shoulder,l_elbow)
                        render_extremities(image,WHITE, l_elbow,l_wrist)

                        render_extremities(image,WHITE, r_ankle,r_knee)
                        render_extremities(image,WHITE, r_knee,r_hip)
                        render_extremities(image,WHITE, r_hip,r_shoulder)
                        render_extremities(image,WHITE, r_shoulder,r_elbow)
                        render_extremities(image,WHITE, r_elbow,r_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_arm_right), 
                                tuple(np.multiply(right_elbow, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_arm_left), 
                                tuple(np.multiply(left_elbow, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

                        cv2.putText(image, str(angle_leg_left), 
                                tuple(np.multiply(knee_left, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_leg_right), 
                                tuple(np.multiply(knee_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

                        if (counter % 2) == 0:
                            side = "LEFT"
                            self.stage=="UP" 
                            render_extremities(image,YELLOW,l_shoulder,l_elbow)
                            render_extremities(image,YELLOW,l_elbow,l_wrist)
                            render_extremities(image,YELLOW,r_hip,r_knee)
                            render_extremities(image,YELLOW,r_knee,r_ankle)
                        else:
                            side="RIGHT"
                            self.stage=="UP"
                            render_extremities(image,YELLOW,r_shoulder,r_elbow)
                            render_extremities(image,YELLOW,r_elbow,r_wrist)
                            render_extremities(image,YELLOW,l_hip,l_knee)
                            render_extremities(image,YELLOW,l_knee,l_ankle)
                        if side=="LEFT" and angle_arm_left>=20 and angle_arm_left<=80 and angle_leg_right>=70 and angle_leg_right <=100:
                        #if side=="LEFT" and angle_leg_right>=70 and angle_leg_right <=95:
                        #render_extremities(image,GREEN,l_shoulder,l_elbow)
                        #render_extremities(image,GREEN,l_elbow,l_wrist)
                        #render_extremities(image,GREEN,r_hip,r_knee)
                        #render_extremities(image,GREEN,r_knee,r_ankle)
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            self.stage="DOWN"
                            counter+=1

                        if side=="RIGHT" and angle_arm_right>=20 and angle_arm_right<=80 and angle_leg_left>=70 and angle_leg_left <=100:
                        #if side=="RIGHT" and angle_leg_left>=70 and angle_leg_left <=95:
                        #  render_extremities(image,GREEN,r_shoulder,r_elbow)
                        #  render_extremities(image,GREEN,r_elbow,r_wrist)
                        #  render_extremities(image,GREEN,l_hip,l_knee)
                        #  render_extremities(image,GREEN,l_knee,l_ankle)
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            self.stage="DOWN"
                            counter+=1
                        
                    except:
                        pass

                # Stage data
                    cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    cv2.putText(image, 'FEET', (125,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(side), 
                            (100,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    cv2.putText(image, str(self.stage), 
                            stage_info, 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                        

                    self.change_pixmap_signal.emit(image)
        # shut down capture system
            cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Run_March(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARCH IN PLACE")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        self.back_btn = QPushButton(self)
        self.back_btn.setObjectName("back_btn")
        self.back_btn.setText("Back")

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.back_btn)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = March()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.back_btn.clicked.connect(self.close_window)

    def close_window(self):
        self.thread.stop()
        self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class Leg_Push(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.stage = "START"

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
        upper_left = (image_width  // 10 , image_height  // 10)
        bottom_right = (image_width * 9 // 10 , image_height * 9 // 10 )
        counter = 0 
        side= None
        stage_info=(image_width  // 4 , image_height  // 10)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, image = cap.read()
                image = cv2.flip(image,1)
                if ret:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]                    
                        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    
                        left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                        left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


                        l_knee=convert_to_point(knee_left,image_width,image_height)
                        r_knee=convert_to_point(knee_right,image_width,image_height)
                        
                        l_ankle=convert_to_point(ankle_left,image_width,image_height)
                        r_ankle=convert_to_point(ankle_right,image_width,image_height)

                        l_hip=convert_to_point(hip_left,image_width,image_height)
                        r_hip=convert_to_point(hip_right,image_width,image_height)                

                        l_shoulder=convert_to_point(left_shoulder,image_width,image_height)
                        r_shoulder=convert_to_point(right_shoulder,image_width,image_height)

                        l_elbow=convert_to_point(left_elbow,image_width,image_height)
                        r_elbow=convert_to_point(right_elbow,image_width,image_height)

                        l_wrist=convert_to_point(left_wrist,image_width,image_height)
                        r_wrist=convert_to_point(right_wrist,image_width,image_height)
                        

                        angle_body_left=round(calculate_angle(left_elbow,left_shoulder,hip_left),2)  
                        angle_body_right=round(calculate_angle(right_elbow,right_shoulder, hip_right),2)  
                        
                        angle_back=round(calculate_angle(right_shoulder,hip_right, knee_right),2)

                        angle_leg_r=round(calculate_angle(right_shoulder,hip_right,ankle_right),2)
                        angle_leg_l=round(calculate_angle(left_shoulder,hip_left,ankle_left),2)
                    

                        render_point(image,BLACK,RED,l_ankle)
                        render_point(image,BLACK,RED,l_knee)
                        render_point(image,BLACK,RED,l_hip)
                        render_point(image,BLACK,RED,l_shoulder)
                        render_point(image,BLACK,RED,l_elbow)
                        render_point(image,BLACK,RED,l_wrist)
                        
                        render_point(image,BLACK,RED,r_ankle)
                        render_point(image,BLACK,RED,r_knee)
                        render_point(image,BLACK,RED,r_hip)
                        render_point(image,BLACK,RED,r_shoulder)
                        render_point(image,BLACK,RED,r_elbow)
                        render_point(image,BLACK,RED,r_wrist)

                        render_extremities(image,WHITE, l_ankle,l_knee)
                        render_extremities(image,WHITE, l_knee,l_hip)
                        render_extremities(image,WHITE, l_hip,l_shoulder)
                        render_extremities(image,WHITE, l_shoulder,l_elbow)
                        render_extremities(image,WHITE, l_elbow,l_wrist)

                        render_extremities(image,WHITE, r_ankle,r_knee)
                        render_extremities(image,WHITE, r_knee,r_hip)
                        render_extremities(image,WHITE, r_hip,r_shoulder)
                        render_extremities(image,WHITE, r_shoulder,r_elbow)
                        render_extremities(image,WHITE, r_elbow,r_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_body_left), 
                                tuple(np.multiply(left_shoulder, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_body_right), 
                                tuple(np.multiply(right_shoulder, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                        
                        cv2.putText(image, str(angle_leg_r), 
                                tuple(np.multiply(hip_right, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_leg_l), 
                                tuple(np.multiply(hip_left, [image_width, image_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)


                        if angle_body_right>=160 and angle_body_right>=160 :
                            render_extremities(image,YELLOW,l_shoulder,l_elbow)
                            render_extremities(image,YELLOW,l_elbow,l_wrist)
                            render_extremities(image,YELLOW,r_shoulder,r_elbow)
                            render_extremities(image,YELLOW,r_elbow,r_wrist)
                            self.stage="DOWN"

                        if self.stage=="DOWN" and angle_back>=130 and angle_back<=150:
                            self.stage="PUSH"
                        
                        if (counter % 2) == 0 :
                            side="LEFT"
                            render_extremities(image,YELLOW,l_shoulder,l_elbow)
                            render_extremities(image,YELLOW,l_elbow,l_wrist)
                            render_extremities(image,YELLOW,r_hip,r_knee)
                            render_extremities(image,YELLOW,r_knee,r_ankle)
                        else:
                            side="RIGHT"
                            render_extremities(image,YELLOW,r_shoulder,r_elbow)
                            render_extremities(image,YELLOW,r_elbow,r_wrist)
                            render_extremities(image,YELLOW,l_hip,l_knee)
                            render_extremities(image,YELLOW,l_knee,l_ankle)
                        
                        if side=="LEFT" and self.stage=="PUSH" and angle_leg_r>=110 and angle_leg_r<=140:
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            self.stage="UP"
                            counter+=1

                        if side=="RIGHT" and self.stage=="PUSH" and angle_leg_l>=110 and angle_leg_l<=140 :
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            self.stage="UP"
                            counter+=1

                    except:
                        pass
                    
                # Stage data
                    cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    cv2.putText(image, 'FEET', (125,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(side), 
                            (100,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                            
                    cv2.putText(image, str(self.stage), 
                            stage_info, 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                        

                    self.change_pixmap_signal.emit(image)
        # shut down capture system
            cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Run_Leg_Push(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALTERNATE LEG BEHIND")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        self.back_btn = QPushButton(self)
        self.back_btn.setObjectName("back_btn")
        self.back_btn.setText("Back")

  
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.back_btn) 
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = Leg_Push()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.back_btn.clicked.connect(self.close_window)
    
    def close_window(self):
        self.thread.stop()
        self.close()
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class Split_Squat(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.stage = "START"

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        image_width = int(cap.get(3))
        image_height = int(cap.get(4))
        upper_left = (image_width  // 10 , image_height  // 10)
        bottom_right = (image_width * 9 // 10 , image_height * 9 // 10 )
        counter = 0 
        side= None
        stage_info=(image_width  // 4 , image_height  // 10)
        toe_info=(image_width  // 3 , image_height  // 3)
        back_info=(image_width  // 3 , image_height  // 5)
        status_left = None
        status_right = None
        back_status=None
        toe = None
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, image = cap.read()
                image = cv2.flip(image,1)
                if ret:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]                    
                        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    
                        left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                        left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        left_foot=[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                        right_foot= [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]


                        l_knee=convert_to_point(knee_left,image_width,image_height)
                        r_knee=convert_to_point(knee_right,image_width,image_height)
                        
                        l_ankle=convert_to_point(ankle_left,image_width,image_height)
                        r_ankle=convert_to_point(ankle_right,image_width,image_height)

                        l_hip=convert_to_point(hip_left,image_width,image_height)
                        r_hip=convert_to_point(hip_right,image_width,image_height)                

                        l_shoulder=convert_to_point(left_shoulder,image_width,image_height)
                        r_shoulder=convert_to_point(right_shoulder,image_width,image_height)

                        l_elbow=convert_to_point(left_elbow,image_width,image_height)
                        r_elbow=convert_to_point(right_elbow,image_width,image_height)

                        l_wrist=convert_to_point(left_wrist,image_width,image_height)
                        r_wrist=convert_to_point(right_wrist,image_width,image_height)

                    # Calculate angle
                        angle_left = round(calculate_angle(hip_left, knee_left, ankle_left))
                        angle_right = round(calculate_angle(hip_right, knee_right, ankle_right))
                        center_right=round(calculate_angle(knee_right,hip_right,hip_left))
                        center_left=round(calculate_angle(knee_left,hip_left,hip_right))


                        render_point(image,BLACK,RED,l_ankle)
                        render_point(image,BLACK,RED,l_knee)
                        render_point(image,BLACK,RED,l_hip)
                        render_point(image,BLACK,RED,l_shoulder)
                        render_point(image,BLACK,RED,l_elbow)
                        render_point(image,BLACK,RED,l_wrist)
                        
                        render_point(image,BLACK,RED,r_ankle)
                        render_point(image,BLACK,RED,r_knee)
                        render_point(image,BLACK,RED,r_hip)
                        render_point(image,BLACK,RED,r_shoulder)
                        render_point(image,BLACK,RED,r_elbow)
                        render_point(image,BLACK,RED,r_wrist)

                        render_extremities(image,WHITE, l_ankle,l_knee)
                        render_extremities(image,WHITE, l_knee,l_hip)
                        render_extremities(image,WHITE, l_hip,l_shoulder)
                        render_extremities(image,WHITE, l_shoulder,l_elbow)
                        render_extremities(image,WHITE, l_elbow,l_wrist)

                        render_extremities(image,WHITE, r_ankle,r_knee)
                        render_extremities(image,WHITE, r_knee,r_hip)
                        render_extremities(image,WHITE, r_hip,r_shoulder)
                        render_extremities(image,WHITE, r_shoulder,r_elbow)
                        render_extremities(image,WHITE, r_elbow,r_wrist)

                        # Visualize angle
                        cv2.putText(image, str(angle_left), 
                                tuple(np.multiply(knee_left, [image_width, image_height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, str(angle_right), 
                                tuple(np.multiply(knee_right, [image_width, image_height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, str(center_right), 
                                tuple(np.multiply(hip_right, [image_width, image_height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, str(center_left), 
                                tuple(np.multiply(hip_left, [image_width, image_height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        if (counter % 2) == 0:
                            side = "LEFT"

                        else:
                            side="RIGHT"


                        if side=="LEFT":
                            toe=correct_toe_left(left_foot,knee_left,image_height,image_width)
                            angle_l_sh=calculate_angle(left_shoulder,hip_left,knee_left)
                            render_extremities(image,BLUE,l_shoulder,l_hip)
                            render_extremities(image,BLUE,l_hip,l_knee)
                            render_extremities(image,BLUE,l_knee,l_ankle)
                            
                            true_position_down_left(image,GREEN,l_hip)
                            true_position_up_left(image,GREEN,r_hip)#на правую ногу в пол
                            if angle_l_sh >=80 and angle_l_sh <=110 :
                                render_extremities(image,GREEN,l_shoulder,l_hip)
                                back_status="true"
                            else :
                                render_extremities(image,RED,l_shoulder,l_hip)
                                cv2.rectangle(image, upper_left, bottom_right,YELLOW, thickness=5)
                                cv2.putText(image, str("Watch your back"), back_info,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                                back_status="false"


                        if side=="RIGHT":
                            toe=correct_toe_right(right_foot,knee_right,image_height,image_width)
                            angle_r_sh=calculate_angle(right_shoulder,hip_right,knee_right)
                            render_extremities(image,BLUE,r_shoulder,r_hip)
                            render_extremities(image,BLUE,r_hip,r_knee)
                            render_extremities(image,BLUE,r_knee,r_ankle)

                            true_position_down_right(image,GREEN,l_hip)
                            true_position_up_right(image,GREEN,r_hip) # для левой ноги в пол

                            if angle_r_sh >=80 and angle_r_sh <=110 :
                                render_extremities(image,GREEN,r_shoulder,r_hip)
                                back_status="true"
                            else :
                                render_extremities(image,RED,r_shoulder,r_hip)
                                cv2.rectangle(image, upper_left, bottom_right,YELLOW, thickness=5)
                                cv2.putText(image, str("Watch your back"),back_info,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                                back_status="false"


                        if angle_left >=80 and angle_left <=110 :
                            render_extremities(image,GREEN,l_hip,l_knee)
                            render_extremities(image,GREEN,l_knee,l_ankle)
                            status_left = "OK"
                        else :
                            render_extremities(image,RED,l_hip,l_knee)
                            render_extremities(image,RED,l_knee,l_ankle)
                            status_left="not ok"


                        if angle_right >=80 and angle_right <=110 :
                            render_extremities(image,GREEN,r_hip,r_knee)
                            render_extremities(image,GREEN,r_knee,r_ankle)
                            status_right = "OK"
                        else :
                            render_extremities(image,RED,r_hip,r_knee)
                            render_extremities(image,RED,r_knee,r_ankle)
                            status_right="not ok"


                        if toe =="BAD" and side=="LEFT" :
                            render_extremities(image,YELLOW,l_knee,l_ankle)
                            cv2.rectangle(image, upper_left, bottom_right,YELLOW, thickness=5)
                            cv2.putText(image, str("Watch your knee"), 
                                toe_info,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)


                        if toe =="BAD" and side=="RIGHT" :
                            render_extremities(image,YELLOW,r_knee,r_ankle)
                            cv2.rectangle(image, upper_left, bottom_right,YELLOW, thickness=5)
                            cv2.putText(image, str("Watch your knee"), 
                                toe_info,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)


                        if  status_right == "OK" and status_left=="OK":
                            if back_status=="true" :
                                cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            else :
                                cv2.rectangle(image, upper_left, bottom_right,YELLOW, thickness=5)
                            self.stage="UP"

                            
                        if angle_left >100 and angle_left <=120 and angle_right >100 and angle_right<=120 : 
                            cv2.rectangle(image, upper_left, bottom_right,RED , thickness=5)


                        if angle_left>120 and angle_right>120 and self.stage=="UP" :
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)
                            render_extremities(image,GREEN, l_hip,l_knee)
                            render_extremities(image,GREEN, l_knee,l_ankle)
                            render_extremities(image,GREEN, r_hip,r_knee)
                            render_extremities(image,GREEN, r_knee,r_ankle)
                            self.stage = "DOWN"
                            counter+=1


                        if self.stage == "DOWN" :
                            cv2.rectangle(image, upper_left, bottom_right,GREEN, thickness=5)


                        if status_right== "not ok" or status_left=="not ok":
                            cv2.rectangle(image, upper_left, bottom_right,RED, thickness=5)

                    except:
                        pass
                    
                    

                    cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

                    cv2.putText(image, 'FEET', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                    cv2.putText(image, str(side), 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)
                    # Stage data
                
                    cv2.putText(image, str(self.stage), 
                            stage_info, 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, BLACK, 2, cv2.LINE_AA)
                        

                    self.change_pixmap_signal.emit(image)
        # shut down capture system
            cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Run_Split_Squat(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPLIT SQUAT")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        self.back_btn = QPushButton(self)
        #self.back_btn.setGeometry(QtCore.QRect(1150, 10, 100, 50))
        self.back_btn.setObjectName("back_btn")
        self.back_btn.setText("Back")
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.back_btn)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = Split_Squat()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.back_btn.clicked.connect(self.close_window)

    def close_window(self):
        self.thread.stop()
        self.close()
   
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)