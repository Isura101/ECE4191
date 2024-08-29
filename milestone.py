import RPi.GPIO as GPIO
import serial
import math
import time
from distance_ball import calculate_ball_distances_and_angles
from object_detection import turn_camera_on, capture_and_process_image, turn_camera_off
import cv2
import random
import supervision as sv
from inference import get_model
import matplotlib.pyplot as plt
import os
import time
import warnings
import math
import numpy as np
import os
import glob


# Suppress specific warnings
warnings.filterwarnings("ignore")

# Initialize the model once
ROBOFLOW_API_KEY = "O9esxBwq5ZDGn9ZEDRAU"
model = get_model(model_id="tennis-ball-8ixxh/1", api_key=ROBOFLOW_API_KEY)

# Define the directory to save the annotated images
output_directory = "/home/isura99/Desktop/ECE4191/Project/images"

os.makedirs(output_directory, exist_ok=True)

# Function to turn the camera on
def turn_camera_on(camera_id=0):
    camera = cv2.VideoCapture(camera_id)
    time.sleep(2)  # Allow the camera to warm up
    return camera

state_dict = {
    "TURNING_TO_CENTRE": "COMPLETED_TURN_CENTRE",
    "COMPLETED_TURN_CENTRE": "MOVING_TO_CENTRE",
    "MOVING_TO_CENTRE": "COMPLETED_TURN_SEARCH",
    "MOVING_TO_SEARCH": "TURNING_TO_SEARCH",
    "TURNING_TO_SEARCH": "COMPLETED_TURN_SEARCH" ,  # need to manually change states from COMPLETED_TURN_SEARCH to either TURNING_TO_SEARCH or "TURNING_TO_BALL"
    "COMPLETED_TURN_SEARCH": "COMPLETED_TURN_SEARCH",  # state comes back to itself 
    "TURNING_TO_BALL": "COMPLETED_TURN_BALL",
    "COMPLETED_TURN_BALL" : "MOVING_TO_BALL",
    "MOVING_TO_BALL": "BALL_COLLECTED",
    "BALL_COLLECTED": "TURNING_TO_ORIGIN",
    "TURNING_TO_ORIGIN": "COMPLETED_TURN_ORIGIN",
    "COMPLETED_TURN_ORIGIN": "MOVING_TO_ORIGIN",
    "MOVING_TO_ORIGIN": "AT_ORIGIN"
}

court_dimensions = {
    "length": 5.11,  # in metres 
    "width": 3.9,  # in metres 
}
# motor 1 is left motor 
# motor 2 is right motor 
class LocalisationPosition:
    def __init__(self, x = 0, y = 0, theta = math.pi/2):
        self.x = x  # robots x coordinate
        self.y = y  # robots y coordinate
        self.theta = theta  # robots angle (what direction it is facing)
        self.tune_theta = False  # variable indicates if we should tune theta to be more accurate
        self.tuning_theta_factor_clockwise = 0.015  # factor to decrement theta by
        self.tuning_theta_factor_anticlockwise = 0.015  # factor to decrement theta by
        self.tangential_vel_l = 0  # left wheel tangential velocity 
        self.tangential_vel_r = 0  # right wheel tangential velocity 
        self.velocity = 0  # robot velocity
        self.prev_time = 0  # previous time when we updated positonal data
        self.current_time = 0  # current time when we updated positional data
    
    def update_theta(self, tangential_vel_r, tangential_vel_l, wheel_dist_apart, deltaT, direction: int):
        omega_robot = (tangential_vel_r - tangential_vel_l)/(wheel_dist_apart/100) # Rotating clockwise is negative, anti-clockwise is positive
        print("UPDATE THETA: t_v1: {}, t_v2: {}, omega_robot: {}, deltaT: {}".format(tangential_vel_l, tangential_vel_r, omega_robot*(180/math.pi)*deltaT, deltaT))
        self.theta = self.theta + omega_robot*deltaT # in radians
        #print("NEW THETA: {}".format(self.theta))
        if self.tune_theta and abs(omega_robot) > 0:
            print("TUNING IS HAPPENING ------------------------------!!!!!")
            if direction == 0: #anticlockwise
                self.theta += - self.tuning_theta_factor_anticlockwise
            else:
                #print("TUNING IS HAPPENING ------------------------------!!!!!")
                self.theta += self.tuning_theta_factor_clockwise
        #print("NEW THETA AFTER TUNING: {}".format(self.theta))
    
    def reset_theta(self):
        if self.theta > 2*math.pi:
            while self.theta > 2*math.pi:
                self.theta += - 2*math.pi 
    
        elif self.theta < 0:
            while self.theta < 0:
                self.theta +=  2*math.pi 
        else:
            self.theta = self.theta
        
class PositionToMove:
    def __init__(self, x = 0, y = 0, theta = 0):
        self.des_x: int = x
        self.des_y :int = y
        self.direction: int = 0  # anticlockwise if 0 and clockwise if 1
        self.theta: int = theta
        self.manage_theta : bool = False
        self.manage_position: bool = False
        self.prev_x = 0
        self.prev_y = 0
        self.distance = 0 

    
    def update_theta(self, theta_to_change, current_theta):
        self.theta = current_theta + theta_to_change 
    
    def reset_theta(self):
        if self.theta > 2*math.pi:
            while self.theta > 2*math.pi:
                self.theta += - 2*math.pi 
    
        elif self.theta < 0:
            while self.theta < 0:
                self.theta +=  2*math.pi 
        else:
            self.theta = self.theta

class RobotRPM:
    def __init__(self, rpm1 = 0, rpm2 = 0):
        self.rpm1 = rpm1
        self.rpm2 = rpm2
    
    def clear(self):
        self.rpm1 = 0
        self.rpm2 = 0

class PID:
    def __init__(self, rpmM1Des, rpmM2Des, integral_error1, integral_error2, Ki1, Ki2, Kp1, Kp2):
        self.rpmM1Des = rpmM1Des
        self.rpmM2Des = rpmM2Des
        self.integral_error1 = integral_error1
        self.integral_error2 = integral_error2
        self.Ki1 = Ki1
        self.Ki2 = Ki2 
        self.Kp1 = Kp1
        self.Kp2 = Kp2 

class CameraTime:
    def __init__(self):
        self.prevTime = 0
        self.currentTime = 0
        
# Raspbery pi pin definitions
in1 = 16 # Motor 1 direction
in2 = 26 # Motor 1 direction
ena1 = 12 # Motor 1 enable

in3 = 5 # Motor 2 direction
in4 = 6 # Motor 2 direction
ena2 = 13 # Motor 2 enable

GPIO.setmode(GPIO.BCM)  # this indicates how we are numbering our pins in the PI

GPIO.setup(in1,GPIO.OUT)  
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(ena1,GPIO.OUT)
d1 = GPIO.PWM(ena1,100) # Duty cycle of motor 1
d1.start(0)

GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(ena2,GPIO.OUT)
d2 = GPIO.PWM(ena2,100) # Duty cycle of motor 2
d2.start(0)

GPIO.output(in1,GPIO.HIGH) # ------- Clockwise direction 
GPIO.output(in2,GPIO.LOW)  # ---------------------------
GPIO.output(in3,GPIO.HIGH) # ------- Clockwise direction 
GPIO.output(in4,GPIO.LOW)  # ---------------------------

# GPIO pin 14 is the RX port for UART
comm_uart = serial.Serial('/dev/serial0', 9600)

# Wheel parameters
wheel_dist_apart = 19.65 # in cm
wheel_diameter = 5.6 # in cm
wheel_circumference = math.pi*wheel_diameter # in cm

robot_state = "TURNING_TO_CENTRE"

robot_localisation = LocalisationPosition(0,0,0)
position_to_move = PositionToMove()
rpmMotor = RobotRPM()

# PID control parameters
pid_control = PID(40, 40, 0, 0, 3, 3, 0.3, 0.3)

deltaT = 0.5  # variable used to track the change in angle and distance (using RPM) 

def turn_right():
    GPIO.output(in1,GPIO.HIGH) # ------- Clockwise direction 
    GPIO.output(in2,GPIO.LOW)  # ---------------------------
    GPIO.output(in3,GPIO.LOW) # ------- Anticlockwise direction 
    GPIO.output(in4,GPIO.HIGH)  # ---------------------------
    d1.ChangeDutyCycle(25)
    d2.ChangeDutyCycle(25)

def turn_left():
    GPIO.output(in1,GPIO.LOW) # ------- Anticlockwise direction 
    GPIO.output(in2,GPIO.HIGH)  # ---------------------------
    GPIO.output(in3,GPIO.HIGH) # ------- Clockwise direction 
    GPIO.output(in4,GPIO.LOW)  # ---------------------------
    d1.ChangeDutyCycle(25) 
    d2.ChangeDutyCycle(25)

def move_forward():
    global pid_control
    pid_control.integral_error1 = 0 
    pid_control.integral_error2 = 0
    GPIO.output(in1,GPIO.HIGH) # ------- Clockwise direction 
    GPIO.output(in2,GPIO.LOW)  # ---------------------------
    GPIO.output(in3,GPIO.HIGH) # ------- Clockwise direction 
    GPIO.output(in4,GPIO.LOW)  # ---------------------------

def stop():
    d1.ChangeDutyCycle(0)
    d2.ChangeDutyCycle(0)


def PID_control(rpmMotor1Act, rpmMotor2Act, pid_control: PID):
    error_motor_1 = pid_control.rpmM1Des - rpmMotor1Act # Error for motor 1
    error_motor_2 = pid_control.rpmM2Des - rpmMotor2Act # Error for motor 2

    pid_control.integral_error1 = pid_control.integral_error1 + error_motor_1*deltaT # Estimated integral of error for motor 1
    pid_control.integral_error2 = pid_control.integral_error2 + error_motor_2*deltaT # Estimated integral of error for motor 2

    u_motor_1 = pid_control.Kp1*error_motor_1 + pid_control.Ki1*pid_control.integral_error1 # Controller output for motor 1
    u_motor_2 = pid_control.Kp2*error_motor_2 + pid_control.Ki2*pid_control.integral_error2 # Controller output for motor 2

    # Changing duty cycle
    pwrMotor2 = abs(u_motor_2)
    if (pwrMotor2 > 255):
        pwrMotor2 = 255
    d2.ChangeDutyCycle((pwrMotor2/255)*100)

    # Changing duty cycle
    pwrMotor1 = abs(u_motor_1)
    if (pwrMotor1 > 255):
        pwrMotor1 = 255
    d1.ChangeDutyCycle((pwrMotor1/255)*100)

    #print("rpm1: {}, rpm2: {}".format(rpmMotor1Act, rpmMotor2Act))


def update_position(positional_data: LocalisationPosition, rpmMotor1: int, rpmMotor2: int, position_to_move: PositionToMove):
    #print("rpm1: {}, rpm2: {}".format(rpmMotor1, rpmMotor2))
    global deltaT

    positional_data.tangential_vel_l = (rpmMotor1*wheel_circumference/100)/60 # tangential velocity of Motor 1 in m/s
    positional_data.tangential_vel_r = (rpmMotor2*wheel_circumference/100)/60 # tangential velocity of Motor 2 in m/s
    positional_data.velocity = (positional_data.tangential_vel_l + positional_data.tangential_vel_r)/2
    
    positional_data.update_theta(positional_data.tangential_vel_r, positional_data.tangential_vel_l, wheel_dist_apart, deltaT, position_to_move.direction)    

    if not position_to_move.manage_theta:
        positional_data.x = positional_data.x + positional_data.velocity*math.cos(positional_data.theta)*deltaT # in m
        positional_data.y = positional_data.y + positional_data.velocity*math.sin(positional_data.theta)*deltaT # in m

def update_time(positional_data: LocalisationPosition):
    global deltaT
    if positional_data.current_time == 0 and positional_data.prev_time == 0:
        positional_data.prev_time = time.time()
        deltaT = 0.5  # set to 0.5 seconds since this function is called approximately every 0.5 secs (since receiving rpm comes every 0.5s)
    else:
        positional_data.current_time = time.time()
        deltaT = positional_data.current_time - positional_data.prev_time
        positional_data.prev_time = positional_data.current_time

def track_position(position_to_move: PositionToMove, current_position: LocalisationPosition, rpmMotor: RobotRPM):  # function used to indicate if we have reached the destination coordinates 
    #print("current position theta: {} and position to move theta: {}".format(current_position.theta, position_to_move.theta))
    print("curr x: {} curr y: {} pos x: {} pos y: {} curr theta: {} pos theta {}".format(current_position.x, current_position.y, position_to_move.des_x, position_to_move.des_y, current_position.theta *180/math.pi, position_to_move.theta*180/math.pi))
    #print("delta_position_x: {}, delta_position_y: {}".format(current_position.x - position_to_move.x, current_position.y - position_to_move.y))
    #print("delta theta: {}".format(abs(current_position.theta - position_to_move.theta)))
    global robot_state
    if position_to_move.manage_theta:
        if abs(current_position.theta - position_to_move.theta) <= 0.07 or \
            (position_to_move.direction == 0 and current_position.theta >= position_to_move.theta) or \
            (position_to_move.direction == 1 and current_position.theta <= position_to_move.theta):  # if we are within 4 degrees of desired angle then stop. 0.07 is 4 degrees in radians
                position_to_move.manage_theta = False
                #current_position.reset_theta()
                #position_to_move.reset_theta()
                print("ANGLE COMPLETE!!")
                stop()  # stop both motors
                time.sleep(1)  # sleep to ensure the rpm read will be 0 in the next serial transmition
                rpmMotor.clear()  # manually clear the rpm to 0 for correctness
                current_position.tune_theta = False  # we don't need to tune theta when we aren't turning 
                robot_state = state_dict[robot_state]
                
    elif position_to_move.manage_position:
        distance = math.sqrt((current_position.x - position_to_move.prev_x)**2 + (current_position.y - position_to_move.prev_y)**2)
        print("distance to target is : {}".format(distance))
        if distance >= position_to_move.distance:
            position_to_move.manage_position = False
            stop()
            rpmMotor.clear()
            time.sleep(1)
            if robot_state == "MOVING_TO_ORIGIN":
                print("distance has reached for ORIGIN")
            else:
                print("distance has reached for BALL COLLECTION")
            robot_state = state_dict[robot_state]
        elif robot_state[0:14] == "COMPLETED_TURN":
            move_forward()
            robot_state = state_dict[robot_state]
            print("MOVING FORWARD TO BALL!!!")
        


def calculate_coords(theta: int, distance: int, position_to_move: PositionToMove, current_position: LocalisationPosition):
    # theta is positive if we are rotating anticlockwise and theta is negative if clockwise 
    print("CALCULATE COORDS: theta: {}, distance {}".format(theta, distance))
    global robot_state
    position_to_move.update_theta(theta, current_position.theta)
    position_to_move.direction = 0 if theta > 0 else 1
    new_theta = position_to_move.theta
    position_to_move.des_x = current_position.x + distance*math.cos(new_theta)
    position_to_move.des_y = current_position.y + distance*math.sin(new_theta)
    position_to_move.prev_x = current_position.x
    position_to_move.prev_y = current_position.y
    position_to_move.distance = distance
    if abs(theta) >= math.pi/4:
            current_position.tune_theta = True

def start_move_to_ball(position_to_move: PositionToMove, current_position: LocalisationPosition):
    global robot_state
    position_to_move.manage_position = True 
    theta = abs(current_position.theta - position_to_move.theta)
    if theta != 0:
        position_to_move.manage_theta = True
        if position_to_move.direction == 1:
            turn_right()
        elif position_to_move.direction == 0:
            turn_left()
        robot_state = "TURNING_TO_BALL"
    else:
        robot_state = "COMPLETED_TURN_BALL"


def angle_to_origin(current_position: LocalisationPosition):
    x = current_position.x
    y = current_position.y
    angle_to_turn = 0
    theta_d = current_position.theta
    theta_c = current_position.theta
    if x == 0 or y == 0:
        angle_to_turn = math.pi
        theta_d = theta_c + math.pi 
    else:
        if x > 0 and y > 0:
            theta_d = math.pi + math.atan(y/x)
            print("FIRST QUADRANT")
        elif x < 0 and y > 0:
            theta_d = 2*math.pi - math.atan(y/x)
        elif x < 0 and y < 0:
            theta_d = math.atan(y/x)
        elif x > 0 and y < 0 : 
            theta_d = math.pi - math.atan(y/x)        
        angle_to_turn = theta_d - theta_c
    return angle_to_turn



def return_to_origin(position_to_move: PositionToMove, current_position: LocalisationPosition):
    current_position.reset_theta()
    position_to_move.reset_theta()
    position_to_move.des_x = 0  # return to origin x
    position_to_move.des_y = 0  # return to origin y
    position_to_move.prev_x = current_position.x 
    position_to_move.prev_y = current_position.y
    position_to_move.distance =  math.sqrt((current_position.x - position_to_move.des_x)**2 + (current_position.y - position_to_move.des_y)**2)
    turning_angle = angle_to_origin(current_position) 
    if turning_angle < 0:
        turning_angle += 10*(math.pi/180)
    else:
        turning_angle += 15*(math.pi/180)
    position_to_move.update_theta(turning_angle, current_position.theta)  
    position_to_move.direction = 0 if turning_angle >= 0 else 1 # rotate 180 degrees anticlockwise
    position_to_move.manage_position = True
    position_to_move.manage_theta = True
    if abs(turning_angle) >= math.pi/4:
        current_position.tune_theta = True
    if position_to_move.direction == 1:
        turn_right()
    elif position_to_move.direction == 0:
        turn_left()
    
    print("-----------------RETURNING TO ORIGIN where desired angle is {} and turning angle is {}----------------------".format(position_to_move.theta * (180/math.pi), turning_angle * (180/math.pi)))

def move_to_centre(position_to_move: PositionToMove, current_position: LocalisationPosition):
    distance_to_centre = math.sqrt((court_dimensions["length"])**2 + (court_dimensions["width"])**2)/2
    angle = 40 * (math.pi/180)
    calculate_coords(angle, distance_to_centre, position_to_move, current_position)
    position_to_move.manage_theta = True 
    position_to_move.manage_position = True 
    turn_left()
    
    
#centre_theta = 5 * (math.pi/180) # in radians
#centre_distance = 1.4 # in m

#calculate_coords(centre_theta, centre_distance, position_to_move, robot_localisation)

# Example camera matrix
camera_matrix = np.array( [[628.97221861, 0, 299.88990706],
                            [0, 627.75868542, 235.84576944],
                            [0, 0, 1, ]])

#camera = turn_camera_on()  # Turn the camera on
camera_time_track = CameraTime()
camera_time_track.prevTime = time.time()

move_to_centre(position_to_move, robot_localisation)

try: 
    while(1):
        camera_time_track.currentTime = time.time()
        if robot_state == "COMPLETED_TURN_SEARCH" and camera_time_track.currentTime - camera_time_track.prevTime >= 1:
            camera = turn_camera_on()  # Turn the camera on
            print("___________________TAKE PHOTO___________________")
            image_file, txt_file = capture_and_process_image(camera)
            turn_camera_off(camera)
            print("___________________PROCESS IMAGE___________________")
            result = calculate_ball_distances_and_angles(txt_file, camera_matrix)
            camera_time_track.prevTime = camera_time_track.currentTime
            if len(result["tennis-ball"]) > 0:
                print("___________________BALL FOUND___________________")
                D_horizontal, theta_true = result["tennis-ball"][0] # taking the 0th index in this file since we only detect 1 ball for milestone 1 
                theta = -theta_true * (math.pi/180) # in radians
                print("BALL DETECTED, theta is :{} and distance is: {}".format(theta_true, D_horizontal))
                calculate_coords(theta, D_horizontal, position_to_move, robot_localisation)
                start_move_to_ball(position_to_move, robot_localisation)
            else:  # we did not find a ball. Turn 45 degrees
                robot_state = "TURNING_TO_SEARCH"
                theta = -45 * (math.pi/180) # in radians
                calculate_coords(theta, 0, position_to_move, robot_localisation)
                position_to_move.manage_theta = True 
                turn_right()        

        if comm_uart.in_waiting > 0: # If there's any byte of data available in the buffer
            data = comm_uart.readline().decode('utf-8').strip()
            parts = data.split(',')

            if len(parts) == 2:
                rpmMotor.rpm1 = float(parts[0].split(':')[1].strip())
                rpmMotor.rpm2 = float(parts[1].split(':')[1].strip())
                
                #update_time(robot_localisation) 

                # statement is required as sometimes the buffer uses old rpm values after turn is completed and gives incorrect position updates. 
                if robot_state[0:14] == "COMPLETED_TURN": #"COMPLETED_TURN_BALL" or robot_state == "COMPLETED_TURN_ORIGIN" or robot_state == "COMPLETED_TURN_ORIGIN": 
                    # Thus 0 rpm is hard coded as an argument using robot state
                    rpmMotor.rpm1 = 0
                    rpmMotor.rpm2 = 0

                print("rpm1: {}, rpm2: {}".format(rpmMotor.rpm1, rpmMotor.rpm2))
                update_position(robot_localisation, rpmMotor.rpm1 , rpmMotor.rpm2, position_to_move)  # call function repeatedly for robot localisation
                track_position(position_to_move, robot_localisation, rpmMotor)   # track position is used to turn off motors if we are trying to get somewhere
            

                #print("x: {}, y: {}, theta: {}".format(robot_localisation.x, robot_localisation.y, robot_localisation.theta))
                if robot_state[0:6] == "MOVING": # "MOVING_TO_BALL" or robot_state == "MOVING_TO_ORIGIN" or robot_state == "MOVING_TO_SEARCH":
                    #print("PID IS WORKING")
                    PID_control(rpmMotor.rpm1, rpmMotor.rpm2, pid_control)
                
                if robot_state == "BALL_COLLECTED":
                    return_to_origin(position_to_move, robot_localisation)
                    robot_state = state_dict[robot_state]    # "TURNING_TO_ORIGIN"

                print("robot state: {}".format(robot_state))
                #print("deltaT: {}".format(deltaT))

except KeyboardInterrupt:
    # Get all files in the directory
    files = glob.glob(os.path.join('/home/isura99/Desktop/ECE4191/Project/images', '*'))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    GPIO.cleanup()
    comm_uart.reset_input_buffer()