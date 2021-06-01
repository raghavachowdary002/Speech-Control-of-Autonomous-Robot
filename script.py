#!/usr/bin/env python
import librosa
import tensorflow as tf
import numpy as np
import sounddevice as sd
import soundfile as sf
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#import new2

SAVED_MODEL_PATH = "/home/raghav/catkin_ws/src/turtle/src/newmodel.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = [
        "Go_To_Home_Position",
        "Go_To_Kitchen",
        "Go_To_Living_Room",
        "Go_To_Origin",
        "Go_To_Room_One",
        "Go_To_Room_Two",
        "Move_To_Home_Position",
        "Move_To_Kitchen",
        "Move_To_Living_Room",
        "Move_To_Origin",
        "Move_To_Room_One",
        "Move_To_Room_Two"
    ]
    _instance = None
    
    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword
        
    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)
        return MFCCs.T
     
def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance
    
import os
import tkinter as tk
from tkinter import filedialog
global file_path
global v1

root = tk.Tk()
root.title("Speech Control of Autonomus Robots")

canvas = tk.Canvas(root, width=1200, height=400)
canvas.grid(rowspan=20, columnspan=20)

tk.Label(root, text="Load Audio File", font='customFont1', fg="black", bg="sky blue",
         width=15).grid(columnspan=1, row=2, column=1)

def browse_file() :
    global file_path
    filename = filedialog.askopenfilename()
    file_path.set(filename)


file_path = tk.StringVar()
filePath = tk.Entry(root, width=60, textvariable=file_path).grid(row=2, column=2)

browseFileBtn = tk.Button(root, text="Browse File", command=lambda:browse_file(),
                          font='customFont1', bg="azure", fg="black", height=2, width=15)

browseFileBtn.grid(row=2, column=3)
def mains():
    kss = Keyword_Spotting_Service()
    global keyword1
    print(file_path.get())
    keyword1 = kss.predict(str(file_path.get()))
    return keyword1
v1=""
def keyword_service() :
    global v1
    # v1 = keyword_spotting_service._Keyword_Spotting_Service.Keyword_Spotting_Service()
    v1.set(mains())
    print(v1)
    
    command = tk.Entry(root, width=60, textvariable=v1).grid(row=3, column=2)
    
    
    
v1 = tk.StringVar()

browseActionBtn = tk.Button(root, text="Recorded Audio Command", command=lambda:keyword_service(),
                          font='customFont1', bg="azure", fg="black", height=2, width=20)

browseActionBtn.grid(row=3, column=1)
command = tk.Entry(root, width=60, textvariable=v1.get()).grid(row=3, column=2)

#def final():
 #   main()
#    #os.system('cd /home/raghav/catkin_ws/src/turtle/src')
#   #os.system('python script.py')
    
#moveBtn = tk.Button(root, text="Move the Robot to the above recorded location", command=lambda:final(),
#                          font='customFont1', bg="azure", fg="black", width=40, wraplength=250)

#moveBtn.grid(row=4, column=2)

#root.mainloop()

#Room name

room = v1
if room=="Move_To_Room_One" or room=="Go_To_Room_one":
    room="Room One"
elif room=="Move_To_Room_Two" or room=="Go_To_Room_Two":
    room="Room Two"
elif room=="Move_To_Kitchen" or room=="Go_To_Kitchen":
    room="Room Four"
elif room=="Move_To_Living_Room" or room=="Go_To_Living_Room":
    room="Room Three"
elif room=="Move_To_Home_Position" or room=="Go_To_Living_Room":
    room="Room Four"
elif room=="Move_To_Origin" or room=="Go_To_Origin":
    room="Origin"

#Defining desired heading (Degrees)
goal_angle = np.deg2rad(-90)

#Define desired velocity(m/s)
desired_velocity = 0.5

#Initialising Global Variables
switch = 0
ret=0
#Gains for controllers designed for rotation and forward movement
x_gain = 0.9
y_gain = 1.8

# Publisher for sending velocity commands to Turtlebot
pub = rospy.Publisher('/cmd_vel',Twist, queue_size=1)


def initNode():
    # Here we initialize our node running the automation code
    rospy.init_node('polaris', anonymous=True)
     
    # Subscribe to topics for velocity and laser scan from Flappy Bird game
    rospy.Subscriber("/odom", Odometry,rotate)
    
    # Ros spin to prevent program from exiting
    rospy.spin()


def turtlemove():
    #Defining scope of the variables as global
    global switch
    global goal_angle
    global goal_x
    global goal_y
    global x_gain
    global y_gain
    global desired_velocity
    global ret
    global room
    #Cloning the Twist method into a variable	
    move = Twist()

    #Assuring no unnecessary movements
    move.linear.y = 0
    move.linear.z = 0
    move.angular.x = 0
    move.angular.y = 0

    #Switch Cases

    #Switch Case 0
    if switch ==0:
        print("switch case: " + str(switch))
        if (xx >4.5 and yy<0):
            switch=10
            
        if (xx >3.5 and yy<1.5 and switch!=10):
            switch=11
            
        
            
        if (xx<-6 and yy<0.5 and switch!=12):
            switch=13
        if (xx <-1.5 and yy>2.5):
            switch=12
            
                  
        if (room=="Room one" and switch==0) or (room == "Room two" and switch ==0):
            goal_angle = np.deg2rad(0) 
            switch=2
        elif (room == "Room three" and switch==0) or ( room == "Room four" and switch==0):
            goal_angle = np.deg2rad(180) 
            switch=2

    #Switch Case 1
    elif switch ==1:

        #Calculating and defining the error between current heading and desired heading (radians) and current (x,y) and desired (x,y) (m,m)
        error_angle = (goal_angle)-(yaw)

        #P controller for maintaing the desired heading and moving the turtlebot forward at a desired velocity 
        move.angular.z = y_gain*(error_angle)
        move.linear.x = 0.3
        pub.publish(move)
        if (yy>1.2):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(0)  
            switch=2
            
            
            
    elif switch ==2:
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians)
        error_angle = goal_angle-yaw
        
        #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
        move.angular.z = x_gain*(error_angle)    
        move.linear.x = 0
        pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
        if error_angle<0.01:
            move.angular.z = 0
            pub.publish(move)
            switch=3
           
            
    elif switch ==3: 
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians) and current (x,y) and desired (x,y) (m,m)
        error_angle = (goal_angle)-(yaw)

        #P controller for maintaing the desired heading and moving the turtlebot forward at a desired velocity 
        move.angular.z = y_gain*(error_angle)
        move.linear.x = 0.3
        pub.publish(move)
        if (xx>5 and room == "Room one" and ret!=1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)  
            switch=4
        if (xx>4 and room == "Room two" and ret!=1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(90)  
            switch=4
        if (xx<-2 and room == "Room three" and ret!=1) or (xx<-2 and room == "Room four" and ret!=1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(90)  
            switch=4    
        if (xx>4.5 and yy>0 and room == "Room two" and ret==1) or (xx>4.5 and yy>0 and room == "Room three" and ret==1) or (xx>4.5 and yy>0 and room == "Room four" and ret==1) or (xx>4.5 and yy>0 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(180)  
            switch=4  
        if (yy>3 and xx<-5 and room == "Room one" and ret==1) or (yy>3 and xx<-5 and room == "Room three" and ret==1) or (yy>3 and xx<-5 and room == "Room two" and ret==1) or (yy>3 and xx<-5 and room == "Origin" and ret==1):
            move.linear.x = 0 
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(0)  
            switch=4
            
        if (xx>3.5 and xx<4.5 and yy<0 and room == "Room three" and ret==1) or (xx>3.5 and yy<0 and xx<4.5 and room == "Room four" and ret==1) or (xx>3.5 and xx<4.5 and yy<0 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(180)  
            switch=4      
        if (xx>3.5 and yy<0 and room == "Room one" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(0)
            ret=0
            
            switch=3       
            
            
            
                
                  
    elif switch ==4:
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians)
        error_angle = goal_angle-yaw
        
        #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
        move.angular.z = x_gain*(error_angle)    
        move.linear.x = 0
        pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
        if error_angle<0.01:
            move.angular.z = 0
            pub.publish(move)
            
            switch=5
            
    elif switch ==5: 
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians) and current (x,y) and desired (x,y) (m,m)
        error_angle = (goal_angle)-(yaw)

        #P controller for maintaing the desired heading and moving the turtlebot forward at a desired velocity 
        move.angular.z = y_gain*(error_angle)
        move.linear.x = 0.3
        pub.publish(move)
        if (yy<-1 and room == "Room one" and ret!=1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            room=""
            switch=10
        if (yy<1 and xx<4 and room == "Room two" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(90)  
            switch=6    
        if (xx<-2 and yy<1 and room == "Room three" and ret==1) or (xx<-2 and yy<1 and room == "Room four" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(90)  
            switch=6     
        if (yy>1 and room == "Room two" and ret!=1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            room=""
            switch=11
            #goal_angle = np.deg2rad(90)      
        if (yy>3 and xx>-3 and room == "Room three") or (yy>3 and xx>-3 and room == "Room four"):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(180)  
            switch=6    
        
        if (xx<0 and yy<0.5 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            room=""
            switch=0 
        
        if (xx<0 and yy<1 and room == "Room three" and ret==1) or (xx<0 and yy<1 and room == "Room four" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            ret=0
            switch=3
            
              
        if (xx>-4 and yy>2 and room == "Room three" and ret==1):
            move.linear.x = 0
            
            pub.publish(move)
            goal_angle = np.deg2rad(-90)
            error_angle = goal_angle-yaw
            
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                
                    
                switch=14
            
            
            
            
        if (xx>-2 and yy>2 and room == "Room one" and ret==1) or (xx>-2 and yy>2 and room == "Room two" and ret==1) or (xx>-2 and yy>2 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)
            switch=6
            
            
                
    elif switch ==6:
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians)
        error_angle = goal_angle-yaw
        
        #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
        move.angular.z = x_gain*(error_angle)    
        move.linear.x = 0
        pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
        if error_angle<0.01:
            move.angular.z = 0
            pub.publish(move)
            switch=7        
            
            
    elif switch ==7: 
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians) and current (x,y) and desired (x,y) (m,m)
        error_angle = (goal_angle)-(yaw)

        #P controller for maintaing the desired heading and moving the turtlebot forward at a desired velocity 
        move.angular.z = y_gain*(error_angle)
        move.linear.x = 0.3
        pub.publish(move)
           
        if (xx<-4 and room == "Room three" and ret==0):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)  
            switch=14   
        if (xx<-7.5 and room == "Room four" and ret==0):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)  
            switch=8     
        if (xx>2 and yy>1 and room == "Room two" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            room=""
            switch = 11
            
        if (yy>3 and room == "Room three" and ret==1) or (yy>3 and room == "Room four" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(180)  
            switch=8       
        if (yy<0 and room == "Room one" and ret==1) or (yy<0 and room == "Room two" and ret==1) or (yy<0 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(0)  
            switch=8     
            
            
    elif switch ==8:
        print("switch case: " + str(switch))
        #Calculating and defining the error between current heading and desired heading (radians)
        error_angle = goal_angle-yaw
        
        #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
        move.angular.z = x_gain*(error_angle)    
        move.linear.x = 0
        pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
        if error_angle<0.01:
            move.angular.z = 0
            pub.publish(move)
            
            switch=9
       
      
    elif switch ==9:
        print("switch case: " + str(switch)) 
        #Calculating and defining the error between current heading and desired heading (radians) and current (x,y) and desired (x,y) (m,m)
        error_angle = (goal_angle)-(yaw)

        #P controller for maintaing the desired heading and moving the turtlebot forward at a desired velocity 
        move.angular.z = y_gain*(error_angle)
        move.linear.x = 0.3
        pub.publish(move)
        if (yy<0 and room == "Room four"):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move) 
            room=""
            switch=13
        
            
        if (xx<-4 and room == "Room three" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)  
            switch = 14
        
        if (xx<-7.5 and room == "Room four" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            goal_angle = np.deg2rad(-90)  
            switch = 14   
            
        if (xx>0 and room == "Origin" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            room=""
            ret=0 
            switch = 0     
             
        if (xx>0 and room == "Room two" and ret==1) or (xx>0 and room == "Room one" and ret==1):
            move.linear.x = 0
            move.angular.z = 0
            pub.publish(move)
            
            ret=0 
            switch = 3 
            
            
            
    #Room 1 return case      
    elif switch == 10:
        ret=1
        print("Case 10 and return from room 1")
        if (room == "Room two"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
           
        if (room == "Room three"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
           
        if (room == "Room four"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
        if (room == "Origin"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
                
                
                
                
    elif switch == 11:    # ROom 2 return cases
        ret=1
        print("Case 11 and return from room 2")
        if (room == "Room four" or room =="Room one" or room =="Room three" or room=="Origin"):
            goal_angle = np.deg2rad(-90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
           
        
    
    elif switch == 12:   # Room 3 return cases
        ret=1
        print("Case 12 and return from room 3")
        if (room == "Room two" or room=="Room one" or room == "Origin"):
            goal_angle = np.deg2rad(0) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=5
           
        if (room == "Room four"):
            goal_angle = np.deg2rad(180) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                ret=0
                switch=7
           
       
                
                
                
    elif switch == 13:  # Room 4 return cases
        ret=1
        print("Case 13 and return from room 4")
        if (room == "Room two"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
           
        if (room == "Room three"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3
           
        if (room == "Room one"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3    
        if (room == "Origin"):
            goal_angle = np.deg2rad(90) 
            error_angle = goal_angle-yaw
        
            #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
            move.angular.z = x_gain*(error_angle)    
            move.linear.x = 0
            pub.publish(move)
        
            #Switching the state after turtlebot's heading is now the desired heading
            if error_angle<0.01:
                move.angular.z = 0
                pub.publish(move)
                switch=3    
    elif switch == 14:
         #Calculating and defining the error between current heading and desired heading (radians)
        error_angle = goal_angle-yaw
        
        #P controller for controlled rotation of turtlebot and publishing required rotation to maintain the error at zero  
        move.angular.z = x_gain*(error_angle)    
        move.linear.x = 0
        pub.publish(move)
        
        #Switching the state after turtlebot's heading is now the desired heading
        
        
        if error_angle<0.01:
            move.angular.z = 0
            pub.publish(move)
            if (room=="Room three" and ret!=1):
                room=""
                switch = 12
            if (room=="Room three" and ret==1):
                room=""
                switch=12
     
            if (room == "Room four" and ret==1):
                 switch =9
                 ret=0            
     
        
def rotate(msg):
    #Defining yaw as a global variable so that it can be used for turtlebot movement in turtlemove function
    global yaw
    global xx
    global yy
    
    xx = msg.pose.pose.position.x
    yy = msg.pose.pose.position.y
    
    print("x=" + str(xx))
    
    print("y=" + str(yy))
    
    
    #Reading the turtlebot's orientation from odometry
    orientation_q = msg.pose.pose.orientation
    #Converting the orientation from quaternion to euler or radians
    orientation_list = [orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
    (roll,pitch,yaw) = euler_from_quaternion(orientation_list)
    print("Angle=" + str(yaw))
    #Calling the turtlemove function to move the turtlebot
    turtlemove()



def main():
    try:
        initNode()
    except rospy.ROSInterruptException:
        pass
        
        
def final():
    main()
    #os.system('cd /home/raghav/catkin_ws/src/turtle/src')
    #os.system('python script.py')
    
moveBtn = tk.Button(root, text="Move the Robot to the above recorded location", command=lambda:final(),
                          font='customFont1', bg="azure", fg="black", width=40, wraplength=250)

moveBtn.grid(row=4, column=2)

root.mainloop()

#if __name__ == '__main__':
#    main()
