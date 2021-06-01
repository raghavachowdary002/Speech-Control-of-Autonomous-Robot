# Speech-Control-of-Autonomous-Robot
**Designed a Graphical User Interface(GUI) and integrated with a Deep Learning model and ROS nodes to move the robot.**

This project goal is to design a moving robot which takes a voice command given by a human and performs the action accordingly. The speech recognition framework refers to a system where a person can talk to a computer through a microphone. The computer converts the words into text or commands to perform computer functions. The intelligent speech recognition system makes spoken instructions understandable by the robot. The speech-recognition system is trained to recognize given commands and the programmed robot navigates through the speech commands based on the instructions. The results prove that the proposed robot is capable of understanding the meaning of speech commands given as a recorded input. This robot simulation has been carried in ROS middleware and the map has been designed in Gazebo.

# Graphical user interface (GUI)

![image](https://user-images.githubusercontent.com/84661500/120316612-e75b1880-c2dd-11eb-9f79-631fb7335ac9.png)

# Gazebo Map

![image](https://user-images.githubusercontent.com/84661500/120316812-1d000180-c2de-11eb-92d3-da54c5d2df77.png)

# Project Implementaion

The project implementation can be separated into four steps.
1.	GUI Creation
2.	Preprocessing of data
3.	Design of Deep Learning Model
4.	Gazebo Map Design

**1.	GUI Creation:** This section gives a brief overview over the architecture and implementation of the Graphical User Interface (GUI). We have designed the GUI as shown in the Figure that allow the user to allow a prerecorded audio file and the user can actually see the text form of the prerecorded command by clicking on a button. And finally, the user can move the robot to the desired location by clicking on a button according to the voice command loaded. GUI is built by using the Tkinter framework. It is a cross- platform framework that’s built into the Python standard library.

**2.	Preprocessing of data:** The first phase in the speech recognition process in the development of an ASR system is to distinguish the voiced or unvoiced signal and generate feature vectors. Pre-processing adapts or alters the speech signal to make it acceptable for the analysis of feature extraction. In speech signal processing, the signal should mainly be examined if it is corrupted by the background or ambient noise. MFCCs are the Mel Frequency Cepstral Coefficients. MFCC takes into account human perception for sensitivity at appropriate frequencies by converting the conventional frequency to Mel Scale, and are thus suitable for speech recognition tasks quite well (as they are suitable for understanding humans and the frequency at which humans speak/utter). MFCCs vector for each audio file is extracted by a predefined library called librosa. The code snippet is given in Figure 5. The gist of the code is to loop through the whole dataset and take a voice signal in each iteration and by predefining the sample rate we can drop the audio signals which does not have the predecided number of samples. Next step is to extract the MFCC vector by passing the number of coefficients to extract, interval to apply FFT, hoplength which is slicing window for FFT. The extracted MFCC vectors are all saved into a Json file. Each audio file is mapped with a MFCC vector so that it is easy to train the deep learning model to train a digital data rather than analog data. Next step is to design a model and train the model by splitting the dataset.

**3.	Design of Deep Learning Model:** The deep learning model is build using TensorFlow which is a predefined library. The training data is loaded as a json file which we have saved in the preprocessing step. The Json file consists of labels and MFCC coefficients of each audio signal. The loaded dataset is split by using a train_test_split function from sklearn.  The dataset is divided into train, test, validation data. The test and validation are of each 20% of whole dataset and the remaining goes into training data. The code snippet of the model architecture is given in the below figure. The architecture involves three convolutional layers, one dense layer and one softmax output layer. The loss function is chosen as sparse categorical cross entropy and the learning rate is fixed at 0.001. Now the model is to be trained. This can be done by fit function. This fit function takes the argument of X and Y labels of dataset and number of epochs, batch size, and the X, Y labels of validation dataset. The trained model is saved as a .h5 file. This saved model file is loaded into the prediction method and the test data will be predicted with the help of saved model file. This saved model file contains all the information about the model architecture and the information about the training data. We can save lot of time during the time of prediction by saving a model otherwise it will take huge amount of time to train the model and then use it for prediction.

**4.	Gazebo Map Design:** The Gazebo simulator materializes the behavior of the physical robot in a virtual scenario. In order to start a world simulation for a specific exercise, a configuration file that determines the scenario, the robots involved, etc. is created and used. Gazebo provides a local viewer to observe the evolution of the simulated world, including the behavior of the robot programmed by the user and also allowing interaction with the scene by adding certain element at runtime, stopping, re-launching the simulation, etc. We designed a gazebo world file which involves a house with different rooms in which a robot can move from one room to another room with a recorded input. The rooms and the commands to each room are given below.

| **Room Name**  | **Command to the Room** |
| ------------- | ------------- |
| Room One  | Move to Room One/ Go to Room One  |
| Room Two  | Move to Room Two/Go to Room Two  |
| Origin / Home Position | Move to Origin/Go to Origin/Move to Home Position/Go to Home Position |
| Living Room | Move to Living Room/Go to Living Room |
| Kitchen | Move to Kitchen/Go to Kitchen |         

As shown above in the map we have created four rooms and an origin position. The turtlebot is placed at     the origin position initially and it can be moved from one room to another room by a recorded voice command. The code for this is written in Python Programming Language. As mentioned in the GUI section to move the robot from one room to another room it is recommended to first load the audio file and then click the Move the Robot button to the desired location. Navigation and Simulation: After performing the map generation in Gazebo, we used the Odometry package, euler_from_quaternion, quaternion_from_euler, Twist Package, P controller for the controlled rotation of turtlebot for navigation inside the generated map. 
