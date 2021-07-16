import cv2
import gym
from gym import spaces
import numpy as np
import os
import time
import random
from src.utils.training_tools import OBSERVATION_FILE, IMAGE_SIZE, SQUARE_SIZE_X, \
                                    SQUARE_SIZE_Y, STEP_X, STEP_Y, ERROR, FINAL_X, FINAL_Y, \
                                    in_range, pos_to_state, state_to_pos


from src.utils.useful_functions import is_modified, calculate_center

import time

class RobotEnv(gym.Env):
    """
    Gym environment for the Teresa robot
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot, client):
        """Constructor of environment

        Args:
            robot (RobotController): Controller of the robot, either real or simulation
            client (ROSConnection): Connection with the ROS server
        """
        super(RobotEnv, self).__init__()
        self.robot = robot
        self.state = 0
        self.real_position = (0, 0, 0, 0) #Reset real position
        self.action_space = spaces.Discrete(robot.NUMBER_MOVEMENTS)
        self.set_up_parameters_training()

        if os.path.exists(OBSERVATION_FILE):
            self.old_file = os.stat(OBSERVATION_FILE).st_mtime
        else:
            self.old_file = -1


    def takeoff(self):
	    self.robot.takeoff()
    def land(self):
	    self.robot.land()


    def object_in_place(self, x, y, w, h):
        """Method that calculates if the object is in the desire position
        Parameters:
        -   x (Float) --> x coordinate of the object position
        -   y (Float) --> y coordinate of the object position
        -   w (Float) --> width of the object detected (NOT USED YET)
        -   h (Float) --> height of the object detected (NOT USED YET)
        Returns
        -   True  --> if the face is in place
        -   False --> otherwise
        """
        if in_range(x, self.IMAGE_SIZE[0]/2, self.ERROR):
            return True
        return False
    
    def get_state(self):
        self.update_state_image() # Process image
        object_locations = self.define_state() # Define state
        done = 0

        if (len(object_locations) > 0):
            x, y, w, h = object_locations[0]
            self.state = [x] 
            self.real_position = (x, y, w, h)
            if self.object_in_place(x, y, w, h):
                done = 1
        else
            self.state = [-1] 
            self.real_position = (0, 0, 0, 0)
        
        return [self.state], done




    def step(self, action, time):
        """Execute a command into the robot and retrieve
        a picture of the environment

        Args:
            action (Integer): Action to execute

        Returns:
            List: a list containing the new state, the reward obtained from the step
            and if it finish doing the task
        """
        reward = 0 # Reward of the state
        done = 0 # Boolean that indicates that an episode has finished

        self.robot.move_robot(action) # Execute Move
        self.update_state_image() # Process image
        object_locations = self.define_state() # Define state

        if (len(object_locations) > 0):
            x, y, w, h = object_locations[0]
            self.state = [x] #pos_to_state(x, y)
            self.real_position = (x, y, w, h)
            #--------The code below will change------
            if self.object_in_place(x, y, w, h):# and self.state <= self.NB_STATES:
                reward = 100 / time
                done = 1

            else:
                reward = (2 - (2*abs(x - self.IMAGE_SIZE[0]/2)/self.IMAGE_SIZE[0]))/time
            #----------------------------------------
        else:
            self.state = [-1] #pos_to_state(0, 0) #Reset state
            self.real_position = (0, 0, 0, 0)
            done = 0
            reward = -0.5

        return [self.state], reward, done

    def reset(self):
        """Reset the simulation putting the object to
        follow in a random position

        Returns:
            Integer: The state after the reset
        """
        #self.robot.reset_simulation()
        random_number = random.uniform(-1.0, 1.0)
        #self.gzcontroller.set_position(random_number)


        self.robot.move_robot(0) # Execute Move
        self.update_state_image() # Process image
        object_locations = self.define_state() # Define state

        for i in range(4):
            self.robot.move_robot(1 + random.randint(0,1)) # Execute Move
            self.update_state_image() # Process image
            object_locations = self.define_state() # Define state

        while (len(object_locations) == 0):
            self.robot.move_robot(1) # Execute Move
            self.update_state_image() # Process image
            object_locations = self.define_state() # Define state


        x, y, w, h = object_locations[0]
        self.state = [x]

        return [self.state]

    def render(self, mode='human'):
        """Render an image that represents the state in a moment of the simulation

        Args:
            mode (str, optional). Defaults to 'human'.

        Returns:
            Integer
        """
       

        image = cv2.imread(OBSERVATION_FILE)
        window_name = 'image'
       # x, y = state_to_pos(self.state)
        w = self.SQUARE_SIZE_X
        h = self.SQUARE_SIZE_Y

        #cv2.rectangle(image, (x, y), (x+w, y+h), (25, 125, 225), 5)
        xreal, yreal, wreal, hreal = self.real_position
        cv2.rectangle(image, (xreal, yreal), (xreal+wreal, yreal+hreal), (255, 0, 0), 5)

        cv2.imshow(window_name, image)
        value = 5
        imgcpy = image.copy()

        img = cv2.resize(imgcpy, None, fx=0.5, fy=0.5)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        return 0

    def update_state_image(self):
        """In here we identify if the object we want to follow is in sight or no. This is
        used to calculate the reward
        """
        time_passed = 0
        while not os.path.exists(OBSERVATION_FILE) or not is_modified(self.old_file, os.stat(OBSERVATION_FILE).st_mtime): # Wait until the file exists
            time.sleep(1)
            if time_passed == 10:
                break
            time_passed += 1

        self.old_file = os.stat(OBSERVATION_FILE).st_mtime # In here we get the time we got the picture

    def define_state(self):
        """Apply the image recognition function to the state

        Returns:
            List<Tuples>: list of tuples where each element is a tuple describing 
            the position
        """
        body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')
        image = cv2.imread(OBSERVATION_FILE)
        return body_classifier.detectMultiScale(image, 1.2, 3)

    def set_up_parameters_training(self):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.ERROR = ERROR
    
        self.SQUARE_SIZE_X = SQUARE_SIZE_X # This is the step in the X axis
        self.SQUARE_SIZE_Y = SQUARE_SIZE_Y # This is the step in the Y axis
    
        self.FINAL_X, self.FINAL_Y = calculate_center(self.IMAGE_SIZE, self.SQUARE_SIZE_X, self.SQUARE_SIZE_Y)
    
        self.STEP_X = STEP_X # It moves the square 20 pixeles in the X axis
        self.STEP_Y = STEP_Y # It moves the square 40 pixeles in the Y axis
    
        MAX_X = int(((self.IMAGE_SIZE[0] - self.SQUARE_SIZE_X) / self.STEP_X) + 1)
        MAX_Y = int(((self.IMAGE_SIZE[1] - self.SQUARE_SIZE_Y) / self.STEP_Y) + 1)
        self.NB_STATES = MAX_X*MAX_Y

    def close(self):
        pass
