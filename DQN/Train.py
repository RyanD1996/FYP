import pong
import Agent as Agent
import numpy as np
import random
import pickle, os, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from collections import deque
import matplotlib.pyplot as plt
import sys
import cv2
training_time = 100000
stack_size = 4

'''
Function to preprocess the raw frame from PyGame.
'''
def preprocess_image(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   # grayscale = skimage.color.rgb2gray(frame)
    #cropped_image = grayscale[0:400, 0:400]
    reduced_image = cv2.resize(grayscale, (40,40))

   # reduced_image = skimage.transform.resize(grayscale, (40,40), mode='reflect')
   # reduced_image = skimage.exposure.rescale_intensity(reduced_image, out_range=(0,255))
    preprocessed_image = reduced_image/128
    return preprocessed_image

'''
Function for the main training loop
'''
def train_agent():
    step = 0
    print("Hello")
    # Deque is created to check the last three scores.
    score_check = deque(maxlen=3)
    continue_playing = True
    # Create an instance of the game
    game = pong.Pong()
    # Create an instance of the agent
    agent = Agent.Agent()
    # Initialise the value for the action variable [0:stay, 1:up, 2:down]
    action = 0
    # Perform an initial action, observe the reward and the new state.
    # user_action = None since the opponent is not human.
    [initial_score, first_frame] = game.get_next_frame(action, user_action=None)
    # Preprocess the first_frame
    initial_frame = preprocess_image(first_frame)
    # Input State = stack of 4 x initial images

    state = np.stack((initial_frame,initial_frame,initial_frame,initial_frame), axis=2)

    # We need to reshape the state since keras expects a format of (1x40x40x4)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])


    '''
    Main training loop
    '''
    while((step < training_time) and continue_playing):
        action = agent.find_best_action(state)

        # Action is performed on the environment, the score(reward) and resulting frame is returned
        [score, new_image] = game.get_next_frame(action, user_action=None)
        # New frame is preprocessed to reduce complexity
        new_frame = preprocess_image(new_image)
        # Preprocessed frame is reshaped to fit Keras format.
        new_frame = new_frame.reshape(1, new_frame.shape[0], new_frame.shape[1],1)
        # New frame is appended to the stack of states, removing the oldest frame.
        new_state = np.append(new_frame, state[:,:,:,:3], axis=3)


        # Add the experience to the experience replay buffer
        agent.add_experience((state, action, score, new_state))

        agent.minibatch_train()

        state = new_state

        step += 1

        if(step % 5000 == 0):
            # Save the weights
            agent.save_weights()
            print("Model saved")
        if(step % 200 == 0):
            print("Step: ", step," Score: ", "{0:.2f}".format(game.tally), " Epsilon: ", "{0:.4f}".format(agent.epsilon))
            score_check.append(game.tally)
            sum = 0.0
            for item in score_check:
                sum += item
            if (sum / 3) > 9.75:
                print("Good performance, saving the model.")
                agent.save_best_weights()
                continue_playing = False


def main():
    #
	# Main Method Just Play our Experiment
	train_agent()

	# =======================================================================
if __name__ == "__main__":
    main()