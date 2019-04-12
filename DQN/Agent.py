import random, math, json
import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import *
import tensorflow as tf
from collections import deque

possible_actions = 3 #[0:Stay, 1:Up, 2:Down]
stack_size = 4       # Stack size for input images (State representation)
observation_period = 2500
gamma = 0.975    # Discount factor on future rewards
batch_size = 64
memory_capacity = 2000

class Agent:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)
        self.model = self.create_model()
        self.memory = deque(maxlen=2000)
        self.steps = 0
        self.epsilon = 1.0
        self.epsilon_stop = 0.01
        self.epsilon_decay = 0.0001
        self.epsilon_start = 1.0

    '''
    Function for creating the sequential CNN model using the Keras library
    '''
    def create_model(self):
        print("Creating the CNN")
        model = Sequential()
        # CNN to predict the Q-values from 40x40x4 input stack of images.
        model.add(Conv2D(32, kernel_size=4, strides=(2,2), input_shape=(40,40,4), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=4, strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(units=possible_actions, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        print('Finished building the CNN')
        return model


    '''
    Function for loading the best model for the CNN
    '''
    def load_best_model(self):
        self.model.load_weights("DQN\BestWeights.h5")
        self.model.compile(loss='mse', optimizer='adam')
        self.epsilon = 0.0


    '''
    Find the best action given an input state representation, action depends
    on whether the agent is exploiting or exploring. 
    '''
    def find_best_action(self, state):
        threshold_value = random.random()
        if(threshold_value < self.epsilon or self.steps < observation_period):
            # Explore the state space by taking a random action
            action = random.randint(0, possible_actions-1)
            return action
        else:
            # Exploit the existing knowledge and use the CNN to predict an action.
            q_values = self.model.predict(state)
            # Model returns Q-values for each possible action, the best action is the one
            # with the largest Q-value.
            action = np.argmax(q_values)
            return action


    '''
    We need a function to return the best action given a state
    '''
    def return_best_action(self,state):
        q_values = self.model.predict(state)
        action = np.argmax(q_values)
        return action


    '''
    We need a function to add the current experience to the experience replay (memory)
    '''
    def add_experience(self, experience):
        # Experience is in (state, action, reward, new_state) format
        self.memory.append(experience)
        # If the memory reaches the max length (2000) then the deque will automatically
        # remove the oldest element and add the new element, so we don't have to worry
        # about handling that functionality.
        self.steps += 1
        self.reduce_epsilon()


    '''
    We need a function to decay the current value of epsilon
    '''
    def reduce_epsilon(self):
        if(self.steps > observation_period):
            self.epsilon = self.epsilon_stop + (self.epsilon_start - self.epsilon_stop) * np.exp(-self.epsilon_decay * self.steps)
        else:
            pass
    '''
                self.epsilon = 0.75
                if(self.steps > 7500):
                    self.epsilon = 0.5
                if(self.steps > 15000):
                    self.epsilon = 0.25
                if(self.steps > 30000):
                    self.epsilon = 0.15
                if(self.steps > 45000):
                    self.epsilon = 0.1
                if(self.steps > 75000):
                    self.epsilon = 0.05
            '''

    '''
    Function for training the agent using a minibatch from the Experience Replay (Memory)
    '''
    def minibatch_train(self):
        if(self.steps > observation_period):
            # We only want to train once the observation (explore) period is over.
            # First, retrieve a sample from the ER
            mb = random.sample(self.memory, batch_size)
            batch_length = len(mb)

            # Inputs = (batch size, frame height, frame width, stack size)
            inputs = np.zeros((batch_size, 40, 40, 4))
            targets = np.zeros((inputs.shape[0], possible_actions))
            for i in range(0, batch_length):
                state_mb = mb[i][0]
                action_mb = mb[i][1]
                reward_mb = mb[i][2]
                new_state_mb = mb[i][3]
                # Populate inputs
                inputs[i:i+1] = state_mb
                # Fill target
                targets[i] = self.model.predict(state_mb)
                q_stateaction = self.model.predict(new_state_mb)
                # If next state is terminal the reward  = reward_mb
                if(new_state_mb is None):
                    targets[i,action_mb] = reward_mb
                else:
                    # Predict the Q value
                    targets[i,action_mb] = reward_mb + gamma * np.max(q_stateaction)
            self.model.fit(inputs, targets, batch_size=batch_size, epochs=1, verbose=0)

    '''
    Function for saving the weights for the CNN
    '''
    def save_weights(self):
        print("Saving Weights")
        self.model.save_weights("ModelWeights.h5", overwrite=True)
        with open("PongModel.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    '''
    Function for saving the weights once we have solved the environment
    '''
    def save_best_weights(self):
        print("Saving Best Weights")
        self.model.save_weights("BestWeights.h5", overwrite=True)
        with open("BestPongModel.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)
