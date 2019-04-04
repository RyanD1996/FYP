import numpy as np
import gym
import random
import string_recorder
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from gym.monitoring import VideoRecorder

from matplotlib.figure import Figure

class QLearning():
    def __init__(self):

        #self.env = gym.wrappers.Monitor(self.env, "recording")
        self.score_over_time = 0

        self.rewards = []
        self.epsilons = []
        self.rewards_during_testing = []

    def create_q_table(self, env_name):
        self.env = gym.make(env_name)  # Make the environment

        # CREATE THE Q TABLE
        self.action_space_size = self.env.action_space.n  # Create a variable for the action space
        self.state_size = self.env.observation_space.n
        print('Action space size is {}'.format(self.action_space_size))
        print('State space size is {}'.format(self.state_size))
        # Q table, rows correspond to states and columns correspond to actions
        # We create a np array of zeros with shape 500x6
        self.q_table = np.zeros((self.state_size, self.action_space_size))

    def learn(self, total_episodes, max_steps, learning_rate, gamma, epsilon,min_epsilon, max_epsilon, decay_rate):
        '''
        THE Q LEARNING ALGORITHM
           - Initialise Q values (Q(s,a)) arbitrarily for all state-action pairs
           - For loop (until dead or learning has stopped)
               - Choose an action(a) in the current world state(s) based on the current Q value estimates (Q(s,.))
               - Take the action(a) and observe the outcome state(s') and reward(r)
               - Update Q(s,a):= Q(s,a) + alpha*[reward + gamma*maxQ(s',a') - Q(s,a)]
        '''
        self.epsilons = []
        for episode in range(total_episodes):  # For life or until learning stops.
            if episode % 1000 == 0:
                self.test_during_training()
            state = self.env.reset()  # Reset the environment
            step = 0  # Counter to track number of steps taken
            done = False  # Boolean denotes whether we are dead/episode is finished or not.
            print("Starting episode {}".format(episode))
            self.epsilons.append(epsilon)
            for step in range(max_steps):
                # Choose an action(a) in current world state(s)
                tradeoff = random.uniform(0, 1)
                # If tradeoff value is lower than the current value of epsilon, take a random action (EXPLORE)
                # Else, take the action that has the maximum Q value in the Q table for the current state (EXPLOIT).
                if tradeoff > epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = np.random.choice(self.action_space_size)

                # Take the action
                new_state, reward, done, info = self.env.step(action)

                # Update Q(s,a):= Q(s,a) + alpha*[reward + gamma*maxQ(s',a') - Q(s,a)]
                self.q_table[state, action] = self.q_table[state, action] + learning_rate * (
                                reward + gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                # Assign the new state as our current state for the next step.
                state = new_state

                if done == True:
                    break

            # Reduce epsilon
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        print("Learning complete.")

    def test_during_training(self):
        # During training, every 1000 episodes, USE Q_TABLE TO PLAY
        self.env.reset()
        state = self.env.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(99):
            # env.render()
            action = np.argmax(self.q_table[state, :])
            new_state, reward, done, info = self.env.step(action)
            total_rewards += reward  # If the taxi is penalised increase the penalty counter
            if done or step == 99 - 1:
                self.rewards_during_testing.append(total_rewards)
                break

            state = new_state
        print("Testing Episode Reward: {}".format(total_rewards))
        self.env.close()

    def test(self,max_steps, total_test_episodes):
        # TRAINING IS FINISHED, USE Q_TABLE TO PLAY
        self.env.reset()
        #env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)

        self.rewards = []
        for episode in range(total_test_episodes):
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0
            self.recorder = string_recorder.StringRecorder()
            print("****************************")
            print("EPISODE {0} of {1}".format(episode, total_test_episodes))
            out_path = 'C:/Users/Ryan/PycharmProjects/FinalYearProject/records/episode{}.json'.format(episode)
            video = VideoRecorder(self.env, out_path)

            for step in range(max_steps):
                #sys.exit()
                frame = self.env.render()
                #subprocess.call('clear',shell=False)
                action = np.argmax(self.q_table[state, :])
                new_state, reward, done, info = self.env.step(action)
                video.capture_frame()
                total_rewards += reward                # If the taxi is penalised increase the penalty counter
                if done or step == max_steps-1:
                    self.rewards.append(total_rewards)
                    break

                state = new_state
            print("Episode Reward: {}".format(total_rewards))
            video.close()
            self.recorder.make_gif_from_gym_record(out_path)
        self.env.close()
        print("Score over time: " + str(sum(self.rewards)/total_test_episodes))

    def plot(self, total_test_episodes):
        # PLOTTING
        f = Figure(figsize=(3,3), dpi=100)

        a = f.add_subplot(111)
        a.set_title("Rewards over {} test episodes" .format(total_test_episodes))
        print(len(self.rewards))

        x_axis = list(range(1,total_test_episodes+1))
        print(len(x_axis))
        y_axis = self.rewards

        a.plot(x_axis, y_axis)
        a.format_coord = lambda x, y: ''
        return a,f

    def plot_epsilon_decay(self, total_episodes):
        # PLOTTING
        f = Figure(figsize=(3,3), dpi=100)
        a = f.add_subplot(111)
        a.set_title("Epsilon decay over {} training episodes" .format(total_episodes))
        print(len(self.epsilons))

        x_axis = list(range(1, total_episodes+1))
        print(len(x_axis))
        y_axis = self.epsilons

        a.plot(x_axis, y_axis)
        a.format_coord = lambda x, y: ''
        return a,f

    def plot_training_rewards(self, total_episodes):
        # PLOTTING
        f = Figure(figsize=(3,3), dpi=100)
        a = f.add_subplot(111)
        a.set_title("Epsilon decay over {} training episodes" .format(total_episodes))
        print(len(self.epsilons))
        total_episodes = int(total_episodes / 1000)
        x_axis = list(range(1, total_episodes+1))
        print(len(x_axis))
        y_axis = self.rewards_during_testing

        a.plot(x_axis, y_axis)
        a.format_coord = lambda x, y: ''
        return a,f



