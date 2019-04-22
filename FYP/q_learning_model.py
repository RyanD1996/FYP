import numpy as np
import gym
import random
import string_recorder
from gym.monitoring import VideoRecorder

from matplotlib.figure import Figure

class QLearning():
    def __init__(self):
        self.progress = 0
        self.test_progress =0
        self.score_over_time = 0
        self.rewards = []
        self.epsilons = []
        self.rewards_during_testing = []
        self.best_episode = 0

    def create_q_table(self, env_name):
        # Create the environment
        self.env = gym.make(env_name)
        # Create variables for the number of possible actions
        # and the state space size.
        self.action_space_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
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
            self.progress += 1
            print("Progess: ", self.progress)
            if episode % 100 == 0:
                self.test_during_training()
            state = self.env.reset()  # Reset the environment
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
#                self.q_table[state, action] =(1- learning_rate)* self.q_table[state, action] + learning_rate * (
#                                reward + gamma * np.max(self.q_table[new_state, :]))# - self.q_table[state, action])

                # Update Q(s,a):= Q(s,a) + alpha*[reward + gamma*maxQ(s',a') - Q(s,a)]
                self.q_table[state, action] = self.q_table[state, action] + learning_rate * (
                                reward + gamma * np.max(self.q_table[new_state, :])- self.q_table[state,action])# - self.q_table[state, action])


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
        self.env.reset()
        self.rewards = []
        self.test_means=[]

        for episode in range(total_test_episodes):
            state = self.env.reset()
            total_rewards = 0
            self.recorder = string_recorder.StringRecorder()
            print("****************************")
            print("EPISODE {0} of {1}".format(episode, total_test_episodes))
            out_path = 'C:/Users/Ryan/PycharmProjects/FinalYearProject/records/episode{}.json'.format(episode)
            video = VideoRecorder(self.env, out_path)

            for step in range(max_steps):
                action = np.argmax(self.q_table[state, :])
                new_state, reward, done, info = self.env.step(action)
                video.capture_frame()
                total_rewards += reward                # If the taxi is penalised increase the penalty counter
                if done or step == max_steps-1:
                    if(total_rewards > self.best_episode):
                        self.best_episode = episode
                    self.rewards.append(total_rewards)
                    print("SUM: {}".format(sum(self.rewards)))
                    curr_eps = episode+1
                    self.test_means.append(sum(self.rewards) / curr_eps)
                    break
                state = new_state
            print("Episode Reward: {}".format(total_rewards))
            video.close()
            self.recorder.make_gif_from_gym_record(out_path)
            self.test_progress += 1
        self.env.close()
        print("Score over time: " + str(sum(self.rewards)/total_test_episodes))

    def random_agent(self, max_step):
        self.env.reset()
        #self.rewards = []
        state = self.env.reset()
        total_rewards = 0
        self.recorder = string_recorder.StringRecorder()
        out_path = 'C:/Users/Ryan/PycharmProjects/FinalYearProject/records/random.json'
        video = VideoRecorder(self.env, out_path)
        for step in range(max_step):
            #  frame = self.env.render()
            new_state, reward, done, info = self.env.step(self.env.action_space.sample())
            video.capture_frame()
            total_rewards += reward  # If the taxi is penalised increase the penalty counter
            state = new_state
        print("Episode Reward: {}".format(total_rewards))
        video.close()
        self.recorder.make_gif_from_gym_record(out_path)
        self.env.close()

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
        a.plot(x_axis, self.test_means)
        a.format_coord = lambda x, y: ''
        return a,f

    def plot_epsilon_decay(self, total_episodes):
        # PLOTTING
        f = Figure(figsize=(3,3), dpi=100)
        a = f.add_subplot(111)
        a.set_title("Epsilon Decay")
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
        a.set_title("Rewards During Training")
        print(len(self.epsilons))
        total_episodes = int(total_episodes / 100)
        x_axis = list(range(1, total_episodes+1))
        print(len(x_axis))
        y_axis = self.rewards_during_testing
        a.plot(x_axis, y_axis)
        a.format_coord = lambda x, y: ''
        return a,f



