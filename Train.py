import pong as pong
import Agent as Agent
import pygame
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import sys
import csv
import cv2
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
import time
training_time = 100000
stack_size = 4
total_episodes = 1000000
'''
Function to preprocess the raw frame from PyGame.
'''
def preprocess_image(frame):
    # Use cv2 to convert input frame from RGB to grayscale
    #grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayscale = skimage.color.rgb2gray(frame)
    # Crop the image
    cropped_image = grayscale[0:400, 0:400]
    #cv2.imshow("d",cropped_image)
    #time.sleep(1000)

    # Use cv2 to resize the image to 40x40
    #reduced_image = cv2.resize(cropped_image, (40,40))
    reduced_image = skimage.transform.resize(cropped_image, (40,40), mode='reflect')
    reduced_image = skimage.exposure.rescale_intensity(reduced_image, out_range=(0,255))
    # Divide by 128 so each pixel is represented in the range 0 - 1
    preprocessed_image = reduced_image / 128
    return preprocessed_image

'''
Function for the main training loop
'''
def train_agent():
    step = 0
    episode = 0
    # Deque is created to check the last three scores.
    score_check = deque(maxlen=3)
    continue_playing = True
    # Create an instance of the game
    game = pong.Pong()
    # Create an instance of the agent
    agent = Agent.Agent()
    reward_list = []
    return_rates = deque(maxlen=10)
    STEP_COUNTER = 0

    epsilon_history = []
    test_during_training_rewards = []
    mean_score = 0
    record_stats = False
    '''
    Main training loop
    '''

    while((episode < total_episodes) and continue_playing):
        done = False
        episode += 1
        step = 0
        episode_rewards = []
        # Initialise the value for the action variable [0:stay, 1:up, 2:down]
        action = 0
        # Perform an initial action, observe the reward and the new state.
        # user_action = None since the opponent is not human.
        [initial_score, first_frame, done] = game.get_next_frame(action, user_action=None)
        #[first_frame] = game.get_next_frame(action, user_action=None)

        initial_frame = preprocess_image(first_frame)
        state = np.stack((initial_frame, initial_frame, initial_frame, initial_frame), axis=2)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        target_updated = False
        while(step < 1000000):
            STEP_COUNTER +=1
            step+=1
            action = agent.find_best_action(state)
            # Action is performed on the environment, the score(reward) and resulting frame is returned
            [score, new_image, done] = game.get_next_frame(action, user_action=None)
            if(score > 0 or score < 0):
                pass
            if done:
                if game.opponent_score == 1:
                    rewards = (game.returns / (game.opponent_score + game.returns)) * 100
                    #if game.returns == 0:
                   # rewards = -1
                else:
                    rewards = (game.returns / (game.opponent_score + game.returns))*100
                    #rewards = 1
                episode_rewards.append(rewards)

                if(episode%10 ==0):
                    test_during_training_rewards.append(rewards)


                print("Episode Rewards: {}" .format(rewards))

                new_frame = preprocess_image(new_image)
                new_frame = new_frame.reshape(1, new_frame.shape[0], new_frame.shape[1], 1)

                new_state = np.append(new_frame, state[:, :, :, :3], axis=3)
                agent.add_experience((state, action, score, new_state, done))

                # Append the current value of epsilon to the list for plotting
                epsilon_history.append(agent.epsilon)

                # Append the percentage return for the episode to the list
                reward_list.append((episode, rewards, game.returns ,mean_score, STEP_COUNTER))

                mean_score = [item[1] for item in reward_list[-10:]]
                mean_score = np.mean(mean_score)


                if(record_stats == True):
                    with open('training_statistics.csv', 'w') as out:
                        csv_out = csv.writer(out)
                        csv_out.writerow(['Episode', 'Reward','Episode_Returns', 'Mean_Score_Over_Last_10', 'Global_Step_Counter'])
                        for row in reward_list:
                            csv_out.writerow(row)
                    record_stats = False


                print("Episode: ", episode," Global Step Counter: ", STEP_COUNTER, "\nBot Returns: ", game.returns, " Opponent Score: ", game.opponent_score,"\nMean Score over last 10: ", mean_score, " Epsilon: ",
                      "{0:.4f}".format(agent.epsilon),"\n\n")
                game.returns = 0
                game.opponent_score = 0
                if (np.mean(mean_score > 95)):
                    print("Good performance, saving the model.")
                    agent.save_best_weights()
                    continue_playing = False
                #if(STEP_COUNTER % 10000 == 0):
                #    agent.minibatch_train(update_target=True)
                #else:
                #    agent.minibatch_train(update_target=False)
                agent.minibatch_train(update_target=False)
                break
            else:
                new_frame = preprocess_image(new_image)
                 # Preprocessed frame is reshaped to fit Keras format.
                new_frame = new_frame.reshape(1, new_frame.shape[0], new_frame.shape[1],1)
                # New frame is appended to the stack of states, removing the oldest frame.
                new_state = np.append(new_frame, state[:,:,:,:3], axis=3)
                # Add the experience to the experience replay buffer
                agent.add_experience((state, action, score, new_state, done))

                # Append the current value of epsilon to the list for plotting
                epsilon_history.append(agent.epsilon)
                state = new_state
                #agent.minibatch_train(update_target=False)
                if(STEP_COUNTER % 1000 == 0):
                    record_stats = True
                agent.minibatch_train(update_target=False)

        if(episode % 10 == 0):
            # Save the weights
            agent.save_weights(episode)
            print("Model saved")




def main():
	train_agent()

if __name__ == "__main__":
    main()