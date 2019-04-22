import DQN.pong as pong
import DQN.Agent as PongAgent
import numpy
import warnings
warnings.filterwarnings('ignore')
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from collections import deque
import pygame
import pandas as pd
import csv
total_frames = 5000

def preprocessFrame(frame):
    grayscale = skimage.color.rgb2gray(frame)
    cropped = grayscale[0:400, 0:400]

    resized_img = skimage.transform.resize(cropped, (40,40), mode='reflect')
    resized_img = skimage.exposure.rescale_intensity(resized_img, out_range=(0,255))
    normalised_frame = resized_img/128
    return normalised_frame


def Play(human_mode):
    time = 0
    history = []
    game = pong.Pong(human_mode)
    agent = PongAgent.Agent()
    game.get_curr_frame()
    agent.load_best_model()
    action = 0
    user_action = 0
    [initialScore, initialFrame, done] = game.get_next_frame(action, user_action)
    initialProcessedFrame = preprocessFrame(initialFrame)
    state = numpy.stack((initialProcessedFrame,initialProcessedFrame,initialProcessedFrame,initialProcessedFrame), axis=2)
    state = state.reshape(1,state.shape[0],state.shape[1],state.shape[2])
    continue_playing = True
    # Main Loop
    while(time < total_frames and continue_playing):

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # As long as an arrow key is held down, the respective speed is set to 3 (or minus 3)
                if event.key == pygame.K_UP:
                    user_action = 1
                elif event.key == pygame.K_DOWN:
                    user_action = 2
                elif event.key == pygame.K_q:
                    continue_playing = False
                    pygame.display.quit()
                    pygame.quit()
                    break
            elif event.type == pygame.KEYUP:
                # As soon as an arrow key is released, reset the respective speed to 0
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    user_action = 0
            elif event.type == pygame.QUIT:
                    continue_playing = False
                    pygame.display.quit()
                    pygame.quit()
                    break

        if(continue_playing):
            action = 0
            action = agent.return_best_action(state)
            if(human_mode==False):
                user_action = None
           # print(user_action)
            [rtn_score, new_frame, done] = game.get_next_frame(action, user_action)
            new_processed_frame = preprocessFrame(new_frame)
            # Reshape for Keras
            new_processed_frame = new_processed_frame.reshape(1, new_processed_frame.shape[0], new_processed_frame.shape[1],1)
            next_state = numpy.append(new_processed_frame, state[:,:,:, :3],axis=3)
            state = next_state
            time +=1
            opp_score = game.opponent_score
            AI_score = game.AI_score
            if opp_score == 11 or AI_score ==11:
                df = pd.read_csv("human_vs_AI.csv")
                max = df['Game'].max()
                curr_game = max + 1
               # curr_game = max(int(l.split(',')[0]) for l in open("human_vs_AI.csv").readlines())
                #print(game)
                with open('human_vs_AI.csv', 'a', newline='') as fd:
                    fields = [curr_game, AI_score, opp_score]
                    writer = csv.writer(fd)
                    writer.writerow(fields)
                game.opponent_score = 0
                game.AI_score = 0

def main():
    Play(human_mode=False)

if __name__ =="__main__":
    main()