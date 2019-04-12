import pong
import DQN.Agent as PongAgent
import numpy
import warnings
warnings.filterwarnings('ignore')
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from collections import deque
import pygame

total_frames = 5000

def preprocessFrame(frame):
    grayscale = skimage.color.rgb2gray(frame)
    cropped = grayscale[0:400, 0:400]

    resized_img = skimage.transform.resize(cropped, (40,40), mode='reflect')
    resized_img = skimage.exposure.rescale_intensity(resized_img, out_range=(0,255))
    normalised_frame = resized_img/128
    return normalised_frame


def Play():
    time = 0
    history = []
    game = pong.Pong()
    agent = PongAgent.Agent()
    game.get_curr_frame()
    agent.load_best_model()
    action = 0
    user_action = 0
    [initialScore, initialFrame] = game.get_next_frame(action, user_action)
    initialProcessedFrame = preprocessFrame(initialFrame)
    state = numpy.stack((initialProcessedFrame,initialProcessedFrame,initialProcessedFrame,initialProcessedFrame), axis=2)
    state = state.reshape(1,state.shape[0],state.shape[1],state.shape[2])

    # Main Loop
    while(time < total_frames):

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # As long as an arrow key is held down, the respective speed is set to 3 (or minus 3)
                if event.key == pygame.K_UP:
                    user_action = 1
                elif event.key == pygame.K_DOWN:
                    user_action = 2
            elif event.type == pygame.KEYUP:
                # As soon as an arrow key is released, reset the respective speed to 0
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    user_action = 0
        action = 0
        action = agent.return_best_action(state)
        [rtn_score, new_frame] = game.get_next_frame(action, user_action)
        new_processed_frame = preprocessFrame(new_frame)
        # Reshape for Keras
        new_processed_frame = new_processed_frame.reshape(1, new_processed_frame.shape[0], new_processed_frame.shape[1],1)
        next_state = numpy.append(new_processed_frame, state[:,:,:, :3],axis=3)
        state = next_state
        time +=1

def main():
    Play()

if __name__ =="__main__":
    main()