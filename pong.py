import pygame
import random
import numpy as np
import sys
FPS = 60
# Game window dimensions
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

# Size of the paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

PADDLE_BUFFER= 10

# Size of the ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

# Speed of ball and paddle objects
PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

# Colours for the paddles and ball
WHITE = (255,255,255)
BLACK = (0,0,0)

#Initialise Screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

def draw_ball(x_pos, y_pos):
    ball = pygame.Rect(x_pos,y_pos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)

def draw_paddle_AI(AI_paddle_y_position):
    # Paddle is located on the left side of the screen.
    paddle = pygame.Rect(PADDLE_BUFFER, AI_paddle_y_position, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen,WHITE, paddle)

def draw_paddle_user(user_paddle_y_pos):
    # Paddle is located on the right side of the screen.
    paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER, user_paddle_y_pos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen,WHITE, paddle)

def update_ball(AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction):
    ball_x_pos = ball_x_pos + ball_x_direction * BALL_X_SPEED
    ball_y_pos = ball_y_pos + ball_y_direction * BALL_Y_SPEED
    score = 0

    # Check for a collision, if the ball hits the left side then switch direction.
    if(ball_x_pos <= PADDLE_BUFFER + PADDLE_WIDTH and
            ball_y_pos + BALL_HEIGHT >= AI_paddle_y_pos and
            ball_y_pos - BALL_HEIGHT <= AI_paddle_y_pos + PADDLE_HEIGHT):
        ball_x_direction = 1
    elif (ball_x_pos <=0):
        ball_x_direction = 1
        score = -1
        return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]

    if(ball_x_pos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and
            ball_y_pos + BALL_HEIGHT >= user_paddle_y_pos and
            ball_y_pos - BALL_HEIGHT <= user_paddle_y_pos + PADDLE_HEIGHT):
        ball_x_direction = -1
    elif(ball_x_pos >= WINDOW_WIDTH - BALL_WIDTH):
        ball_x_direction = -1
        score = 1
        return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]

    if(ball_y_pos <=0):
        ball_y_pos = 0
        ball_y_direction = 1
    elif(ball_y_pos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ball_y_pos = WINDOW_HEIGHT - BALL_HEIGHT
        ball_y_direction = -1
    return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]

def update_AI_paddle(action, AI_paddle_y_pos):
    # If action == move up
    if(action == 1):
        print('move up')
        AI_paddle_y_pos -= PADDLE_SPEED
    # If action == move down
    elif(action == 2):
        print('move down')
        AI_paddle_y_pos += PADDLE_SPEED
    elif(action == 3):
        print('stay still')
        AI_paddle_y_pos = AI_paddle_y_pos

    if(AI_paddle_y_pos < 0):
        AI_paddle_y_pos = 0
    if(AI_paddle_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        AI_paddle_y_pos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return AI_paddle_y_pos

def update_user_paddle(user_paddle_y_pos, ball_y_pos):
    #move down if ball is in upper half
    if ((user_paddle_y_pos + PADDLE_HEIGHT/2) < (ball_y_pos + BALL_HEIGHT/2)):
        user_paddle_y_pos = user_paddle_y_pos + PADDLE_SPEED
    #move up if ball is in lower half
    if(user_paddle_y_pos + PADDLE_HEIGHT/2 > ball_y_pos + BALL_HEIGHT/2):
        user_paddle_y_pos = user_paddle_y_pos - PADDLE_SPEED
    #don't let it hit top
    if (user_paddle_y_pos < 0):
        user_paddle_y_pos = 0
    #dont let it hit bottom
    if (user_paddle_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return user_paddle_y_pos

class Pong:
    def __init__(self):
        # Random number between 0-9 for direction of ball
        num = np.random.randint(0,9)
        print(num)

        # Keep Score
        self.tally = 0
        # Initialise pos of paddles
        self.AI_paddle_y_pos = WINDOW_HEIGHT /2 - PADDLE_HEIGHT /2
        self.user_paddle_y_pos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        # Ball direction
        #self.ball_x_direction =1
        #self.ball_y_direction =1
        # Starting point
        self.ball_x_pos = WINDOW_HEIGHT/2 - BALL_WIDTH/2

        #randomly decide where the ball will move
        if(0 < num < 3):
            self.ball_x_direction = 1
            self.ball_y_direction = 1
        if (3 <= num < 5):
            self.ball_x_direction = -1
            self.ball_y_direction = 1
        if (5 <= num < 8):
            self.ball_x_direction = 1
            self.ball_y_direction = -1
        if (8 <= num < 10):
            self.ball_x_direction = -1
            self.ball_y_direction = -1
        #new random number
        num = np.random.randint(0,9)
        #where it will start, y part
        self.ball_y_pos = num*(WINDOW_HEIGHT - BALL_HEIGHT)/9

    def get_curr_frame(self):
        # For each frame, call event queue
        pygame.event.pump()
        #Paint background
        screen.fill(BLACK)
        #Draw paddles
        draw_paddle_AI(self.AI_paddle_y_pos)
        draw_paddle_user(self.user_paddle_y_pos)
        #Draw ball
        draw_ball(self.ball_x_pos, self.ball_y_pos)
        # Get pixels
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the window
        pygame.display.flip()
        # Return screen data
        return image_data

    def get_next_frame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        self.AI_paddle_y_pos = update_AI_paddle(action, self.AI_paddle_y_pos)
        draw_paddle_AI(self.AI_paddle_y_pos)
        self.user_paddle_y_pos = update_user_paddle(self.user_paddle_y_pos, self.ball_y_pos)
        draw_paddle_user(self.user_paddle_y_pos)
       # print(self.ball_x_direction)
        #print("IF 1")
        #sys.exit()
        #print(update_ball(self.AI_paddle_y_pos, self.user_paddle_y_pos, self.ball_x_pos, self.ball_y_pos, self.ball_x_direction, self.ball_y_direction))
        #sys.exit()
        [score, self.AI_paddle_y_pos, self.user_paddle_y_pos, self.ball_x_pos, self.ball_y_pos, self.ball_x_direction, self.ball_y_direction] = update_ball(self.AI_paddle_y_pos, self.user_paddle_y_pos, self.ball_x_pos, self.ball_y_pos, self.ball_x_direction, self.ball_y_direction)
        draw_ball(self.ball_x_pos, self.ball_y_pos)
        # Get pixels
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the window
        pygame.display.flip()
        self.tally = self.tally + score
        # Return screen data
        return [score,image_data]