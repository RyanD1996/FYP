import pygame
import random
import numpy as np
import sys
FPS = 60
# Game window dimensions
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 420
GAME_HEIGHT=400
# Size of the paddle
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60

PADDLE_BUFFER= 15

# Size of the ball
BALL_WIDTH = 20
BALL_HEIGHT = 20

MAX_SCORE = 11

# Speed of ball and paddle objects
PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

# Colours for the paddles and ball
WHITE = (255,255,255)
BLACK = (0,0,0)
BROWN = (210,105,30)




def draw_ball(screen, x_pos, y_pos):
    ball = pygame.Rect(x_pos,y_pos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)

def draw_paddle_AI(screen, AI_paddle_y_position):
    # Paddle is located on the left side of the screen.
    paddle = pygame.Rect(PADDLE_BUFFER, AI_paddle_y_position, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen,WHITE, paddle)

def draw_paddle_user(screen, user_paddle_y_pos):
    # Paddle is located on the right side of the screen.
    paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, user_paddle_y_pos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen,BROWN, paddle)

def update_ball(AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction, episode_counter, done, misses, returns, AI_score, opponent_score):
    ball_x_pos = ball_x_pos + ball_x_direction * BALL_X_SPEED*3
    ball_y_pos = ball_y_pos + ball_y_direction * BALL_Y_SPEED*3
    score = 0.0

    # Check for a collision, if the ball hits the left side then switch direction.
    if((ball_x_pos <= (PADDLE_BUFFER + PADDLE_WIDTH)) and
            ((ball_y_pos + BALL_HEIGHT) >= AI_paddle_y_pos) and       # Check the ball is not below the paddle
            (ball_y_pos <= (AI_paddle_y_pos + PADDLE_HEIGHT)) and (ball_x_direction==-1)):       # Check ball is above the bottom of the paddle
        ball_x_direction = 1
        #print("Bot hits the ball")
        #score = 1.0
        returns += 1

    elif (ball_x_pos <=0):
        ball_x_direction = 1
        opponent_score = opponent_scores(opponent_score)
        score = -1.0
        done=True
        misses += 1
        ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction = reset_ball(ball_x_pos, ball_y_pos)
        AI_paddle_y_pos, user_paddle_y_pos = reset_paddles(AI_paddle_y_pos, user_paddle_y_pos)
        episode_counter += 1
        #if(AI_score == MAX_SCORE or opponent_score == MAX_SCORE):
            #done = True
        return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction, episode_counter, done, misses, returns, AI_score, opponent_score]

    if(ball_x_pos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and
            ball_y_pos + BALL_HEIGHT >= user_paddle_y_pos and
            ball_y_pos - BALL_HEIGHT <= user_paddle_y_pos + PADDLE_HEIGHT):
        ball_x_direction = -1

    elif(ball_x_pos >= WINDOW_WIDTH - BALL_WIDTH):
        #score = 10.0
        ball_x_direction = -1
        AI_score = AI_scores(AI_score)
        score = 1.0
        done=True
        print("Bot WINS!")
       # score = 1.0
        ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction = reset_ball(ball_x_pos, ball_y_pos)
        AI_paddle_y_pos, user_paddle_y_pos = reset_paddles(AI_paddle_y_pos, user_paddle_y_pos)
        episode_counter += 1
        #if(AI_score == MAX_SCORE or opponent_score == MAX_SCORE):
         #   done = True
        return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction, episode_counter, done, misses, returns, AI_score, opponent_score]

    if(ball_y_pos <=0):
        ball_y_pos = 0
        ball_y_direction = 1
    elif(ball_y_pos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ball_y_pos = WINDOW_HEIGHT - BALL_HEIGHT
        ball_y_direction = -1
    return [score, AI_paddle_y_pos, user_paddle_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction, episode_counter, done, misses, returns, AI_score, opponent_score]

def reset_ball(ball_x_pos, ball_y_pos):
    num = np.random.randint(0, 9)
    ball_x_pos = WINDOW_HEIGHT/2 - BALL_WIDTH/2
    # randomly decide where the ball will move
    if (0 <= num < 3):
        ball_x_direction = 1
        ball_y_direction = 1
    if (3 <= num < 5):
        ball_x_direction = -1
        ball_y_direction = 1
    if (5 <= num < 8):
        ball_x_direction = 1
        ball_y_direction = -1
    if (8 <= num < 10):
        ball_x_direction = -1
        ball_y_direction = -1
    # new random number
    num = np.random.randint(0, 9)
    # where it will start, y part
    ball_y_pos = num * (WINDOW_HEIGHT - BALL_HEIGHT)

    return ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction

def AI_scores(AI_score):
    AI_score += 1
    return AI_score

def opponent_scores(opponent_score):
    opponent_score += 1
    #print("Hello")
    return opponent_score

def reset_paddles(AI_paddle_y_pos, user_paddle_y_pos):
    # Initialise pos of paddles
    AI_paddle_y_pos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
    user_paddle_y_pos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
    return AI_paddle_y_pos, user_paddle_y_pos

def update_AI_paddle(action, AI_paddle_y_pos):
    # If action == move up
    if(action == 1):

        AI_paddle_y_pos -= PADDLE_SPEED*5
    # If action == move down
    elif(action == 2):

        AI_paddle_y_pos += PADDLE_SPEED*5
    elif(action == 3):

        AI_paddle_y_pos = AI_paddle_y_pos

    if(AI_paddle_y_pos < 0):
        AI_paddle_y_pos = 0
    if(AI_paddle_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        AI_paddle_y_pos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return AI_paddle_y_pos



def update_user_paddle(user_paddle_y_pos, ball_y_pos, user_action, ball_x_pos):
    if(user_action is None):
        #move down if ball is in upper half
        if(ball_x_pos > WINDOW_WIDTH/2):
            if ((user_paddle_y_pos + PADDLE_HEIGHT/2) < (ball_y_pos + BALL_HEIGHT/2)):
                user_paddle_y_pos = user_paddle_y_pos + PADDLE_SPEED*7.5
            #move up if ball is in lower half
            if(user_paddle_y_pos + PADDLE_HEIGHT/2 > ball_y_pos + BALL_HEIGHT/2):
                user_paddle_y_pos = user_paddle_y_pos - PADDLE_SPEED*7.5
            #don't let it hit top
            if (user_paddle_y_pos < 0):
                user_paddle_y_pos = 0
            #dont let it hit bottom
            if (user_paddle_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT):
                user_paddle_y_pos = WINDOW_HEIGHT - PADDLE_HEIGHT
            return user_paddle_y_pos
        else:
            return user_paddle_y_pos
    else:
        #move down if ball is in upper half
        if (user_action == 2):
            user_paddle_y_pos = user_paddle_y_pos + PADDLE_SPEED*7.5
        #move up if ball is in lower half
        if(user_action == 1):
            user_paddle_y_pos = user_paddle_y_pos - PADDLE_SPEED*7.5
        if(user_action == 0):
            user_paddle_y_pos = user_paddle_y_pos
        #don't let it hit top
        if (user_paddle_y_pos < 0):
            user_paddle_y_pos = 0
        #dont let it hit bottom
        if (user_paddle_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return user_paddle_y_pos


class Pong:
    def __init__(self, human_mode):
        self.human_mode = human_mode
        pygame.init()
        # Initialise Screen
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont("calibri", 20)
        # Random number between 0-9 for direction of ball
        self.AI_score = 0
        self.opponent_score = 0
        self.ball_x_direction = 0
        self.ball_y_direction = 0
        num = np.random.randint(0,9)
        self.done = False
        # Keep Score
        self.tally = 0
        self.return_rate = 0
        self.returns = 0
        self.misses = 0
        self.episode_counter =0
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
        self.screen.fill(BLACK)
        #Draw paddles
        draw_paddle_AI(self.screen,self.AI_paddle_y_pos)
        draw_paddle_user(self.screen, self.user_paddle_y_pos)
        #Draw ball
        draw_ball(self.screen, self.ball_x_pos, self.ball_y_pos)
        # Get pixels

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the window
        pygame.display.flip()
        # Return screen data
        return [image_data]

    def get_next_frame(self, action, user_action):
        pygame.event.pump()
        score = 0
        done = False
        self.screen.fill(BLACK)
        self.AI_paddle_y_pos = update_AI_paddle(action, self.AI_paddle_y_pos)
        draw_paddle_AI(self.screen, self.AI_paddle_y_pos)
        self.user_paddle_y_pos = update_user_paddle(self.user_paddle_y_pos, self.ball_y_pos, user_action, self.ball_x_pos)
        draw_paddle_user(self.screen, self.user_paddle_y_pos)
        [score, self.AI_paddle_y_pos, self.user_paddle_y_pos, self.ball_x_pos, self.ball_y_pos, self.ball_x_direction, self.ball_y_direction, self.episode_counter, done, misses, returns, self.AI_score, self.opponent_score] = update_ball(self.AI_paddle_y_pos, self.user_paddle_y_pos, self.ball_x_pos, self.ball_y_pos, self.ball_x_direction, self.ball_y_direction, self.episode_counter, self.done, self.misses, self.returns, self.AI_score, self.opponent_score)
        draw_ball(self.screen, self.ball_x_pos, self.ball_y_pos)

        self.misses = misses
        self.returns = returns

        exit_message = "Press Q to exit"
        #  Display Parameters
        if(not self.human_mode):
            returns_display = self.font.render("Returns: " + str(self.returns), True, (255, 255, 255))
            self.screen.blit(returns_display, (300., 400.))
            misses_display = self.font.render("Misses: " + str(self.misses), True, (255, 255, 255))
            self.screen.blit(misses_display, (400., 400.))
            TimeDisplay = self.font.render(" " + str(exit_message), True, (255, 255, 255))
            self.screen.blit(TimeDisplay, (10., 400.))
        if(self.human_mode):
            bot_score = self.font.render("Bot Score: " + str(self.AI_score), True, (255, 255, 255))
            self.screen.blit(bot_score, (200., 400.))
            human_score = self.font.render("Human Score: " + str(self.opponent_score), True, (255, 255, 255))
            self.screen.blit(human_score, (350., 400.))
            TimeDisplay = self.font.render(" " + str(exit_message), True, (255, 255, 255))
            self.screen.blit(TimeDisplay, (10., 400.))
        # Get pixels
        if(score>0.5 or score <-0.5):
            if(self.returns == 0):
                self.return_rate = 0
            else:
                self.return_rate = (self.returns/(self.misses + self.returns))*100
            self.tally = 0.1*score + self.tally*0.9
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the window
        pygame.display.flip()

        return [score,image_data, done]

