import tkinter as tk
from tkinter.ttk import Progressbar as Progressbar, Button, Entry, Label
import DQN.PlayAgainstAgent2 as Play
from DQN.q_learning_model import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
from itertools import count
from threading import Thread
import queue
import matplotlib.animation as animation
import sys
import pandas as pd
class GUI(tk.Tk):
    def __init__(self):

        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.title("Reinforcement Learning Environment")

        self.frames = {}
        for F in (StartPage,PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # Put all pages in same location
            # One on top of stacking order will be visible
            frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        # Show a frame for the given page name
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller
        top_frame = tk.Frame(self)
        top_frame.grid(row=0, sticky='ew')
        middle_frame = tk.Frame(self)
        bottom_frame = tk.Frame(self)
        bottom_frame.grid(row=2,sticky='ew')
        self.grid_rowconfigure(1, weight=1)

        label1 = tk.Message(top_frame, text="This is a tool where you can train your own Q-Learning agent in two OpenAI Gym environments using your own hyperameters.\n\n", width=300)
        label1.grid(row=0, column=0)


        label2 = tk.Message(top_frame, text=" Alternatively, you can play against a Pong agent that has learned to play the game by observing the pixels and the rewards it receives, or simply watch it play against a hardcoded AI.", width=300)


        label2.grid(row=3, column=0)
        train_q_learning_btn = Button(top_frame, text='Train OpenAI gym using Q Learning', command= lambda: controller.show_frame("PageOne"))
        train_q_learning_btn.grid(row=0,column=2)

        train_dqn_learning_btn = Button(top_frame, text='Train Atari using Deep Q Learning', command= lambda: controller.show_frame("PageTwo"))
        train_dqn_learning_btn.grid(row=3,column=2)

        quit_btn = Button(bottom_frame, text='Quit', command=self.quit)
        quit_btn.grid(row=1)

class PageOne(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        self.queue = queue.Queue()
        self.thread = None
        self.in_work = False


        tk.Frame.__init__(self, parent)
        self.controller = controller
        heading = tk.Label(self, text='Enter the hyperparameters you want to train the agent with')
        heading.config(font=("Arial", 10))
        heading.grid(row=0)
        # Select the environment to train/test with.
        environment_label = tk.Label(self, text='Environment')
        environment_label.config(font=("Arial", 9))
        environment_label.grid(row=1)
        self.env_frozenlake_btn = Button(self, text='Frozen Lake', command= self.select_frozenlake)
        self.env_frozenlake_btn.grid(row=1, column=1)

        self.env_taxi_btn = Button(self, text='Taxi', command= self.select_taxi)
        self.env_taxi_btn.grid(row=1, column=2)

        # Entry for total number of episodes to train on
        total_eps_label = Label(self, text='Total episodes to train')
        total_eps_label.config(font=("Arial", 9))
        total_eps_ttp = CreateToolTip(total_eps_label, "Define the total number of episodes the agent should complete.")
        total_eps_label.grid(row=2)
        self.total_eps_entry = Entry(self)
        self.total_eps_entry.insert(0, '3000')
        self.total_eps_entry.config(state='disabled')
        self.total_eps_entry.grid(row=2, column=1)


        # Entry for total number of test episodes to train on
        total_test_eps_label = tk.Label(self, text='Total episodes to test')
        total_test_eps_label.config(font=("Arial", 9))
        total_test_eps_ttp = CreateToolTip(total_test_eps_label, "Define the total number of episodes the agent should be tested on once it has completed training.")
        total_test_eps_label.grid(row=3)
        self.total_test_eps_entry = Entry(self)
        self.total_test_eps_entry.insert(0, '100')
        self.total_test_eps_entry.config(state='disabled')
        self.total_test_eps_entry.grid(row=3, column=1)

        # Entry for learning rate (alpha)
        learning_rate_label = tk.Label(self, text='Learning Rate')
        learning_rate_label.config(font=("Arial", 9))
        learning_rate_label.grid(row=5)
        self.learning_rate_entry = Entry(self)
        self.learning_rate_entry.insert(0, '0.7')
        self.learning_rate_entry.config(state='disabled')
        self.learning_rate_entry.grid(row=5, column=1)

        # Entry for max steps per episode
        max_steps_label = tk.Label(self, text='Maximum steps per epsiode')
        max_steps_label.config(font=("Arial", 9))
        max_steps_ttp = CreateToolTip(max_steps_label, "The maximum number of actions the agent can take in a single episode.")
        max_steps_label.grid(row=4)
        self.max_steps_entry = Entry(self)
        self.max_steps_entry.insert(0, '99')
        self.max_steps_entry.config(state='disabled')
        self.max_steps_entry.grid(row=4, column=1)

        # Entry for Gamma
        gamma_label = tk.Label(self, text='Gamma (Discount Rate)')
        gamma_label.config(font=("Arial", 9))
        discount_ttp = CreateToolTip(gamma_label, "Define the discount rate that should be applied to future rewards (Range 0-1). \nA value of 0 will prioritise immediate rewards whereas 1 will weight rewards evenly.")
        gamma_label.grid(row=6)
        self.gamma_entry = Entry(self)
        self.gamma_entry.insert(0, '0.618')
        self.gamma_entry.config(state='disabled')
        self.gamma_entry.grid(row=6, column=1)

        # Entry for Epsilon
        epsilon_label = tk.Label(self, text='Starting value for Epsilon')
        epsilon_label.config(font=("Arial", 9))
        epsilon_ttp = CreateToolTip(epsilon_label,"The probability that a random action will be taken (Range 0-1).\nThe higher the value of epsilon the higher the probability that the agent will take a random action.\nEpsilon is decayed throughout training to reduce the number of random actions.")
        epsilon_label.grid(row=2, column=2)
        self.epsilon_entry = Entry(self)
        self.epsilon_entry.insert(0, '1.0')
        self.epsilon_entry.config(state='disabled')
        self.epsilon_entry.grid(row=2, column=3)

        # Max Epsilon
        max_epsilon_label = tk.Label(self, text='Starting value for Epsilon')
        max_epsilon_label.config(font=("Arial", 9))
        #max_epsilon_label.grid(row=3, column=2)
        self.max_epsilon_entry = Entry(self)
        self.max_epsilon_entry.insert(0, '1.0')
        self.max_epsilon_entry.config(state='disabled')
        #self.max_epsilon_entry.grid(row=3, column=3)

        # Min Epsilon
        min_epsilon_label = tk.Label(self, text='Minumum value for Epsilon')
        min_epsilon_label.config(font=("Arial", 9))
        min_epsilon_label.grid(row=3, column=2)
        self.min_epsilon_entry = Entry(self)
        self.min_epsilon_entry.insert(0, '0.01')
        self.min_epsilon_entry.config(state='disabled')
        self.min_epsilon_entry.grid(row=3, column=3)

        # Decay Rate
        decay_rate_label = tk.Label(self, text='Decay Rate')
        decay_rate_label.config(font=("Arial", 9))
        decay_rate_ttp = CreateToolTip(decay_rate_label,"The decay rate is used to reduce the value of epsilon at each time step.")
        decay_rate_label.grid(row=4, column=2)
        self.decay_rate_entry = Entry(self)
        self.decay_rate_entry.insert(0, '0.01')
        self.decay_rate_entry.config(state='disabled')
        self.decay_rate_entry.grid(row=4, column=3)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = Button(self, text='Train the Q Agent!', command= self.train_q_agent)
        self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=11, column=0)

        # Button for viewing the current Q table, calls the print_q_table function onclick.
     #   self.view_qtable_btn = Button(self, text='View the Q table!', command= self.print_q_table)
     #   self.view_qtable_btn.config(state='disabled')
     #   self.view_qtable_btn.grid(row=11, column=1)

        # Button for testing the agent, calls the test_agent function onclick.
        self.test_agent_btn = Button(self, text='Test the agent!', command= self.test_agent)
        self.test_agent_btn.config(state='disabled')
        self.test_agent_btn.grid(row=11, column=2)

        # Button for returning to the start screen.
        self.return_btn = Button(self, text='Go Back', command = lambda: controller.show_frame("StartPage"))
        self.return_btn.grid(row=12)

        # Button for viewing rewards plot.
        self.view_plot_btn = Button(self, text='View rewards plot', command = self.plot)
        self.view_plot_btn.config(state='disabled')
        self.view_plot_btn.grid(row=11, column=3)

        # Label for displaying the current score of the agent after it has been tested, hidden by default.
        self.score = tk.Label(self, text=qlearn.score_over_time)
        self.score.grid(row=13)
        self.score.grid_forget()

        self.status = tk.Label(self, text='Waiting for user to select an environment...', bd=1, relief='sunken', anchor='w')
        self.status.grid(row=25, columnspan=4,sticky='we')
        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(22, weight=1)
    '''
    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background='yellow', relief='solid', borderwidth=1,
                       font=("times", "8", "normal"))
        label.pack(ipadx=1)
    '''
    def close(self, event=None):
        if self.tw:
            self.tw.destroy()

    def select_frozenlake(self):
        self.env_name = "FrozenLake-v0"
        self.env_frozenlake_btn.state(['pressed'])
        self.env_taxi_btn.state(['!pressed'])
        self.enable_entry()
        self.status['text'] = "Ready to begin training!"
        self.reset_gui()



    def reset_gui(self):
        self.test_agent_btn.config(state='disabled')
        self.view_plot_btn.config(state='disabled')
     #   self.view_qtable_btn.config(state='disabled')
        self.destroy_plots()
        try:
            self.random_label.destroy()
            self.trained_label.destroy()
            self.random_agent.unload()
            self.lbl.unload()

        except:
            pass

    def select_taxi(self):
        if(self.env_frozenlake_btn.state ==['pressed']):
            print("Hello")
            self.reset_gui()
        self.env_name = "Taxi-v2"
        self.env_taxi_btn.state(['pressed'])
        self.env_frozenlake_btn.state(['!pressed'])
        self.status['text'] = "Ready to begin training!"
        self.enable_entry()

        #self.reset_gui()

    def enable_entry(self):
        self.total_eps_entry.config(state='normal')
        self.total_test_eps_entry.config(state='normal')
        self.learning_rate_entry.config(state='normal')
        self.max_steps_entry.config(state='normal')
        self.gamma_entry.config(state='normal')
        self.epsilon_entry.config(state='normal')
        self.max_epsilon_entry.config(state='normal')
        self.min_epsilon_entry.config(state='normal')
        self.decay_rate_entry.config(state='normal')

        self.train_agent_btn.config(state='normal')

    def enable_test_btn(self):
     #   self.view_qtable_btn.config(state='normal')
        self.test_agent_btn.config(state='normal')

    def disable_entry(self):
        self.total_eps_entry.config(state='normal')
        self.total_test_eps_entry.config(state='normal')
        self.learning_rate_entry.config(state='normal')
        self.max_steps_entry.config(state='normal')
        self.gamma_entry.config(state='normal')
        self.epsilon_entry.config(state='normal')
        self.max_epsilon_entry.config(state='normal')
        self.min_epsilon_entry.config(state='normal')
        self.decay_rate_entry.config(state='normal')

    def print_q_table(self):
        print(qlearn.q_table)

    def training_progress_bar(self):
        self.progress_bar = Progressbar(self, orient=tk.HORIZONTAL, length=200, mode='indeterminate')

    def run_learning_thread(self):
        Thread(target=self.run_learning_function).start()
        Thread(target=self.set_status).start()

    def set_status(self):
        while(qlearn.progress <= self.total_eps ):
            status = str(qlearn.progress)
            self.status['text']  = "Training Agent on Episode {0}/{1}.".format(status, self.total_eps)
            if(qlearn.progress == self.total_eps):
                self.status['text'] = "Training Complete!"
            if(qlearn.progress == self.total_eps):
                break

    def run_learning_function(self):
        self.progress_bar.start()
        qlearn.learn(self.total_eps, self.max_steps, self.learning_rate, self.gamma, self.epsilon, self.min_epsilon,
                     self.max_epsilon, self.decay_rate)
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.enable_test_btn()

    def train_q_agent(self):
        self.progress_bar = Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress_bar.grid(row=19)
        qlearn.create_q_table(self.env_name)
        qlearn.q_table = np.zeros((qlearn.state_size, qlearn.action_space_size))
        self.total_eps = int(self.total_eps_entry.get())
        self.learning_rate = float(self.learning_rate_entry.get())
        self.max_steps = int(self.max_steps_entry.get())
        self.gamma = float(self.gamma_entry.get())
        self.epsilon = float(self.epsilon_entry.get())
        self.max_epsilon = float(self.max_epsilon_entry.get())
        self.min_epsilon = float(self.min_epsilon_entry.get())
        self.decay_rate = float(self.decay_rate_entry.get())
        self.run_learning_thread()

    def run_testing_thread(self):
        self.thread1 = Thread(target=self.run_testing_function)
        self.thread1.start()
        self.list_threads.append(self.thread1)
        self.thread2 = Thread(target=self.set_testing_status)
        self.thread2.start()
        self.thread3 = Thread(target=self.plot_live_graph)
        self.thread3.start()
        self.list_threads.append(self.thread2)

    def plot_live_graph(self):
        self.fig=plt.figure(num='Live Plot of Agent Training')

        self.ax1 = self.fig.add_subplot(1,1,1)
        self.live_plot_counter = 0
        self.live_animation = animation.FuncAnimation(self.fig, self.animate, interval=1000)
        self.live_plot_means = []
        #plt.yticks(np.arange(0, 20, step=1))

        thismanager = plt.get_current_fig_manager()
        thismanager.window.wm_geometry("+800+0")

        plt.show()

    def animate(self,i):
        xs = []
        ys = []
        xs_means = []

        counter=1
        continue_anim = True
        if(len(qlearn.rewards)>0):
            if(len(qlearn.rewards) == self.total_test_eps):
                continue_anim = False
            for reward in qlearn.rewards:
                xs.append(counter)
                ys.append(reward)
                if(qlearn.test_means == []):
                    xs_means = [0]
                else:
                    xs_means = qlearn.test_means
                counter+=1
        self.ax1.clear()
        plt.ylim([0,20])
        plt.yticks(np.arange(0, 20, 2))
        plt.xlim([0,self.total_test_eps])
        x_axis_step = self.total_test_eps
        plt.xticks(np.arange(0, self.total_test_eps, 10))
        rewards, = self.ax1.plot(xs, ys, label='Episode Reward')
        mean_rewards, = self.ax1.plot(xs, xs_means, label='Mean Reward Over Time')
        plt.legend(handles=[rewards, mean_rewards])
        plt.legend(loc='upper left')

        if continue_anim!= True:
            self.live_animation.event_source.stop()


    def set_testing_status(self):
        while(qlearn.test_progress <= self.total_test_eps +1):
            status = str(qlearn.test_progress)
            self.status['text']  = "Testing Agent on Episode {0}/{1}.".format(status, self.total_test_eps)
            if(qlearn.test_progress == self.total_test_eps):
                self.status['text'] = "Testing Complete!"
            if(qlearn.test_progress == self.total_test_eps):
                break

    def run_testing_function(self):
       # self.progress_bar.start()
        qlearn.test(self.max_steps, self.total_test_eps)

    def test_agent(self):
        self.destroy_plots()
        self.total_test_eps = int(self.total_test_eps_entry.get())
        self.max_steps = int(self.max_steps_entry.get())
        self.list_threads=[]
        self.run_testing_thread()
        print("All threads complete")
        self.thread1.join()
        gif_frame = tk.Frame(self)
        self.random_agent = ImageLabel(gif_frame)
        self.random_label = Label(gif_frame, text="Random Agent")
        self.random_label.grid(row=1, column=2)

        self.random_agent.label.grid(row=2, column=2, columnspan=2)
        #self.random_agent.pack(side='left', expand=True)
        self.random_agent.load('C:/Users/Ryan/PycharmProjects/FinalYearProject/records/random.gif')
        self.lbl = ImageLabel(gif_frame)
        self.trained_label = Label(gif_frame, text="Best Trained Agent")
        self.trained_label.grid(row=1, column=6, padx=100)
        #self.lbl.pack(side='left', expand=True)
#        self.spaceframe = tk.Frame(self, row=2, column=3, columnspan=2)
        self.lbl.label.grid(row=2, column=6, columnspan=2)
        self.lbl.load('C:/Users/Ryan/PycharmProjects/FinalYearProject/records/episode{}.gif'.format(qlearn.best_episode))
        self.view_plot_btn.config(state='normal')
        gif_frame.grid(row=21)
        self.update()

    def plot(self):
        self.plot_rewards()
        self.plot_training_rewards()
        self.plot_epsilon_decay()
        self.random_agent.unload()
        self.lbl.unload()
        self.random_label.grid_forget()
        self.trained_label.grid_forget()
        self.update()


    def destroy_plots(self):
        try:
            self.reward_canvas._tkcanvas.destroy()
            self.rewards_toolbar_frame.grid_forget()
            self.epsilon_canvas._tkcanvas.destroy()
            self.epsilon_toolbar_frame.grid_forget()
            self.training_canvas._tkcanvas.destroy()
            self.training_toolbar_frame.grid_forget()
        except:
            pass

    def plot_rewards(self):
        total_test_eps = int(self.total_test_eps_entry.get())
        a,f = qlearn.plot(total_test_eps)
        self.reward_canvas = FigureCanvasTkAgg(f, self)

        self.reward_canvas.draw()
        self.reward_canvas.get_tk_widget().grid(row=14, column=0)
        frame1 = tk.Frame()

        self.rewards_toolbar_frame = tk.Frame(self)
        self.rewards_toolbar_frame.grid(row=15, column=0)
        toolbar = NavigationToolbar2Tk(self.reward_canvas,self.rewards_toolbar_frame)
        toolbar.update()
        #canvas._tkcanvas.grid(row=15)

    def plot_epsilon_decay(self):
        total_eps = int(self.total_eps_entry.get())
        a,f = qlearn.plot_epsilon_decay(total_eps)

        self.epsilon_canvas = FigureCanvasTkAgg(f, self)

        self.epsilon_canvas.draw()
        self.epsilon_canvas.get_tk_widget().grid(row=14, column=2)

        self.epsilon_toolbar_frame = tk.Frame(self)
        self.epsilon_toolbar_frame.grid(row=15, column=2)
        toolbar = NavigationToolbar2Tk(self.epsilon_canvas, self.epsilon_toolbar_frame)
        toolbar.update()

    def plot_training_rewards(self):
        total_eps = int(self.total_eps_entry.get())
        a,f = qlearn.plot_training_rewards(total_eps)

        self.training_canvas = FigureCanvasTkAgg(f, self)

        self.training_canvas.draw()
        self.training_canvas.get_tk_widget().grid(row=14, column=4)

        self.training_toolbar_frame = tk.Frame(self)
        self.training_toolbar_frame.grid(row=15, column=4)
        toolbar = NavigationToolbar2Tk(self.training_canvas, self.training_toolbar_frame)
        toolbar.update()

class PageTwo(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.top_frame = tk.Frame(self)
        self.top_frame.grid(row=0, sticky='ew')
        self.middle_frame = tk.Frame(self)
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.grid(row=2, sticky='ew')
        self.grid_rowconfigure(1, weight=1)

        self.controller = controller
        heading = tk.Label(self.top_frame, text='Select whether you want to play against the RL agent or just watch it play.').grid(row=0)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = Button(self.top_frame, text='Play against the Agent!', command= self.human_play_pong)
       # self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=1, column=0)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = Button(self.top_frame, text='View Human vs AI plots!', command=self.plot_human_vs_AI)
        # self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=2, column=0)

        # Button for viewing the current Q table, calls the print_q_table function onclick.
        self.view_qtable_btn = Button(self.top_frame, text='Watch the agent play!!', command= self.agent_only_pong)
        #self.view_qtable_btn.config(state='disabled')
        self.view_qtable_btn.grid(row=1, column=1)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = Button(self.top_frame, text='View training plots!', command=self.training_plots)
        # self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=2, column=1)


        # Button for returning to the start screen.
        self.return_btn = tk.Button(self.bottom_frame, text='Go Back', command = lambda: controller.show_frame("StartPage"))
        self.return_btn.grid(row=1)


    def plot_human_vs_AI(self):

        df = pd.read_csv('human_vs_AI.csv')
        fig = Figure(figsize=(5, 4), dpi=100)
        fig.suptitle("Human vs DQN Scores.")
        ax= fig.add_subplot(111)
        df.plot(kind='line', x='Game', y='AI_Score', ax=ax)
        df.plot(kind='line', x='Game', y='Opp_Score', ax=ax)

        self.canvas = FigureCanvasTkAgg(fig, self.bottom_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1)

    def training_plots(self):

        df = pd.read_csv('training_statistics.csv')

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(1,1,1)
        x= max(df.Global_Step_Counter)
        ax.set_xticks(range(1,45000, 20000))
        fig.suptitle("Average Return Percentage over last 10 episodes")
       # df.plot(kind='line', x='Episode', y='Reward', ax=ax)
        df.plot(kind='line', x='Global_Step_Counter', y='Mean_Score_Over_Last_10', ax=ax)

        self.canvas = FigureCanvasTkAgg(fig, self.bottom_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1)


    def human_play_pong(self):
       # game = pong.Pong(is_human=True)
       self.play = Play.Play(human_mode=True)

    def humanpong(self):
        play = Play.Play(human_mode=True)
    def agent_only_pong(self):
        play = Play.Play(human_mode=False)
        pass

class PageThree(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        heading = tk.Label(self, text='Select whether you want to play against the RL agent or just watch it play.').grid(row=0)

class CreateToolTip(object):
    '''
    CreateToolTip Code taken from Stack Overflow user crxguy52
    https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
    Create tooltip for a given widget.
    '''
    def __init__(self, widget, text='widget info'):
        self.wait_time = 500    # Miliseconds
        self.wrap_length = 180  # Pixels
        self.widget = widget

        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.wait_time, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wrap_length)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

class ImageLabel():
    '''
    Taken from stack overflow
    https://stackoverflow.com/questions/43770847/play-an-animated-gif-in-python-with-tkinter
    '''
    """a label that displays images, and plays them if they are gifs"""
    def __init__(self, gif_frame):
        self.label = tk.Label(gif_frame)

    def load(self, im):

        if isinstance(im, str):
            im = Image.open(im)
        self.label.loc = 0
        self.label.frames = []

        try:
            for i in count(1):
                self.label.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.label.delay = im.info['duration']
        except:
            self.label.delay = 100

        if len(self.label.frames) == 1:
            self.label.config(image=self.label.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.label.config(image=None)
        self.label.frames = None

    def next_frame(self):
        if self.label.frames:
            self.label.loc += 1
            self.label.loc %= len(self.label.frames)
            self.label.config(image=self.label.frames[self.label.loc])
            self.label.after(self.label.delay, self.next_frame)


if __name__ == "__main__":
    qlearn = QLearning()
    app = GUI()
    app.mainloop()

