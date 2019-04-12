import tkinter as tk
import pong
import DQN.PlayAgainstAgent2 as Play
from q_learning_model import *
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import time
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from itertools import count, cycle
import PlayAgainstAgent
class GUI(tk.Tk):
    def __init__(self):

        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0, weight=1)

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
        train_q_learning_btn = tk.Button(self, text='Train OpenAI gym using Q Learning', command= lambda: controller.show_frame("PageOne"))
        train_q_learning_btn.pack()

        train_dqn_learning_btn = tk.Button(self, text='Train Atari using Deep Q Learning', command= lambda: controller.show_frame("PageTwo"))
        train_dqn_learning_btn.pack()

        quit_btn = tk.Button(self, text='Quit', command=self.quit)
        quit_btn.pack()



class PageOne(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        heading = tk.Label(self, text='Enter the hyperparameters you want to train the agent with').grid(row=0)

        # Select the environment to train/test with.
        environment_label = tk.Label(self, text='Environment').grid(row=1)
        self.env_frozenlake_btn = tk.Button(self, text='Frozen Lake', command= self.select_frozenlake)
        self.env_frozenlake_btn.grid(row=1, column=1)

        self.env_taxi_btn = tk.Button(self, text='Taxi', command= self.select_taxi)
        self.env_taxi_btn.grid(row=1, column=2)

        # Entry for total number of episodes to train on
        total_eps_label = tk.Label(self, text='Total episodes to train').grid(row=2)
        self.total_eps_entry = tk.Entry(self)
        self.total_eps_entry.insert(0, '50000')
        self.total_eps_entry.config(state='disabled')
        self.total_eps_entry.grid(row=2, column=1)


        # Entry for total number of test episodes to train on
        total_test_eps_label = tk.Label(self, text='Total episodes to test').grid(row=3)
        self.total_test_eps_entry = tk.Entry(self)
        self.total_test_eps_entry.insert(0, '100')
        self.total_test_eps_entry.config(state='disabled')
        self.total_test_eps_entry.grid(row=3, column=1)

        # Entry for learning rate (alpha)
        learning_rate_label = tk.Label(self, text='Learning Rate').grid(row=4)
        self.learning_rate_entry = tk.Entry(self)
        self.learning_rate_entry.insert(0, '0.7')
        self.learning_rate_entry.config(state='disabled')
        self.learning_rate_entry.grid(row=4, column=1)

        # Entry for max steps per episode
        max_steps_label = tk.Label(self, text='Maximum steps per epsiode').grid(row=5)
        self.max_steps_entry = tk.Entry(self)
        self.max_steps_entry.insert(0, '99')
        self.max_steps_entry.config(state='disabled')
        self.max_steps_entry.grid(row=5, column=1)

        # Entry for Gamma
        gamma_label = tk.Label(self, text='Gamma (Discount Rate)').grid(row=6)
        self.gamma_entry = tk.Entry(self)
        self.gamma_entry.insert(0, '0.618')
        self.gamma_entry.config(state='disabled')
        self.gamma_entry.grid(row=6, column=1)

        # Entry for Epsilon
        epsilon_label = tk.Label(self, text='Epsilon').grid(row=7)
        self.epsilon_entry = tk.Entry(self)
        self.epsilon_entry.insert(0, '1.0')
        self.epsilon_entry.config(state='disabled')
        self.epsilon_entry.grid(row=7, column=1)

        # Max Epsilon
        max_epsilon_label = tk.Label(self, text='Starting value for Epsilon').grid(row=8)
        self.max_epsilon_entry = tk.Entry(self)
        self.max_epsilon_entry.insert(0, '1.0')
        self.max_epsilon_entry.config(state='disabled')
        self.max_epsilon_entry.grid(row=8, column=1)

        # Min Epsilon
        min_epsilon_label = tk.Label(self, text='Minumum value for Epsilon').grid(row=9)
        self.min_epsilon_entry = tk.Entry(self)
        self.min_epsilon_entry.insert(0, '0.01')
        self.min_epsilon_entry.config(state='disabled')
        self.min_epsilon_entry.grid(row=9, column=1)

        # Decay Rate
        decay_rate_label = tk.Label(self, text='Decay Rate').grid(row=10)
        self.decay_rate_entry = tk.Entry(self)
        self.decay_rate_entry.insert(0, '0.01')
        self.decay_rate_entry.config(state='disabled')
        self.decay_rate_entry.grid(row=10, column=1)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = tk.Button(self, text='Train the Q Agent!', command= self.train_q_agent)
        self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=11, column=0)

        # Button for viewing the current Q table, calls the print_q_table function onclick.
        self.view_qtable_btn = tk.Button(self, text='View the Q table!', command= self.print_q_table)
        self.view_qtable_btn.config(state='disabled')
        self.view_qtable_btn.grid(row=11, column=1)

        # Button for testing the agent, calls the test_agent function onclick.
        self.test_agent_btn = tk.Button(self, text='Test the agent!', command= self.test_agent)
        self.test_agent_btn.config(state='disabled')
        self.test_agent_btn.grid(row=11, column=2)

        # Button for returning to the start screen.
        self.return_btn = tk.Button(self, text='Go Back', command = lambda: controller.show_frame("StartPage"))
        self.return_btn.grid(row=12)

        # Button for viewing rewards plot.
        self.view_plot_btn = tk.Button(self, text='View rewards plot', command = self.plot)
        self.view_plot_btn.config(state='disabled')
        self.view_plot_btn.grid(row=11, column=3)

        # Label for displaying the current score of the agent after it has been tested, hidden by default.
        self.score = tk.Label(self, text=qlearn.score_over_time)
        self.score.grid(row=13)
        self.score.grid_forget()


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

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()


    def select_frozenlake(self):
        self.env_name = "FrozenLake-v0"
        self.env_frozenlake_btn.config(relief='sunken')
        self.env_taxi_btn.config(relief='raised')
        self.enable_entry()

    def select_taxi(self):
        self.env_name = "Taxi-v2"
        self.env_taxi_btn.config(relief='sunken')
        self.env_frozenlake_btn.config(relief='raised')
        self.enable_entry()

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
        self.view_qtable_btn.config(state='normal')
        self.test_agent_btn.config(state='normal')
        self.view_plot_btn.config(state='normal')

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


    def train_q_agent(self):
        qlearn.create_q_table(self.env_name)
        qlearn.q_table = np.zeros((qlearn.state_size, qlearn.action_space_size))
        total_eps = int(self.total_eps_entry.get())
        learning_rate = float(self.learning_rate_entry.get())
        max_steps = int(self.max_steps_entry.get())
        gamma = float(self.gamma_entry.get())
        epsilon = float(self.epsilon_entry.get())
        max_epsilon = float(self.max_epsilon_entry.get())
        min_epsilon = float(self.min_epsilon_entry.get())
        decay_rate = float(self.decay_rate_entry.get())
        qlearn.learn(total_eps, max_steps, learning_rate, gamma,epsilon, min_epsilon, max_epsilon, decay_rate)


    def test_agent(self):
        total_test_eps = int(self.total_test_eps_entry.get())
        max_steps = int(self.max_steps_entry.get())
        qlearn.test(max_steps, total_test_eps)
        self.score['text'] = "Score over time: " + str(qlearn.score_over_time)
        self.score.grid()


    def plot(self):
        self.plot_rewards()
        self.plot_training_rewards()
        self.plot_epsilon_decay()


    def plot_rewards(self):
        total_test_eps = int(self.total_test_eps_entry.get())
        a,f = qlearn.plot(total_test_eps)
        canvas = FigureCanvasTkAgg(f, self)

        canvas.draw()
        canvas.get_tk_widget().grid(row=14)

        toolbar_frame = tk.Frame(master=self.controller)
        toolbar_frame.pack()
        toolbar = NavigationToolbar2Tk(canvas,toolbar_frame)
        toolbar.update()

        #canvas._tkcanvas.grid(row=15)

    def plot_epsilon_decay(self):
        total_eps = int(self.total_eps_entry.get())
        a, f = qlearn.plot_epsilon_decay(total_eps)

        canvas = FigureCanvasTkAgg(f, self)

        canvas.draw()
        canvas.get_tk_widget().grid(row=14, column=2)

        toolbar_frame = tk.Frame(master=self.controller)
        toolbar_frame.pack()
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def plot_training_rewards(self):
        total_eps = int(self.total_eps_entry.get())
        a, f = qlearn.plot_training_rewards(total_eps)

        canvas = FigureCanvasTkAgg(f, self)

        canvas.draw()
        canvas.get_tk_widget().grid(row=15)

        toolbar_frame = tk.Frame(master=self.controller)
        toolbar_frame.pack()
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

class PageTwo(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        heading = tk.Label(self, text='Select whether you want to play against the RL agent or just watch it play.').grid(row=0)

        # Button for training the agent, calls the train_q_agent function onclick.
        self.train_agent_btn = tk.Button(self, text='Play against the Agent!', command= self.human_play_pong)
       # self.train_agent_btn.config(state='disabled')
        self.train_agent_btn.grid(row=1, column=0)

        # Button for viewing the current Q table, calls the print_q_table function onclick.
        self.view_qtable_btn = tk.Button(self, text='Watch the agent play!!', command= self.agent_only_pong)
        #self.view_qtable_btn.config(state='disabled')
        self.view_qtable_btn.grid(row=1, column=1)


        # Button for returning to the start screen.
        self.return_btn = tk.Button(self, text='Go Back', command = lambda: controller.show_frame("StartPage"))
        self.return_btn.grid(row=2)

    def human_play_pong(self):
       # game = pong.Pong(is_human=True)
       play = Play.Play()
    def agent_only_pong(self):
        pass

class PageThree(tk.Frame):
    # Q Learning page.
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        heading = tk.Label(self, text='Select whether you want to play against the RL agent or just watch it play.').grid(row=0)




if __name__ == "__main__":
    qlearn = QLearning()
    app = GUI()
    app.mainloop()

