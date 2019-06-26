import os
import matplotlib.pyplot as plt
from gym import wrappers
import time
import csv
from DeepLearning import *


def np_to_variable_float(array):
    return Variable(np.array([array], dtype=np.float32))


def np_to_variable_int(array):
    return Variable(np.array([array], dtype=np.int32))


class Agent:
    def __init__(self, model, optimizer, environment, action_num, *, gamma=0.95, epsilon=0.1, epsilon_change=None,
                 epsilon_min=0.1):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        self.env = environment
        self.gamma = gamma
        self.action_num = action_num
        self.best_score = -1
        self.epoch = 0
        self.epsilon = epsilon
        self.model.cleargrads()
        self.epsilon_change = epsilon_change
        self.epsilon_min = epsilon_min
        self.plotagent = None

    # get average score
    def __call__(self, *, best_score_upload=False, record_step=None):
        score = 0
        for times in range(10):
            observation = self.env.reset()
            while True:
                action = self.get_action(observation)
                observation, reward, done, info = self.env.step(action)
                score += reward
                if done:
                    break
        score = score / 10.0
        if best_score_upload:
            if self.best_score < score:
                self.best_score = score
                filename = 'best_model_{}'.format(self.__class__.__name__)
                self.plotagent.save_model(filename, self.model)
        if record_step is not None and self.epoch % record_step == 0:
            filename = 'epoch{}_model_{}'.format(self.epoch, self.__class__.__name__)
            self.plotagent.save_model(filename, self.model)
        return score

    def set_plotagent(self, data_dir, plot_filename, csv_filename, *, plot_show=False):
        self.plotagent = plotagent(data_dir, plot_filename, csv_filename, plot_show=plot_show)

    def get_Q(self, observation):
        return self.model(Variable(np.array([observation], dtype=np.float32))).data[0]

    def get_action(self, observation):
        return np.argmax(self.get_Q(observation))

    def get_action_egreedy(self, observation):
        if self.epsilon_change is not None and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_change
        if rdm.random() < self.epsilon:
            return rdm.randint(0, self.action_num - 1)
        return self.get_action(observation)

    def get_action_softmax(self, observation):
        softmax = F.softmax(self.model(Variable(np.array([observation], dtype=np.float32)))).data[0]
        return np.random.choice(range(self.action_num), p=softmax)

    def training_none_step(self):
        assert False, 'def training_none_step method'

    def save_model(self, filename):
        serializers.save_npz(filename, self.model)

    def load_model(self, filename):
        serializers.load_npz(filename, self.model)

    def display(self, *, counter=10, monitor=False):
        if monitor:
            self.env = self.plotagent.get_Monitor(self.env)
        for times in range(counter):
            observation = self.env.reset()
            while True:
                self.env.render()
                action = self.get_action(observation)
                observation, reward, done, _ = self.env.step(action)
                if done:
                    break
        self.env.close()

    def learning(self, epoch_max, *, record_step=None, csv_recode=False):
        assert False, 'set learning method'


class plotagent:
    def __init__(self, data_dir, plot_filename, csv_filename, *, plot_show=False):
        self.fig, self.ax = plt.subplots(1, 1)
        self.x_list = []
        self.y_list = []
        self.line, = self.ax.plot(self.x_list, self.y_list)
        self.plot_show = plot_show
        self.dir = data_dir
        self.plot_filename = plot_filename
        self.csv_filename = "{}/{}.csv".format(self.dir, csv_filename)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.time = None

    def add_data(self, x, y):
        self.x_list.append(x)
        self.y_list.append(y)
        self.line.set_data(self.x_list, self.y_list)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.savefig("{}/{}.png".format(self.dir, self.plot_filename))
        if self.plot_show:
            plt.pause(0.1)

    def save_model(self, model_filename, model):
        serializers.save_npz("{}/{}.npz".format(self.dir, model_filename), model)

    def get_Monitor(self, env):
        return wrappers.Monitor(env, self.dir)

    def csv_recode_start(self, data):
        with open(self.csv_filename, "w") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([0.0, data])
        self.time = time.time()

    def csv_add_data(self, data):
        with open(self.csv_filename, "a") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([time.time() - self.time, data])
