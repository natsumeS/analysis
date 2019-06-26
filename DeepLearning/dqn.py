from collections import deque
from copy import deepcopy
from DeepLearning import *


class DQNagent(Agent):
    def __init__(self, model, optimizer, environment, action_num, *,
                 predata_num=500, gamma=0.95, memory_size=10 ** 6, update_frequency=20, epsilon=0.1,
                 data_batch_num=20, epsilon_change=None, epsilon_min=0.1):
        super().__init__(model, optimizer, environment, action_num, gamma=gamma, epsilon=epsilon,
                         epsilon_change=epsilon_change, epsilon_min=epsilon_min)
        self.memory = deque(maxlen=memory_size)
        self.data_batch_num = data_batch_num
        self.update_frequency = update_frequency
        self.update_counter = 0
        self.training_model = deepcopy(self.model)
        self.training_optimizer = deepcopy(optimizer)
        self.training_optimizer.setup(self.training_model)
        self.predata(predata_num)

    # data=(state,reward,action,nextstate)
    # done =True => nextstate=None
    def add_data(self, state, reward, action, nextstate=None):
        self.memory.append((state, reward, action, nextstate))

    def predata(self, predata_num):
        data_num = 0
        while True:
            observation = self.env.reset()
            while True:
                action = self.env.action_space.sample()
                nextobservation, reward, done, _ = self.env.step(action)
                if done:
                    nextobservation = None
                self.add_data(observation, reward, action, nextobservation)
                data_num += 1
                if data_num > predata_num:
                    return
                if done:
                    break
                observation = nextobservation

    def training_one_step(self):
        sampling_data = random.sample(self.memory, self.data_batch_num)
        s = []
        t = []
        for sample in sampling_data:
            state, reward, action, nextstate = sample

            teach_data = self.get_Q(state)
            teach = reward
            if nextstate is not None:
                teach += self.gamma * max(self.get_Q(nextstate))
            teach_data[action] = teach
            s.append(state)
            t.append(teach_data)
        loss = F.mean_squared_error(Variable(np.array(t, dtype=np.float32)),
                                    self.training_model(Variable(np.array(s, dtype=np.float32)))) / 2
        self.training_model.cleargrads()
        loss.backward()
        self.training_optimizer.update()
        self.update_counter += 1
        if self.update_counter > self.update_frequency:
            self.model = deepcopy(self.training_model)
            self.update_counter = 0

    def learning(self, epoch_max, *, record_step=None, csv_recode=False):
        self.epoch = 0
        if csv_recode:
            self.plotagent.csv_recode_start(self(best_score_upload=False, record_step=None))
        while self.epoch < epoch_max:
            observation = self.env.reset()
            for t in range(500):
                action = self.get_action_egreedy(observation)
                nextobservation, reward, done, _ = self.env.step(action)
                if done:
                    nextobservation = None
                self.add_data(observation, reward, done, nextobservation)
                self.training_one_step()
                observation = nextobservation
                if done:
                    break
            score = self(best_score_upload=True, record_step=record_step)
            if self.plotagent is not None:
                self.plotagent.add_data(self.epoch, score)
            if csv_recode:
                self.plotagent.csv_add_data(score)
            self.epoch += 1
