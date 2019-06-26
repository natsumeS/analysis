from copy import deepcopy
from multiprocessing import Process, Lock, Value, Barrier
from multiprocessing.managers import BaseManager

from DeepLearning import *


class model_pv:
    def __init__(self, model_p, model_v, optimizer_p, optimizer_v):
        self.model_p = model_p
        self.model_v = model_v
        self.optimizer_p = optimizer_p
        self.optimizer_v = optimizer_v
        self.optimizer_p.setup(self.model_p)
        self.optimizer_v.setup(self.model_v)

    def get_v(self, observation):
        return self.model_v(Variable(np.array([observation], dtype=np.float32))).data[0][0]

    def get_action_softmax(self, observation):
        softmax = F.softmax(self.model_p(Variable(np.array([observation], dtype=np.float32)))).data[0]
        return np.random.choice(range(len(softmax)), p=softmax)

    def cleargrads(self):
        self.model_p.cleargrads()
        self.model_v.cleargrads()

    def update(self):
        self.optimizer_v.update()
        self.optimizer_p.update()

    def addgrads(self, model_p, model_v):
        self.model_p.addgrads(model_p)
        self.model_v.addgrads(model_v)

    def get_model_p(self):
        return deepcopy(self.model_p)


class A3Cagent(Agent):
    def __init__(self, model, model_v, optimizer, optimizer_v, environment, action_num, *,
                 gamma=0.95, t_max=5, process_num=4, epoch_t_max=10000, epsilon=0.1, epsilon_change=None,
                 epsilon_min=0.1):
        super().__init__(model, optimizer, environment, action_num, gamma=gamma, epsilon=epsilon,
                         epsilon_change=epsilon_change, epsilon_min=epsilon_min)
        self.t = 0
        self.t_max = t_max
        self.epoch_t = 0
        self.epoch_t_max = epoch_t_max
        self.process_num = process_num
        BaseManager.register('model_pv', model_pv)
        manager = BaseManager()
        manager.start()
        self.share_model_pv = manager.model_pv(model, model_v, optimizer, optimizer_v)

    def training_one_step(self):
        lock_model = Lock()
        p_list = []
        T = Value('I', 0, lock=True)
        b = Barrier(self.process_num)
        for i in range(self.process_num):
            p_list.append(Process(target=self.each_actor_learner, args=(lock_model, self.share_model_pv, T, b)))
            p_list[i].start()
        for p in p_list:
            p.join()
        self.model = self.share_model_pv.get_model_p()

    def each_actor_learner(self, lock, share_models, T, b):
        t = 0
        mymodels_pv = None
        env = self.env
        lock.acquire()
        try:
            mymodels_pv = deepcopy(share_models)
        finally:
            lock.release()
        b.wait()
        observation = env.reset()
        while True:
            # update_step
            t_start = t
            # reset list
            state_list = []
            action_list = []
            reward_list = []
            R = 0
            while True:
                state_list.append(observation)
                action = mymodels_pv.get_action_softmax(observation)
                action_list.append(action)
                nextobservation, reward, done, _ = env.step(action)
                reward_list.append(reward)
                if done:
                    observation = env.reset()
                    break
                if t - t_start == self.t_max:
                    R = mymodels_pv.get_v(observation)
                    break
                observation = nextobservation
                t += 1
            i = t - t_start
            with T.get_lock():
                T.value += i
            mymodels_pv.cleargrads()
            while i >= 0:
                R = reward_list[i] + self.gamma * R
                V = mymodels_pv.get_v(state_list[i])
                loss_p = F.softmax_cross_entropy(mymodels_pv.model_p(np_to_variable_float(state_list[i])),
                                                 np_to_variable_int(action_list[i]))
                loss_p *= R - V
                loss_v = F.mean_squared_error(mymodels_pv.model_v(np_to_variable_float(state_list[i])),
                                              np_to_variable_float([R]))
                loss_p.backward()
                loss_v.backward()
                i -= 1
            lock.acquire()
            try:
                share_models.cleargrads()
                share_models.addgrads(mymodels_pv.model_p, mymodels_pv.model_v)
                share_models.update()
                mymodels_pv = deepcopy(share_models)
            finally:
                lock.release()
            if T.value > self.epoch_t_max:
                break

    def learning(self, epoch_max, *, record_step=None, csv_recode=False):
        self.epoch = 0
        if csv_recode:
            self.plotagent.csv_recode_start(self(best_score_upload=False, record_step=None))
        while self.epoch < epoch_max:
            self.training_one_step()
            score = self(best_score_upload=True, record_step=record_step)
            if self.plotagent is not None:
                self.plotagent.add_data(self.epoch, score)
            if csv_recode:
                self.plotagent.csv_add_data(score)
            self.epoch += 1
