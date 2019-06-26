import gym
from DeepLearning import *


class QFunction(chainer.Chain):
    def __init__(self):
        super().__init__(l1=L.Linear(4, 50), l2=L.Linear(50, 50), l3=L.Linear(50, 2))

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        return self.l3(h)


class VFunction(chainer.Chain):
    def __init__(self):
        super().__init__(l1=L.Linear(4, 50), l2=L.Linear(50, 50), l3=L.Linear(50, 1))

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        return self.l3(h)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    qnet = QFunction()
    optimizer = chainer.optimizers.Adam(eps=1.0e-2)
    optimizer.setup(qnet)
    vnet = VFunction()
    optimizer_v = chainer.optimizers.Adam(eps=1.0e-2)
    optimizer_v.setup(vnet)
    agent = A3Cagent(qnet, vnet, optimizer, optimizer_v, env, 2, gamma=0.95, epoch_t_max=1000,
                     t_max=5, process_num=6)
    agent.set_plotagent('output/a3c', 'score', 'recoder', plot_show=True)
    agent.learning(1000, record_step=50, csv_recode=True)
