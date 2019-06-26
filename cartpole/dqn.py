import gym
from DeepLearning import *


class QFunction(chainer.Chain):
    def __init__(self):
        super().__init__(l1=L.Linear(4, 50), l2=L.Linear(50, 50), l3=L.Linear(50, 2))

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        return self.l3(h)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    qnet = QFunction()
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(qnet)
    agent = DQNagent(qnet, optimizer, env, 2, predata_num=500, gamma=0.95, memory_size=10 ** 6, update_frequency=20,
                     epsilon=0.3, data_batch_num=20)
    agent.set_plotagent('output/dqn', 'score', 'recoder', plot_show=True)
    agent.learning(1000, record_step=50, csv_recode=True)
