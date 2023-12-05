from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import argparse
import logging
import config
from fl_hist import FL_HIST_PARTIAL_STAGE
from interval_agent_another import VALAgent
# from interval_agent import VALAgent
import matplotlib.pyplot as plt
import copy
import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()  # 创建一个解析对象
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')  # 向该对象中添加你要关注的命令行参数和选项
args = parser.parse_args()  # 进行解析


# RL相关
class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation, num_clients, num_clusters):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=128)
        self.layer_out1 = torch.nn.Linear(in_features=128, out_features=num_clients)  # 输出之一，client number
        # self.layer_out2 = torch.nn.Linear(in_features=128, out_features=5)  # 输出之二，local iteration number
        self.output_layer = torch.nn.Linear(in_features=128, out_features=output_dimension)
        self.normal = torch.nn.LayerNorm(normalized_shape=num_clients, eps=0, elementwise_affine=False)
        self.output_activation = output_activation
        # self.num = num_edges  # cluster number

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        # layer_3_output = self.output_layer(layer_2_output)

        # x1, x2 = layer_3_output.split([3, 20], dim=1)
        y = self.output_activation(self.normal(self.layer_out1(layer_2_output)))  # action，决定选哪些clients
        # y = self.output_activation(self.normal(self.layer_out1(layer_2_output)))  # action，决定选哪些clients
        # y_1=torch.zeros(y.shape)
        # y_1[(torch.arange(len(y)).unsqueeze(1), torch.topk(y, 10).indices)] = 1
        # y1=self.output_activation(y)
        x = []

        # for i in range(self.num):  # 为每个cluster决定local iteration number
        #     # x.append(self.output_activation(self.layer_out1(layer_2_output)))
        #     x.append(self.output_activation(self.layer_out2(layer_2_output)))

        output = torch.tensor([])
        # for i in range(1, len(x)):
        #     output = torch.cat([output, x[i]], dim=1)
        output = torch.cat([output, y], dim=1)
        # x1 = self.output_activation(self.layer_out1(layer_2_output))
        # x2 = self.output_activation(self.layer_out2(layer_2_output))

        # output = torch.cat([x1, x2], dim=1)

        # output = self.output_activation(layer_3_output)
        return output


class Critic(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = self.output_layer(layer_2_output)

        output = self.output_activation(layer_3_output)
        return output


class SACAgent:
    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 50
    DISCOUNT_RATE = 1
    LEARNING_RATE = 10 ** -4
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01

    def __init__(self, environment):
        self.environment = environment
        self.num_clients = environment.config.clients.total
        self.state_dim = environment.state_space  # 状态维度
        self.action_dim = environment.action_space  # action维度
        self.per_round = environment.config.clients.per_round
        self.clusters = environment.config.clusters

        self.critic_local = Critic(input_dimension=self.state_dim,
                                   output_dimension=self.action_dim)
        self.critic_local2 = Critic(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Critic(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_target2 = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Actor(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
            # num_max=self.num_max,
            num_clients=self.num_clients,
            num_clusters=self.clusters,
            # tau_max=self.tau_max

        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        # action_edge_num=self.num_max+self.tau_max
        # action_edge_num = self.tau_max
        client_actions = []
        # edges_action_probabilities = [0] * self.num_edges
        # for i in range(self.num_edges):
        #     edges_action_probabilities[i] = action_probabilities[i * action_edge_num:(i + 1) * action_edge_num]
        # client_actions = action_probabilities[self.num_edges * action_edge_num:]
        client_actions = action_probabilities
        # for i in range(self.num_edges):
            # 映射客户端选择从[0,num_max-1]到[0,length-1]，每个cluster选一个
            # length = len(edges[i].clients) - 1
            # a1 = np.round(1 + ((length - 1) / (self.num_max - 1)) * (a1 - 1))
            # discrete_action.append(np.random.choice(range(0, self.num_max), p=edges_action_probabilities[i][0 : self.num_max]))
            # iteration+1
            # discrete_action.append(np.random.choice(range(0, self.tau_max), p=edges_action_probabilities[i][self.num_max : action_edge_num]))
        #     discrete_action.append(np.random.choice(range(0, self.tau_max),
        #                                             p=edges_action_probabilities[i][0: self.tau_max]))
        # for c in range(self.clusters):

        client_choice = list(np.random.choice(range(0, self.num_clients),
                                              size=self.per_round,
                                              p=client_actions, replace=False))
        # client_choice = [client_choice[i] + c*self.num_clients for i in range(len(client_choice))]
        discrete_action = discrete_action + client_choice
        # discrete_action = client_choice

        # discrete_action.append(np.random.choice(range(0,3), p=action_probabilities[0:3]))
        # discrete_action.append(np.random.choice(range(0,20), p=action_probabilities[3:23]))
        # discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        # action_edge_num = self.num_max + self.tau_max
        # action_edge_num = self.tau_max
        # edges_action_probabilities = [0] * self.num_edges
        # for i in range(self.num_edges):
        #     edges_action_probabilities[i] = action_probabilities[i * action_edge_num:(i + 1) * action_edge_num]
        # 取client selection相关的数据，前面的是第二个参数local iterations
        # client_actions = action_probabilities[self.num_edges * action_edge_num:]
        client_actions = action_probabilities
        # for i in range(self.num_edges):
        #     discrete_action.append(np.argmax(edges_action_probabilities[i][0: self.tau_max]))
        # for c in range(self.clusters):
        for i in range(self.per_round):
            index = np.argmax(client_actions)
            client_actions[index] = 0
            discrete_action.append(index)
        # client_choice=list(np.random.choice(range(0, self.num_clients), size=8, p=client_actions,replace=False))
        # discrete_action=discrete_action+client_choice
        # discrete_action.append(np.argmax(action_probabilities[0:3]))
        # discrete_action.append(np.argmax(action_probabilities[3:23]))
        # discrete_action = np.argmax(action_probabilities)
        return discrete_action

    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        count = 0
        # for i in range(self.num_edges):
        #     discrete_action[i] = discrete_action[i] + count
        #     count += self.tau_max

        # idx = 0
        # for i in range(len(discrete_action) - 0):
        #     discrete_action[i + idx] = discrete_action[i + idx] + idx
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]), dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            # actions_tensor_2 = torch.tensor(np.array(minibatch_separated[5]), dtype=torch.float32)

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE * soft_state_values

        # actions = []
        # num = self.vehicle_num * 2
        temp = torch.split(actions_tensor, 1, dim=1)

        # soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_values = self.critic_local(states_tensor)
        soft_q_values_1 = soft_q_values.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values_2 = soft_q_values.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor)
        soft_q_values2_1 = soft_q_values2.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2_2 = soft_q_values2.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_value = self.critic_local(states_tensor)
        soft_q_value2 = self.critic_local2(states_tensor)
        soft_q_values = []
        soft_q_values2 = []

        # a = temp[1].type(torch.int64).squeeze()

        for i in range(len(temp)):
            soft_q_values.append(soft_q_value.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
            soft_q_values2.append(
                soft_q_value2.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
        critic_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values), next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values2), next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor, ):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class ReplayBuffer:

    def __init__(self, environment, capacity=5000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10 ** -2
        self.delta = 10 ** -4
        self.indices = None

    def get_transition_type_str(self, environment):
        # state_dim = environment.observation_space.shape[0]
        state_dim = environment.state_space
        state_dim_str = '' if state_dim == () else str(state_dim)
        # state_type_str = environment.observation_space.sample().dtype.name
        state_type_str = "float32"
        # action_dim = "2"
        # action_dim = environment.num_edges + 10  # 这个干啥用的？？？
        action_dim = environment.config.clients.per_round
        # action_dim = environment.action_space.shape
        action_dim_str = '' if action_dim == () else str(action_dim)
        # action_type_str = environment.action_space.sample().__class__.__name__
        action_type_str = "int"

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)

        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count


TRAINING_EVALUATION_RATIO = 4
RUNS = 4  # default value = 5
EPISODES_PER_RUN = 300
# STEPS_PER_EPISODE = 200
MAX_EP_STEP = 200  # 每个episode的最大steps

def main():

    # Read configuration file
    fl_config = config.Config(args.config)  # Config对象

    # 算法3.1：每个round，每个cluster设置不同的通信interval，
    # 以及system-level client selection
    env = FL_HIST_PARTIAL_STAGE(fl_config)

    env.make_client(L_hop=fl_config.data.num_layers)
    # env.make_client_iid(L_hop=fl_config.data.num_layers, iid_percent=0)

    agent_results = []
    plot_x = np.zeros(0)
    plot_y = np.zeros(0)
    plot_reward = np.zeros(0)
    plot_acc = np.zeros(0)
    # plot_cost = np.zeros(0)
    plot_all_reward = np.zeros(0)
    print("model:%s, target_acc:%.4f, num_clients:%d"
          % (fl_config.dataset_name, fl_config.data.target_accuracy, fl_config.clients.total))
    RL_path = './results/rl/interval_CS_RL_{}_{}client1.npz'.format(fl_config.dataset_name, fl_config.clients.total)
    reward_path ='./results/rl/interval_CS_RL_reward_{}_{}client1.npz'.format(fl_config.dataset_name, fl_config.clients.total)
    actor_path = './results/rl/interval_CS_actor_model_{}_{}client1'.format(fl_config.dataset_name, fl_config.clients.total)
    val_actor_path = './results/rl/interval_CS_val_actor_model_{}_{}client1'.format(fl_config.dataset_name, fl_config.clients.total)

    for run in range(RUNS):
        agent = SACAgent(env)  # 用于client selection
        val_agent = VALAgent(env)  # 用于设置communication interval
        run_results = []  # 只记录evaluation_episode的reward
        episode_reward_list = []  # 记录每个episode的reward
        val_episode_reward_list = []  # 记录val_agent每个stage的reward
        for episode_number in range(EPISODES_PER_RUN):
            print('episode:', episode_number)
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            # evaluation_episode = False
            episode_reward = 0
            val_episode_reward = 0  # val_agent相关参数
            real_episode_reward = 0
            state = env.get_initiate_state()  # state初始化
            val_state = env.get_inite_cluster_state_loss()  # val_state初始化

            done = False
            # for ep in tqdm(range(MAX_EP_STEP)):
            val_action = []
            for ep in range(MAX_EP_STEP):
                if done:
                    break

                if ep % fl_config.fl.stages == 0:
                    env.val_state = env.val_next_state
                    val_action = val_agent.get_next_action(env.val_state, evaluation_episode=evaluation_episode)

                env.val_action = copy.copy(val_action)
                env.synchronize_hists(ep, val_action)

                # RL选action，即selected clients以及cluster local iteration
                # 这里返回值，修改成action_list的index，
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                next_state, acc, reward, done = env.step(ep, action)  # one round FL training
                if ep == MAX_EP_STEP - 1:  # 最后一个episode
                    # reward = reward + (fl_config.data.target_accuracy - acc) * 100
                    total_comp_time = np.array(env.computation_time).sum()
                    tot_comm_time = np.array(env.communnication_time).sum()
                    total_runtime = np.array(env.epoch_runtime).sum()

                    print("total_comp_time:", total_comp_time, "tot_comm_time:", tot_comm_time, "total_runtime:",
                          total_runtime)

                if not evaluation_episode:  # 状态转移,调整神经网络
                    agent.train_on_transition(state, action, next_state, reward, done)
                    print("transition:", state, action, next_state, reward)
                if (ep + 1) % fl_config.fl.stages == 0:  # 为一个FL stage结束
                    val_agent.train_on_transition(env.val_state, env.val_action, env.val_next_state, env.val_reward, done)
                    val_episode_reward += env.val_reward
                    print("val transition:", env.val_state, env.val_action, env.val_next_state, env.val_reward)
                episode_reward += reward
                # real_episode_reward += real_reward

                plot_x = np.append(plot_x, MAX_EP_STEP * run + ep)
                plot_acc = np.append(plot_acc, acc)
                plot_reward = np.append(plot_reward, reward)
                # plot_cost = np.append(plot_cost, cost)
                np.savez(RL_path, plot_x, plot_acc, plot_reward)
                state = next_state
            if evaluation_episode:
                print(f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
                print("episoed:%d, reward:%.5f ,real_reward:%.5f, ep:%d" % (
                episode_number, episode_reward, real_episode_reward, ep))
                run_results.append(episode_reward)
            # run_results.append(episode_reward)

            episode_reward_list.append(episode_reward)
            val_episode_reward_list.append(val_episode_reward)
            plot_y = np.append(plot_y, run)
            plot_all_reward = np.append(plot_all_reward, real_episode_reward)
            np.savez(reward_path, plot_y, plot_all_reward)
            torch.save(agent.actor_local.state_dict(), actor_path)  # 保存CS agent的actor网络参数
            torch.save(val_agent.actor_local.state_dict(), val_actor_path)  # 保存interval setting agent的actor网络参数
        agent_results.append(run_results)

        write_result(fl_config, run, episode_reward_list, val_episode_reward_list)
    # env.close()

    # n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    n_results = EPISODES_PER_RUN
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    # x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    dataset_name = fl_config.dataset_name + "_partial_stage"
    env.plot_reward_result(x_vals, results_mean, mean_plus_std, mean_minus_std, dataset_name)


def write_result(config, run, episode_reward_list, val_episode_reward_list):
    filename = config.dataset_name + '_reward'
    # 先删除历史文件
    path = config.paths.model + "/"
    # model_files = os.listdir(path)
    # for i, f in enumerate(model_files):
    #     if f.find(filename) >= 0:
    #         os.remove(path + f)

    # 写文件
    a = open(path + filename, 'a+')

    a.write(str(run) + ':' + str(episode_reward_list) + '\n')
    a.write(str(run) + ':' + str(val_episode_reward_list) + '\n')
    a.close()


def rl_model_test():
    # Read configuration file
    fl_config = config.Config(args.config)  # Config对象

    # 算法3.1：每个round，每个cluster选择不同的clients交换emb.，
    # 而不是参与training，貌似会影响单个client的model accuracy
    env = FL_HIST_PARTIAL_STAGE(fl_config)

    env.make_client(L_hop=fl_config.data.num_layers)

    agent_results = []
    plot_x = np.zeros(0)
    plot_y = np.zeros(0)
    plot_reward = np.zeros(0)
    plot_acc = np.zeros(0)
    # plot_cost = np.zeros(0)
    plot_all_reward = np.zeros(0)
    print("model:%s, target_acc:%.4f, num_clients:%d"
          % (fl_config.dataset_name, fl_config.data.target_accuracy, fl_config.clients.total))
    RL_path = './results/rl/hist_partial_RL_{}_{}client1.npz'.format(fl_config.dataset_name, fl_config.clients.total)
    reward_path = './results/rl/hist_partial_RL_reward_{}_{}client1.npz'.format(fl_config.dataset_name,
                                                                                fl_config.clients.total)
    actor_path = './results/rl/hist_partial_actor_model_{}_{}client1'.format(fl_config.dataset_name,
                                                                             fl_config.clients.total)
    print(np.load(reward_path))

if __name__ == "__main__":
    main()
    # test()