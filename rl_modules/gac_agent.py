import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import ReplayBuffer
from rl_modules.gac_models import actor, critic, mmd

from logx import EpochLogger

"""
gac with HER (MPI-version)

"""
class gac_agent:
    def __init__(self, args, env, test_env, env_params):
        self.args = args

        # path to save the model
        if self.args.mmd:
            self.exp_name = '_'.join((self.args.env_name, self.args.alg, 
                                    'mmd'+str(self.args.beta_mmd), 
                                    's'+str(self.args.seed), 
                                    datetime.now().isoformat()))
            self.data_path = os.path.join(self.args.save_dir, 
                                    '_'.join((self.args.env_name, 
                                            self.args.alg, 
                                            'mmd'+str(self.args.beta_mmd))),
                                    self.exp_name)
        else:
            self.exp_name = '_'.join((self.args.env_name, self.args.alg, 
                        str(self.args.seed), datetime.now().isoformat()))
            self.data_path = os.path.join(self.args.save_dir, 
                    '_'.join((self.args.env_name, self.args.alg)),
                    self.exp_name)
        self.logger = EpochLogger(output_dir=self.data_path, exp_name=self.exp_name)
        self.logger.save_config(args)

        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network1 = critic(env_params)
        self.critic_network2 = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network1)
        sync_networks(self.critic_network2)
        # build up the target network
        # self.actor_target_network = actor(env_params)
        self.critic_target_network1 = critic(env_params)
        self.critic_target_network2 = critic(env_params)
        # load the weights into the target networks
        # self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())

        # if use gpu
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.mpi_size = MPI.COMM_WORLD.Get_size()
        if args.cuda:
            device = 'cuda:{}'.format(self.rank % torch.cuda.device_count())
        self.device = torch.device(device)

        if self.args.cuda:
            self.actor_network.cuda(self.device)
            self.critic_network1.cuda(self.device)
            self.critic_network2.cuda(self.device)
            # self.actor_target_network.cuda(self.device)
            self.critic_target_network1.cuda(self.device)
            self.critic_target_network2.cuda(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)
        self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)

        # create the replay buffer
        self.buffer = ReplayBuffer(self.env_params['obs'], self.env_params['action'], self.args.buffer_size)

        self.logger.setup_pytorch_saver(self.actor_network)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        obs, ep_rew, ep_cost, ep_len, done = self.env.reset(), 0, 0, 0, False
        for t in range(self.args.warmup_steps) :
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            ep_rew += reward
            ep_cost += info['cost']
            ep_len += 1
            self.buffer.store(obs, action, reward, info['cost'], next_obs, done)
            obs = next_obs
            if done or (ep_len == self.env_params['max_timesteps']):
                obs, ep_rew, ep_cost, ep_len, done = self.env.reset(), 0, 0, 0, False

        for epoch in range(self.args.n_epochs):
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    action = self.actor_network(input_tensor)
                    action = action.detach().cpu().numpy().squeeze()
                # feed the actions into the environment
                next_obs, reward, done, info = self.env.step(action * self.env_params['action_max'])
                ep_rew += reward
                ep_cost += info['cost']
                ep_len += 1
                self.buffer.store(obs, action, reward, info['cost'], next_obs, done)
                obs = next_obs

                if done or (ep_len == self.env_params['max_timesteps']) or (t % self.args.n_batches == 0):
                    self.buffer.obs_mean = MPI.COMM_WORLD.allreduce(self.buffer.obs_mean, op=MPI.SUM)/self.mpi_size
                    self.buffer.obs_std = MPI.COMM_WORLD.allreduce(self.buffer.obs_std, op=MPI.SUM)/self.mpi_size

                    for _ in range(self.args.n_batches):
                        # train the network
                        self._update_network()
                        # soft update
                        # self._soft_update_target_network(self.actor_target_network, self.actor_network)
                        self._soft_update_target_network(self.critic_target_network1, self.critic_network1, self.args.polyak)
                        self._soft_update_target_network(self.critic_target_network2, self.critic_network2, self.args.polyak)

                if done or (ep_len == self.env_params['max_timesteps']):
                    global_rew = MPI.COMM_WORLD.allreduce(ep_rew, op=MPI.SUM)/self.mpi_size
                    global_cost = MPI.COMM_WORLD.allreduce(ep_cost, op=MPI.SUM)/self.mpi_size
                    global_len = MPI.COMM_WORLD.allreduce(ep_len, op=MPI.SUM)/self.mpi_size
                    self.logger.store(EpReward=global_rew, EpCost=global_cost, EpLen=global_len)
                    obs, ep_rew, ep_cost, ep_len, done = self.env.reset(), 0, 0, 0, False

            # start to do the evaluation
            self._test_policy()

            # save some necessary objects
            state = {'observation_mean':self.buffer.obs_mean, 'observation_std':self.buffer.obs_std}
            self.logger.save_state(state, None)

            t = ((epoch+1) * self.mpi_size * self.env_params['max_timesteps'])

            self.logger.log_tabular('Epoch', epoch+1)
            self.logger.log_tabular('EpReward')
            self.logger.log_tabular('EpCost')
            self.logger.log_tabular('EpLen')
            self.logger.log_tabular('TestReward')
            self.logger.log_tabular('TestCost')
            self.logger.log_tabular('TestLen')
            self.logger.log_tabular('LossPi')
            self.logger.log_tabular('LossQ')
            self.logger.log_tabular('MMDEntropy')
            self.logger.log_tabular('TotalEnvInteracts', t)
            self.logger.dump_tabular()

    # pre_process the inputs
    def _preproc_inputs(self, obs):
        inputs = self.buffer.obs_encoder(obs)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda(self.device)
        return inputs

    # soft update
    def _soft_update_target_network(self, target, source, polyak):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - polyak) * param.data + polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        batches = self.buffer.sample(self.args.batch_size)
        
        o = torch.FloatTensor(batches['obs']).to(self.device)
        o2 = torch.FloatTensor(batches['obs2']).to(self.device)
        a = torch.FloatTensor(batches['act']).to(self.device)
        r = torch.FloatTensor(batches['rew']).to(self.device)
        c = torch.FloatTensor(batches['cost']).to(self.device)
        d = torch.FloatTensor(batches['done']).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            a2 = self.actor_network(o2)
            q_next_value1 = self.critic_target_network1(o2, a2).detach()
            q_next_value2 = self.critic_target_network2(o2, a2).detach()
            target_q_value = r + self.args.gamma * (1 - d) * torch.min(q_next_value1, q_next_value2)
            target_q_value = target_q_value.detach()
        # the q loss
        real_q_value1 = self.critic_network1(o, a)
        real_q_value2 = self.critic_network2(o, a)
        critic_loss1 = (target_q_value - real_q_value1).pow(2).mean()
        critic_loss2 = (target_q_value - real_q_value2).pow(2).mean()

        # the actor loss
        o_exp = o.repeat(self.args.expand_batch, 1)
        a_exp = self.actor_network(o_exp)
        actor_loss = -torch.min(self.critic_network1(o_exp, a_exp),
                    self.critic_network2(o_exp, a_exp)).mean()

        mmd_entropy = torch.tensor(0.0)

        if self.args.mmd:
            # mmd is computationally expensive
            a_exp_reshape = a_exp.view(self.args.expand_batch, -1, a_exp.shape[-1]).transpose(0, 1)
            with torch.no_grad():
                uniform_actions = (2 * torch.rand_like(a_exp_reshape) - 1)
            mmd_entropy = mmd(a_exp_reshape, uniform_actions)
            if self.args.beta_mmd <= 0.0:
                mmd_entropy.detach_()
            else:
                actor_loss += self.args.beta_mmd * mmd_entropy

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        sync_grads(self.critic_network1)
        self.critic_optim1.step()
        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        sync_grads(self.critic_network2)
        self.critic_optim2.step()

        self.logger.store(LossPi=actor_loss.detach().cpu().numpy())
        self.logger.store(LossQ=(critic_loss1+critic_loss2).detach().cpu().numpy())
        self.logger.store(MMDEntropy=mmd_entropy.detach().cpu().numpy())

    # do the evaluation
    def _test_policy(self):
        for _ in range(self.args.n_test_rollouts):
            obs, ep_rew, ep_cost, ep_len, done = self.test_env.reset(), 0, 0, 0, False
            while(not done and ep_len < self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    action = self.actor_network(input_tensor, std=0.5)
                    action = action.detach().cpu().numpy().squeeze()
                obs_next, reward, done, info = self.test_env.step(action)
                obs = obs_next
                ep_rew += reward
                ep_cost += info['cost']
                ep_len += 1
            global_rew = MPI.COMM_WORLD.allreduce(ep_rew, op=MPI.SUM)/self.mpi_size
            global_cost = MPI.COMM_WORLD.allreduce(ep_cost, op=MPI.SUM)/self.mpi_size
            global_len = MPI.COMM_WORLD.allreduce(ep_len, op=MPI.SUM)/self.mpi_size
            self.logger.store(TestReward=global_rew, TestCost=global_cost, TestLen=global_len)