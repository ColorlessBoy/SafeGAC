import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.sac_models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

from logx import EpochLogger
from mpi4py import MPI

"""
gac with HER (MPI-version)

"""
class sac_agent:
    def __init__(self, args, env, env_params):
        self.args = args

        # path to save the model
        self.exp_name = '_'.join((self.args.env_name, self.args.alg, 
                    str(self.args.seed), datetime.now().isoformat()))
        self.data_path = os.path.join(self.args.save_dir, 
                '_'.join((self.args.env_name, self.args.alg)),
                self.exp_name)
        self.logger = EpochLogger(output_dir=self.data_path, exp_name=self.exp_name)
        self.logger.save_config(args)

        self.env = env
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
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        self.logger.setup_pytorch_saver(self.actor_network)

        # auto temperature
        if self.args.alpha < 0.0:
            # if self.args.alpha < 0.0, 
            # sac will use auto temperature and init alpha = - self.args.alpha
            self.alpha = -self.args.alpha
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, 
                            device=device, requires_grad=True)
            self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_actor)
        else:
            self.alpha = self.args.alpha

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi, _ = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                # self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network1, self.critic_network1)
                self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
            # start to do the evaluation
            success_rate = self._eval_agent()

            # save some necessary objects
            # self.logger.save_state will also save pytorch's model implicitly.
            # self.logger.save_state({'env':self.env, 'o_norm':self.o_norm, 'g_norm':self.g_norm}, None)
            state = {'env':self.env, 'o_norm':self.o_norm.get(), 'g_norm':self.g_norm.get()}
            self.logger.save_state(state, None)

            t = ((epoch+1) * self.args.n_cycles * 
                    self.args.num_rollouts_per_mpi * 
                    MPI.COMM_WORLD.Get_size() * 
                    self.env_params['max_timesteps'])

            self.logger.log_tabular('Epoch', epoch+1)
            self.logger.log_tabular('SuccessRate', success_rate)
            self.logger.log_tabular('LossPi')
            self.logger.log_tabular('LossQ')
            self.logger.log_tabular('Entropy')
            self.logger.log_tabular('TotalEnvInteracts', t)
            self.logger.dump_tabular()

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda(self.device)
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        # action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        # action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        # random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
        #                                     size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) * self.args.reward_scale
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda(self.device)
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda(self.device)
            actions_tensor = actions_tensor.cuda(self.device)
            r_tensor = r_tensor.cuda(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next, log_prob_actions_next = self.actor_network(inputs_next_norm_tensor)
            q_next_value1 = self.critic_target_network1(inputs_next_norm_tensor, actions_next).detach()
            q_next_value2 = self.critic_target_network2(inputs_next_norm_tensor, actions_next).detach()
            target_q_value = r_tensor + self.args.gamma * (torch.min(q_next_value1, q_next_value2) 
                                        - self.args.alpha * log_prob_actions_next)
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value1 = self.critic_network1(inputs_norm_tensor, actions_tensor)
        real_q_value2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
        critic_loss1 = (target_q_value - real_q_value1).pow(2).mean()
        critic_loss2 = (target_q_value - real_q_value2).pow(2).mean()

        # the actor loss
        actions, log_prob_actions = self.actor_network(inputs_norm_tensor)
        log_prob_actions = log_prob_actions.mean()
        actor_loss = self.args.alpha * log_prob_actions - torch.min(self.critic_network1(inputs_norm_tensor, actions),
                    self.critic_network2(inputs_norm_tensor, actions)).mean()

        # actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

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
        self.logger.store(Entropy=log_prob_actions.detach().cpu().numpy())

        # auto temperature
        if self.args.alpha < 0:
            logalpha_loss = -self.log_alpha * (log_prob_actions.detach() + self.target_entropy)
            self.alpha_optim.zero_grad()
            logalpha_loss.backward()
            comm = MPI.COMM_WORLD
            local_grad = self.log_alpha.grad.detach().cpu().numpy()
            global_grads = np.zeros_like(local_grad)
            comm.Allreduce(local_grad, global_grads, op=MPI.SUM)
            self.log_alpha.grad = torch.tensor(global_grads, dtype=torch.float32, device=self.device)
            self.alpha_optim.zero_grad()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi, _ = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
