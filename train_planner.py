#!/usr/bin/env python3
import argparse
import os
from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO, A2C, DDPG, PPO1, PPO2, SAC, ACER, ACKTR, GAIL, DQN, HER, TD3, logger
import gym

import time
import numpy as np
import tensorflow as tf
from typing import Dict
from keras.models import load_model
from matplotlib import pyplot as plt
#from tensor_board_cb import TensorboardCallback
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
import gym_SmartLoader.envs
from LLC import LLC_pid

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.sac.policies import SACPolicy, gaussian_entropy, gaussian_likelihood, apply_squashing_func, mlp, nature_cnn

# for custom callbacks stable-baselines should be upgraded using -
# pip3 install stable-baselines[mpi] --upgrade
from stable_baselines.common.callbacks import BaseCallback

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'gail': GAIL
}
JOBS = ['train', 'record', 'BC_agent', 'play']

POLICIES = ['MlpPolicy', 'CnnPolicy','CnnMlpPolicy']

BEST_MODELS_NUM = 0


EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def CnnMlpFeatureExtractor(obs, image_height = 291, image_width = 150):
    image_size = image_height * image_width
    assert len(obs.shape) == 2 and obs.shape[1] > image_size
    feature_num = obs.shape[1] - image_size
    grid_map_flat = tf.slice(obs, [0, 0], [1, image_size])
    grid_map = tf.reshape(grid_map_flat, [1, image_height, image_width, 1])
    extracted_features = nature_cnn(grid_map)
    extracted_features = tf.layers.flatten(extracted_features)
    # extracted_features = tf.Print(extracted_features,[tf.shape(extracted_features)], 'extracted features shape : ')
    features = tf.slice(obs, [0, image_size], [1, feature_num])
    # features = tf.Print(features, [tf.shape(features)], "features shape: " )
    return tf.concat([extracted_features, features], 1)



class CustomSacCnnMlpPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(CustomSacCnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun



    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            # if self.feature_extraction == "cnn":
            #     pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            # else:
            #     pi_h = tf.layers.flatten(obs)

            pi_h = CnnMlpFeatureExtractor(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            # if self.feature_extraction == "cnn":
            #     critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            # else:
            #     critics_h = tf.layers.flatten(obs)
            critics_h = CnnMlpFeatureExtractor(obs)
            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            # if create_qf and action.get_shape().as_list()[0] == critics_h.get_shape().as_list()[0]:
            if create_qf:

                # action = tf.Print(action, [tf.shape(action)], "action shape: ")
                # critics_h = tf.Print(critics_h, [tf.shape(critics_h)], "critics_h shape: ")

                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})




# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CnnMlpPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            feature_num = 7
            h = 291
            w = 150
            proc_obs_float = tf.cast(self.processed_obs, dtype=tf.float32)
            grid_map_flat = tf.slice(proc_obs_float,[0,0],[1,43650] )
            grid_map = tf.reshape(grid_map_flat, [1,h,w,1])
            # kwargs['data_format']='NCHW'
            extracted_features = nature_cnn(grid_map, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)
            features =  tf.slice(proc_obs_float, [0, 43650], [1, feature_num])
            pi_h = tf.concat([extracted_features,features], 1)
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = tf.concat([extracted_features,features], 1)
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    # def _custom_nature_cnn(self):
    #     activ = tf.nn.relu
    #     layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), dataNHWC **kwargs))
    #     layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    #     layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #     layer_3 = conv_to_fc(layer_3)
    #     return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def expert_dataset(name):
    # Benny's recordings to dict
    path = os.getcwd() + '/' + name
    numpy_dict = {
        'actions': np.load(path + '/act.npy'),
        'obs': np.load(path + '/obs.npy'),
        'rewards': np.load(path + '/rew.npy'),
        'episode_returns': np.load(path + '/ep_ret.npy'),
        'episode_starts': np.load(path + '/ep_str.npy')
    }  # type: Dict[str, np.ndarray]

    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    # dataset = TemporaryFile()
    save_path = os.getcwd() + '/dataset'
    os.makedirs(save_path)
    np.savez(save_path, **numpy_dict)

class ExpertDatasetLoader:
    dataset = None

    def __call__(self, force_load=False):
        if ExpertDatasetLoader.dataset is None or force_load:
            print('loading expert dataset')
            ExpertDatasetLoader.dataset = ExpertDataset(expert_path=(os.getcwd() + '/dataset.npz'), traj_limitation=-1)
        return ExpertDatasetLoader.dataset

class CheckEvalCallback(BaseCallback):
    """
    A custom callback that checks agent's evaluation every predefined number of steps.
    :param model_dir: (str) directory path for model save
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param save_interval: (int) Number of timestamps between best mean model saves
    """

    def __init__(self, model_dir, verbose, save_interval=2000):
        super(CheckEvalCallback, self).__init__(verbose)
        self._best_model_path = model_dir
        self._last_model_path = model_dir
        self._best_mean_reward = -np.inf
        self._save_interval = save_interval
        self._best_rew = -np.inf
        self._ep_rew = []
        self._mean_10_ep = -2
        self._last_total_reward = -2

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print('_on_rollout_start')

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """


        env = self.locals['self'].env.unwrapped.envs[0]

        if env.done:
            self._ep_rew.append(self._last_total_reward)
            #            self._ep_rew.append(env.total_reward)
            if len(self._ep_rew) % 10 == 0:
                self._mean_10_ep = np.mean(self._ep_rew[-11:-1])
                self._ep_rew = []
        self._last_total_reward = env.total_reward

        #rew = self.locals['self'].episode_reward[0]
        # if (self.num_timesteps + 1) % self._save_interval == 0:
        #if (rew > self._best_rew):
            # Evaluate policy training performance

            # episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
            #                                                    n_eval_episodes=100,
            #                                                    render=False,
            #                                                    deterministic=True,
            #                                                    return_episode_rewards=True)



            # mean_reward = round(float(np.mean(self.locals['episode_rewards'][-101:-1])), 1)


            # print(self.num_timesteps + 1, 'timesteps')
            # print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(self._best_mean_reward, mean_reward))
            #print("Best  reward: {:.2f} - Last best reward: {:.2f}".format(self._best_rew, rew))
        #New best model, save the agent
        if self._mean_10_ep > self._best_mean_reward:
            print("Saving new best model:"+str(np.round(self._mean_10_ep, 2)) + " last best: " + str(np.round(self._best_mean_reward, 2)))
            self._best_mean_reward = self._mean_10_ep
            self.model.save(self._best_model_path + '_rew_' + str(np.round(self._best_mean_reward, 2)))
            #self._best_rew = rew
            #print("Saving new best model")
            # self.model.save(self._best_model_path + '_rew_' + str(np.round(self._best_rew, 2)))
            path = self._last_model_path + '_' + str(time.localtime().tm_mday) + '_' + str(
                 time.localtime().tm_hour) + '_' + str(time.localtime().tm_min)
            # global BEST_MODELS_NUM
            # BEST_MODELS_NUM=BEST_MODELS_NUM+1
            self.model.save(path)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print('_on_rollout_end')
        # print('locals', self.locals)
        # print('globals', self.globals)
        # print('n_calls', self.n_calls)
        # print('num_timesteps', self.num_timesteps)
        # print('training_env', self.training_env)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print('_on_training_end')


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

        self._ep_rew = []
        self._mean_10_ep = -2
        self._last_total_reward = -2


    def _on_step(self) -> bool:
        # Log additional tensor
        # if not self.is_tb_set:
        #     with self.model.graph.as_default():
        #         tf.summary.scalar('episode_reward', tf.reduce_mean(self.model.episode_reward))
        #         # tf.summary.scalar('episode_reward', tf.reduce_mean(self.model.episode_reward))
        #         self.model.summary = tf.summary.merge_all()
        #     self.is_tb_set = True
        # # Log scalar value (here a random variable)
        #
        #
        # global BEST_MODELS_NUM
        # value = BEST_MODELS_NUM
        #
        #
        # writer = self.locals['writer']

        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('episode_reward', tf.reduce_mean(self.model.episode_reward))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True
            # Log scalar value (here a random variable)

        env = self.locals['self'].env.unwrapped.envs[0]

        if env.done:
            self._ep_rew.append(self._last_total_reward)
#            self._ep_rew.append(env.total_reward)
            if len(self._ep_rew) % 10 == 0:
                self._mean_10_ep = np.mean(self._ep_rew[-11:-1])
        self._last_total_reward = env.total_reward
        summary = tf.Summary(value=[tf.Summary.Value(tag='last_rt', simple_value=env.last_rt),
                                    tf.Summary.Value(tag='last_final_reward', simple_value=env.last_final_reward),
                                    tf.Summary.Value(tag='total_reward', simple_value=env.total_reward),
                                    tf.Summary.Value(tag='mean_10_ep', simple_value=self._mean_10_ep)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True

        # env = self.locals['self'].env.unwrapped.envs[0]
        # summary1 = tf.Summary(value=[tf.Summary.Value(tag='best_models', simple_value=value)])
        # summary2 = tf.Summary(value=[tf.Summary.Value(tag='last_rt', simple_value=env.last_rt)])
        # summary3 = tf.Summary(value=[tf.Summary.Value(tag='last_final_reward', simple_value=env.last_final_reward)])
        # self.locals['writer'].add_summary(summary1, self.num_timesteps)
        # self.locals['writer'].add_summary(summary2, self.num_timesteps)
        # self.locals['writer'].add_summary(summary3, self.num_timesteps)
        # return True

def data_saver(obs, act, rew, dones, ep_rew):
    user = os.getenv("HOME")
    np.save(user+'/git/SmartLoader/saved_ep/obs', obs)
    np.save(user+'/git/SmartLoader/saved_ep/act', act)
    np.save(user+'/git/SmartLoader/saved_ep/rew', rew)

    ep_str = [False] * len(dones)
    ep_str[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            ep_str[i + 1] = True

    np.save(user+'/git/SmartLoader/saved_ep/ep_str', ep_str)
    np.save(user+'/git/SmartLoader/saved_ep/ep_ret', ep_rew)


def build_model(algo, policy, env_name, log_dir, expert_dataset=None):
    """
    Initialize model according to algorithm, architecture and hyperparameters
    :param algo: (str) Name of rl algorithm - 'sac', 'ppo2' etc.
    :param env_name:(str)
    :param log_dir:(str)
    :param expert_dataset:(ExpertDataset)
    :return:model: stable_baselines model
    """
    from stable_baselines.common.vec_env import DummyVecEnv
    model = None
    if algo == 'sac':
        # policy_kwargs = dict(layers=[64, 64, 64],layer_norm=False)

        # model = SAC(policy, env_name, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
        #             learning_starts=5000, train_freq=500, batch_size=64, policy_kwargs=policy_kwargs,
        #             tau=0.01, ent_coef='auto_0.1', target_update_interval=1,
        #             gradient_steps=1, target_entropy='auto', action_noise=None,
        #             random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
        #             _init_setup_model=True, full_tensorboard_log=True,
        #             seed=None, n_cpu_tf_sess=None)

        # SAC - start learning from scratch
        # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32, 32, 32])
        policy_kwargs = dict(layers=[32, 32, 32], layer_norm=False)

        env = DummyVecEnv([lambda: gym.make(env_name)])
        # model = A2C(CnnMlpPolicy, env, verbose=1,gamma=0.99, learning_rate=1e-4,  tensorboard_log=log_dir, _init_setup_model=True, full_tensorboard_log=True,seed=None, n_cpu_tf_sess=None)


        model = SAC(CustomSacCnnMlpPolicy, env=env, gamma=0.99, learning_rate=1e-4, buffer_size=50000,
                    learning_starts=1000, train_freq=100, batch_size=1,
                    tau=0.01, ent_coef='auto', target_update_interval=1,
                    gradient_steps=1, target_entropy='auto', action_noise=None,
                    random_exploration=0.0, verbose=1, tensorboard_log=log_dir,
                    _init_setup_model=True, full_tensorboard_log=True,
                    seed=None, n_cpu_tf_sess=None)

    elif algo == 'ppo1':
        model = PPO1('MlpPolicy', env_name, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2,
                     entcoeff=0.01,
                     optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                     schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
                     policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    elif algo == 'trpo':
        model = TRPO('MlpPolicy', env_name, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
    elif algo == 'gail':
        assert expert_dataset is not None
        model = GAIL('MlpPolicy', env_name, expert_dataset, tensorboard_log=log_dir, verbose=1)
    assert model is not None
    return model


def pretrain_model(dataset, model):
    # load dataset only once
    # expert_dataset('3_rocks_40_episodes')
    assert (dataset in locals() or dataset in globals()) and dataset is not None
    print('pretrain')
    model.pretrain(dataset, n_epochs=2000)


def record(env):
    num_episodes = 10
    obs = []
    actions = []
    rewards = []
    dones = []
    episode_rewards = []
    for episode in range(num_episodes):

        ob = env.reset()
        done = False
        print('Episode number ', episode)
        episode_reward = 0

        while not done:
            act = "recording"
            new_ob, reward, done, action = env.step(act)

            # ind = [0, 1, 2, 18, 21, 24]
            ind = [0, 1, 2]
            # print(ob)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            episode_reward = episode_reward + reward

            ob = new_ob

        episode_rewards.append(episode_reward)
    data_saver(obs, actions, rewards, dones, episode_rewards)

def h_map_func(obs, pass_num):

    hmap = obs['h_map']
    hmap = (hmap - np.min(hmap)) / np.ptp(hmap)
    hmap = hmap.reshape(291, 150)


    Ax, Bx = 9.7, 291
    Ay, By = 6, 0
    x_new = int(obs['y_vehicle'] * Ax + Bx)
    y_new = int(obs['x_vehicle'] * Ay + By)
    # print('xold: ', obs['y_vehicle'], 'xnew: ', x_new, 'yold: ', obs['x_vehicle'], 'ynew: ', y_new)

    pass_offset = -int(pass_num)

    x_hmap = hmap[0:260, y_new - 70:y_new - 10]
    LP_hmap = hmap[x_new + 70 + pass_offset:x_new + 120 + pass_offset, y_new - 70:y_new - 10]
    # plt.imshow(LP_hmap)
    # plt.show(block=False)recg,   # plt.pause(0.01)
    return x_hmap, LP_hmap


def dump_load(env,push_pid):
    print('dumping load')
    obs = env.get_obs()
    des = [obs['y_vehicle'], obs['x_vehicle'], 245, 300]
    for _ in range(70):
        action = push_pid.step(obs, des)
        action[1] = 0
        obs, _, _, _ = env.step(action)
        # print('des x: ', des[0], 'x: ', 30 + obs['y_vehicle'], 'des lift: ', des[2], 'lift: ', obs['arm_lift'], 'des pitch: ',
        #       des[3], 'pitch: ', obs['arm_pitch'])


def move_back(env,push_pid):
    obs = env.get_obs()
    print('moving back')
    des = [5, obs['x_vehicle'], 226, 352]
    for _ in range(70):

        action = push_pid.step(obs, des)
        obs, _, _, _ = env.step(action)
        # print('des x: ', des[0], 'x: ', 30 + obs['y_vehicle'], 'des lift: ', des[2], 'lift: ', obs['arm_lift'], 'des pitch: ',
        #       des[3], 'pitch: ', obs['arm_pitch'])
        print('des x: ', des[0], 'x: ', 30 + obs['y_vehicle'])
    push_pid.lift_pid.save_plot('lift', 'lift')
    push_pid.pitch_pid.save_plot('pitch', 'pitch')
    push_pid.speed_pid.save_plot('speed', 'speed')


def play(save_dir, env):
    # model = SAC.load(save_dir + '/model_dir/sac/test_25_25_14_15', env=env,
    #                  custom_objects=dict(learning_starts=0))  ### ADD NUM1
    # model = SAC.load(save_dir + '/model_dir/sac/test_6_24_17_30', env=env,
    #                  custom_objects=dict(learning_starts=0))  ### ADD NUM

    x_model = load_model('/home/iaiai/git/SmartLoader/Real_agents/new_test_all_recordings_x_model_10_pred')
    LP_model = load_model('/home/iaiai/git/SmartLoader/Real_agents/new_test_new_recordings_LP_model_10_pred')

    # x_model = load_model('/home/iaiai/git/SmartLoader/Real_agents/lift_task_x_model_10_pred')
    # LP_model = load_model('/home/iaiai/git/SmartLoader/Real_agents/lift_task_LP_model_10_pred')


    obs = env.reset()
    push_pid = LLC_pid.PushPidAlgoryx()
    # driveBack_pid = LLC_pid.DriveBackAndLiftPidAlgoryx()
    # done = False

    # while True: ## test loop
    #     obs = env.get_obs()
    #     # action = [0.1, 0, 0, 0]
    #     # env.step(action)
    #     # time.sleep(0.05)

        # print('x= ', obs['x_vehicle'], 'y= ', obs['y_vehicle'])
    num_of_passes = 5
    x_end = 13
    for pass_num in range(num_of_passes):

        print('pass number ', pass_num)
        done = False

        while not done:
        # while True:
        # for _ in range(1000):

            # time.sleep(0.1)

            x_hmap, LP_hmap = h_map_func(obs, pass_num)

            norm_x_des = x_model.predict(x_hmap.reshape(1, 1, 260, 60))
            norm_LP_des = LP_model.predict(LP_hmap.reshape(1, 1, 50, 60))

            norm_lift_des = norm_LP_des[0, np.arange(0, 20, 2)]
            norm_pitch_des = norm_LP_des[0, np.arange(1, 20, 2)]

            lift_offset = 32
            pitch_offest = 10

            lift_des = norm_lift_des[0]*140+140+lift_offset
            pitch_des = norm_pitch_des[0]*293+220+pitch_offest
            x_des = norm_x_des[0, 0]*30

            lift = obs['arm_lift']
            pitch = obs['arm_pitch']

            # des = [x_blade, y_blade, lift, pitch]
            des = [x_des, obs['x_vehicle'], lift_des, pitch_des]

            action = push_pid.step(obs, des)
            # if obs['y_vehicle']-x_des < 0.0
            obs, _, done, _ = env.step(action)
            # obs = env.get_obs()

            # print('des x: ', x_des, 'x: ', 30+obs['y_vehicle'], 'des lift: ', lift_des, 'lift: ', lift, 'des pitch: ', pitch_des, 'pitch: ', pitch)
            print('des x: ', des[0], 'x: ', 30 + obs['y_vehicle'])
            print(action)
            if 30+obs['y_vehicle'] > x_end:#+pass_num*0.4:
                done = True

        # push_pid.lift_pid.save_plot('lift', 'lift')
        # push_pid.pitch_pid.save_plot('pitch', 'pitch')
        # push_pid.speed_pid.save_plot('speed', 'speed')

        dump_load(env,push_pid)
        move_back(env,push_pid)
    # done = False
    # env.ref_pos = env._start_pos
    #
    # while not done:
    #     x_hmap, LP_hmap = h_map_func(obs)
    #
    #     norm_x_des = x_model.predict(x_hmap.reshape(1, 1, 260, 60))
    #     norm_LP_des = LP_model.predict(LP_hmap.reshape(1, 1, 50, 60))
    #
    #     norm_lift_des = norm_LP_des[0, np.arange(0, 20, 2)]
    #     norm_pitch_des = norm_LP_des[0, np.arange(1, 20, 2)]
    #
    #     lift_des = norm_lift_des[0]*140+140
    #     pitch_des = norm_pitch_des[0]*293+220
    #     x_des = norm_x_des[0, 0]*260
    #
    #     des = [x_des, obs['y_vehicle'], lift_des, pitch_des]
    #
    #     action = push_pid.step(obs, des)
    #
    #     obs, _, done, _ = env.step(action)

    # for _ in range(2):
    #
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #         # print('state: ', obs[0:3], 'action: ', action)


def train(algo, policy, pretrain, n_timesteps, log_dir, model_dir, env_name, model_save_interval):
    """
    Train an agent
    :param algo: (str)
    :param policy: type of network (str)
    :param pretrain: (bool)
    :param n_timesteps: (int)
    :param log_dir: (str)
    :param model_dir: (str)
    :param env_name: (str)
    :return: None
    """
    dataset = ExpertDatasetLoader() if pretrain or algo == 'gail' else None
    model = build_model(algo=algo, policy=policy, env_name=env_name, log_dir=log_dir, expert_dataset=dataset)
    if pretrain:
        pretrain_model(dataset, model)

    # learn
    print("learning model type", type(model))
    custom_eval_callback = CheckEvalCallback(model_dir, verbose=2, save_interval=model_save_interval)
    # eval_callback = EvalCallback(env_name=env_name, best_model_save_path=model_dir,
    #                              log_path='./logs/', eval_freq=model_save_interval,
    #                              deterministic=True, render=False, callback_on_new_best=on_new_best_model_cb)
    tensorboard_callback = TensorboardCallback(verbose=1)
    model.learn(total_timesteps=n_timesteps, callback=[custom_eval_callback, tensorboard_callback], tb_log_name="first_run_" + time.time().__str__())
    model.save(env_name)

def train_loaded(algo, policy, load_path, n_timesteps, log_dir, model_dir, env_name, model_save_interval):
    """
    Train an agent
    :param algo: (str)
    :param policy: type of network (str)
    :param load_path: model to load
    :param n_timesteps: (int)
    :param log_dir: (str)
    :param model_dir: (str)
    :param env: (gym env)
    :param env_name: (str)
    :return: None
    """
    #model = build_model(algo=algo, policy=policy, env_name=env_name, log_dir=log_dir, expert_dataset=None)
    #env_name = env.spec.id
    from stable_baselines.common.vec_env import DummyVecEnv
    model = None
    env = DummyVecEnv([lambda: gym.make(env_name)])
    if algo == "sac":
        model= SAC.load(load_path, env=env, policy=CustomSacCnnMlpPolicy, gamma=0.99, learning_rate=1e-4,  buffer_size=50000,
                    learning_starts=1000, train_freq=10, batch_size=1,
                    tau=0.01, ent_coef='auto', target_update_interval=1,
                    gradient_steps=1, target_entropy='auto', action_noise=None,
                    random_exploration=0.0, verbose=1, tensorboard_log=log_dir,
                    _init_setup_model=True, full_tensorboard_log=True,
                    seed=None, n_cpu_tf_sess=None)
    else:
        model= A2C.load(load_path, env=env, policy=CnnMlpPolicy, verbose=1,gamma=0.99, learning_rate=1e-4,  tensorboard_log=log_dir, _init_setup_model=True, full_tensorboard_log=True,seed=None, n_cpu_tf_sess=None)
    #model.set_env(env)
    # learn
    print("learning model type", type(model))
    custom_eval_callback = CheckEvalCallback(model_dir, verbose=2, save_interval=model_save_interval)
    # eval_callback = EvalCallback(env_name=env_name, best_model_save_path=model_dir,
    #                              log_path='./logs/', eval_freq=model_save_interval,
    #                              deterministic=True, render=False, callback_on_new_best=on_new_best_model_cb)
    tensorboard_callback = TensorboardCallback(verbose=1)
    model.learn(total_timesteps=n_timesteps, callback=[custom_eval_callback, tensorboard_callback], tb_log_name="from_learned_model_" + time.time().__str__())
    model.save(env_name)



def CreateLogAndModelDirs(args):
    '''
    Create log and model directories according to algorithm, time and incremental index
    :param args:
    :return:
    '''

    #
    dir = args.dir_pref + args.mission
    model_dir = dir + args.model_dir + args.algo
    log_dir = dir + args.tensorboard_log + args.algo
    os.makedirs(model_dir, exist_ok=True)
    # create new folder
    try:
        tests = os.listdir(model_dir)
        indexes = []
        for item in tests:
            indexes.append(int(item.split('_')[1]))
        if not bool(indexes):
            k = 0
        else:
            k = max(indexes) + 1
    except FileNotFoundError:
        os.makedirs(log_dir)
        k = 0
    suffix = '/test_{}'.format(str(k))
    model_dir = os.getcwd() + '/' + model_dir + suffix
    log_dir += suffix
    logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    print('log directory created', log_dir)
    return dir, model_dir, log_dir


def main(args):
    # register_policy('CnnMlpPolicy',CnnMlpPolicy)
    env_name = args.mission + '-' + args.env_ver
    env = gym.make(env_name)  # .unwrapped  <= NEEDED?
    print('gym env created', env_name, env)

    save_dir, model_dir, log_dir = CreateLogAndModelDirs(args)

    if args.job == 'train':
        model_path = args.load_model
        # If there is a path in load model, then load before training
        if model_path != "" and os.path.exists(model_path):
            train_loaded(args.algo, args.policy, model_path, args.n_timesteps, log_dir, model_dir, env_name,
                         args.save_interval)
        else:
            train(args.algo, args.policy, args.pretrain, args.n_timesteps, log_dir, model_dir, env_name, args.save_interval)
    elif args.job == 'record':
        record(env)
    elif args.job == 'play':
        play(save_dir, env)
    elif args.job == 'BC_agent':
        raise NotImplementedError
    else:
        raise NotImplementedError(args.job + ' is not defined')


def add_arguments(parser):
    parser.add_argument('--mission', type=str, default="PushAlgoryxEnv", help="The agents' task")
    parser.add_argument('--env-ver', type=str, default="v0", help="The custom gym enviornment version")
    parser.add_argument('--dir-pref', type=str, default="stable_bl/", help="The log and model dir prefix")

    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='/log_dir/', type=str)
    parser.add_argument('-mdl', '--model-dir', help='model directory', default='/model_dir/', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--policy', help='Network topography', default='CnnMlpPolicy', type=str, required=False, choices=POLICIES)

    parser.add_argument('--job', help='job to be done', default='play', type=str, required=False, choices=JOBS)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=int(1e6), type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
    parser.add_argument('--save-interval', help='Number of timestamps between model saves', default=2000, type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation', default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--pretrain', help='Evaluate pretrain phase', default=False, type=bool)
    parser.add_argument('--load-expert-dataset', help='Load Expert Dataset', default=False, type=bool)
    parser.add_argument('--load-model', help='Starting model to load', default="", type=str)
    # parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
    #                     help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    # parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
    #                     help='Ensure that the run has a unique ID')
    # parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
    #                     help='Optional keyword argument to pass to the env constructor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)