import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
import datetime

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import loggers
import wrappers


class Dreamer(tools.Module):

    def __init__(self, config, logger, dataset):
        self._config = config
        self._logger = logger
        self._float = prec.global_policy().compute_dtype
        self._should_pretrain = tools.Once()
        self._should_expl = tools.Until(int(config.expl_until))
        self._should_log_sample = tools.Every(config.log_sample_every)
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(config.traindir), dtype=tf.int64)
            self._batches_trained = tf.Variable(0, dtype=tf.int64)
        # Schedules.
        config.actor_entropy = (
            lambda x=config.actor_entropy: tools.schedule(x, self._step))
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
        config.imag_gradient_mix = (
            lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
        self._dataset = iter(dataset)
        self._wm = models.WorldModel(self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad)
        reward = lambda f, s, a: self._wm.heads['reward'](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]()
        # Train step to initialize variables including optimizer statistics.
        self._train(next(self._dataset))

    @property
    def step(self):
        return self._step.numpy().item()

    def inc_step(self, delta):
        self._step.assign_add(delta)

    @property
    def batches_trained(self):
        return self._batches_trained.numpy().item()

    def inc_batches_trained(self, delta):
        self._batches_trained.assign_add(delta)

    def __call__(self, obs, reset, state=None, training=True):
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        action, state = self._policy(obs, state, training)
        return action, state

    @tf.function
    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs['image'])
            latent = self._wm.dynamics.initial(len(obs['image']))
            action = tf.zeros((batch_size, self._config.num_actions), self._float)
        else:
            latent, action = state
        embed = self._wm.encoder(self._wm.preprocess(obs))
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, self._config.collect_dyn_sample)
        if self._config.eval_state_mean:
            latent['stoch'] = latent['mean']
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            action = self._task_behavior.actor(feat).mode()  # TODO: shouldn't this be .sample()? Optimal policy might be stochastic.
        elif self._should_expl(self._step):
            action = self._expl_behavior.actor(feat).sample()
        else:
            action = self._task_behavior.actor(feat).sample()
        if self._config.actor_dist == 'onehot_gumble':
            action = tf.cast(
                tf.one_hot(tf.argmax(action, axis=-1), self._config.num_actions),
                action.dtype)
        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        amount = tf.cast(amount, self._float)
        if 'onehot' in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def train(self):
        steps = self._config.train_steps
        if self._should_pretrain():
            steps += self._config.pretrain
        for _ in range(steps):
            data = next(self._dataset)
            self._train(data)
            self.inc_batches_trained(1)
        for name, mean in self._metrics.items():
            self._logger.scalar(name, float(mean.result()))
            mean.reset_states()

        data_sample = {k: v[:6] for k, v in data.items()}
        if self._config.train_openl_gifs:
            openl = self._wm.video_pred(data_sample)
            self._logger.video('train_openl', openl)
        if self._should_log_sample(self.batches_trained):
            data_pred = self.data_pred(data_sample)
            self._logger.data_dict('data_sample', data_pred)

        self._logger.scalar('batches_trained', self.batches_trained)

    @tf.function
    def _train(self, data):
        print('Tracing train function.')
        metrics = {}

        # Train world model
        embed, post, feat, kl, mets = self._wm.train(data)
        metrics.update(mets)

        if not self._config.wm_only:
            # Train behavior model
            start = post
            if 'discount' in self._config.grad_heads:  # Last step could be terminal.
                start = {k: v[:, :-1] for k, v in post.items()}
                embed, feat, kl = embed[:, :-1], feat[:, :-1], kl[:, :-1]
            reward = lambda f, s, a: self._wm.heads['reward'](f).mean()
            mets = self._task_behavior.train(start, reward)[-1]
            metrics.update(mets)

            # Train exploration
            if self._config.expl_behavior != 'greedy':
                mets = self._expl_behavior.train(start, feat, embed, kl)[-1]
                metrics.update({'expl_' + key: value for key, value in mets.items()})

        for name, value in metrics.items():
            self._metrics[name].update_state(value)

    @tf.function
    def data_pred(self, data):
        """Make predictions for logging and debug purposes"""
        wm = self._wm
        behav = self._task_behavior

        pred = data.copy()
        data = wm.preprocess(data)
        embed = wm.encoder(data)

        # Observe first 5 steps, imagine the rest
        post, _ = wm.dynamics.observe(embed[:, :5], data['action'][:, :5])
        init = {k: v[:, -1] for k, v in post.items()}
        prior = wm.dynamics.imagine(data['action'][:, 5:], init)

        # First 5 observations will be reconstructed from post, the rest predicted from prior
        feat = tf.concat([wm.dynamics.get_feat(post), wm.dynamics.get_feat(prior)], 1)
        for name, head in wm.heads.items():
            if name in ['reward', 'discount']:
                pred[name + '_pred'] = head(feat).mean()
            else:
                pred[name + '_pred'] = head(feat).mode()

        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        rewards = pred['reward_pred']
        target, weights, slow_value = behav._compute_target(swap(feat), swap(rewards), 0, 0, self._config.slow_value_target)
        target = tf.concat([target, slow_value[-1:]], axis=0)  # Target is only (T-1) long, append last value target for information
        pred['value_target'] = swap(target)
        pred['value_weights'] = swap(weights)
        pred['value'] = behav.value(feat).mean()

        return pred


def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
    example = episodes[next(iter(episodes.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    generator = lambda: tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends, config.seed)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, config.size,
            grayscale=config.atari_grayscale,
            life_done=False and (mode == 'train'),
            sticky_actions=True,
            all_actions=True)
        env = wrappers.OneHotAction(env)
    elif suite == 'minigrid':
        assert config.action_repeat == 1, "Do not use action_repeat for MiniGrid"
        # max_steps=time_limit+1, so that we use TimeLimit wrapper, not env max_steps
        env = wrappers.MiniGrid(task, max_steps=config.time_limit + 1)
        env = wrappers.OneHotAction(env)
        assert env.observation_space['image'].high.max() == config.decoder_discrete - 1, "Adjust decoder_discrete according to env"
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
    length = len(episode['reward']) - 1
    if length < config.batch_length:
        print(f'Skipped short episode of length {length}.')
        return
    directory = dict(train=config.traindir, eval=config.evaldir)[mode]
    cache = dict(train=train_eps, eval=eval_eps)[mode]
    filename = tools.save_episodes(directory, [episode])[0]
    score = float(episode['reward'].astype(np.float64).sum())
    video = episode['image']
    if mode == 'eval':
        cache.clear()
    if mode == 'train' and config.dataset_size:
        total = 0
        for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
            if total <= config.dataset_size - length:
                total += len(ep['reward']) - 1
            else:
                del cache[key]
        logger.scalar('dataset_size', total + length)
    cache[str(filename)] = episode
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar_mean(f'{mode}_return', score)
    logger.scalar_mean(f'{mode}_length', length)
    logger.scalar(f'{mode}_episodes', len(cache))
    if mode == 'eval' or config.expl_gifs:
        logger.video(f'{mode}_policy', video[None])


def main(logdir, config):
    config.start_time = datetime.datetime.now()
    logdir = pathlib.Path(logdir.format(**vars(config))).expanduser()
    config.traindir = pathlib.Path(config.traindir or logdir / 'train_eps')
    config.evaldir = pathlib.Path(config.evaldir or logdir / 'eval_eps')
    config.act = getattr(tf.nn, config.act)

    if config.debug:
        tf.config.experimental_run_functions_eagerly(True)
    if config.gpu_growth:
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    logger = loggers.Logger(logdir, config.log_mlflow, config.run_name, vars(config))

    if config.offline_traindir:
        directory = pathlib.Path(config.offline_traindir.format(**vars(config)))
        print(f'Load offline data from {directory} ...')
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    print(f'Found data {count_steps(directory)} steps in {directory}.')
    if config.offline_evaldir:
        directory = pathlib.Path(config.offline_evaldir.format(**vars(config)))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    print('Create envs.')
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    train_envs = [make('train') for _ in range(config.envs)]
    eval_envs = [make('eval') for _ in range(config.envs)]
    acts = train_envs[0].action_space
    if not config.num_actions:
        config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

    if not config.offline_traindir and config.prefill > 0:
        traindir_steps = count_steps(config.traindir)
        prefill = max(0, config.prefill - traindir_steps)
        print(f'Initial training data prefill ({config.prefill} steps, {traindir_steps} found)...')
        random_agent = lambda o, d, s: ([acts.sample() for _ in d], s)
        tools.simulate(random_agent, train_envs, prefill)
    if not config.offline_evaldir:
        print(f'Initial eval data prefill (1 episodes)...')
        tools.simulate(random_agent, eval_envs, episodes=1)

    train_dataset = make_dataset(train_eps, config)
    eval_dataset = iter(make_dataset(eval_eps, config))
    print('Create agent...')
    agent = Dreamer(config, logger, train_dataset)
    if (logdir / 'variables.pkl').exists():
        agent.load(logdir / 'variables.pkl')
        agent._should_pretrain._once = False

    print(f'Start training loop ({agent.step}/{int(config.steps)} steps done)...')
    should_eval = tools.Every(config.eval_every)
    should_save = tools.Every(config.save_every)
    state = None
    log_step = 0
    while log_step < config.steps:

        if should_eval(log_step):
            if not config.offline_evaldir:
                if config.eval_openl_gifs:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video('eval_openl', video_pred)
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(eval_policy, eval_envs, episodes=1)
                logger.write(log_step)
            else:
                # TODO: offline evaluation
                pass

        if not config.offline_traindir:
            state, steps = tools.simulate(agent, train_envs, config.train_every, state=state)
            agent.inc_step(steps)
            logger.scalar('steps', agent.step)

        agent.train()

        log_step = agent.batches_trained if config.offline_traindir else agent.step
        logger.write(log_step, fps=True)

        if should_save(log_step):
            agent.save(logdir / 'variables.pkl')

    print('Training finished.')
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--extra_yaml')
    args, remaining = parser.parse_known_args()

    config_ = {}
    configs = yaml.safe_load((pathlib.Path(__file__).parent / 'configs.yaml').read_text())
    if args.extra_yaml:
        configs.update(yaml.safe_load((pathlib.Path(__file__).parent / args.extra_yaml).read_text()))
    for name in args.configs:
        config_.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in config_.items():
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    main(args.logdir, parser.parse_args(remaining))
