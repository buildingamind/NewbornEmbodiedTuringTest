import os
import time
import json
import logging
import numpy as np
import gym
from gym.utils import atomic_write, closer
logger = logging.getLogger(__name__)

__all__ = ['TraceRecordingWrapper']

trace_record_closer = closer.Closer()

class TraceRecordingWrapper(gym.Wrapper):
    """

    A Wrapper that records a trace of every action, observation, and reward generated by an environment.
    For an episode of length N, this will consist of:
      actions [0..N]
      observations [0..N+1]. Including the initial observation from `env.reset()`
      rewards [0..N]

    Usage:

      from gym_recording.wrappers import TraceRecordingWrapper
      if args.record_trace:
        env = TraceRecordingWrapper(env, '/tmp/mytraces')

    It'll save a numbered series of json-encoded files, with large arrays stored in binary, along
    with a manifest in /tmp/mytraces/openaigym.traces.*.
    See gym_recording.recording for more on the file format

    Later you can load the recorded traces:

      import gym_recording.playback

      def episode_cb(observations, actions, rewards):
          ... do something the episode ...

      gym_recording.playback.scan_recorded_traces('/tmp/mytraces', episode_cb)

    For an episode of length N, episode_cb receives 3 numpy arrays:
      observations.shape = [N + 1, observation_dim]
      actions.shape = [N, action_dim]
      rewards.shape = [N]


    """
    def __init__(self, env, directory=None):
        """
        Create a TraceRecordingWrapper around env, writing into directory
        """
        super().__init__(env)
        self.recording = None
        trace_record_closer.register(self)

        self.recording = TraceRecording(directory)
        self.directory = self.recording.directory

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.recording.add_step(action, observation, reward, done)
        return observation, reward, done, info

    def reset(self):
        self.recording.end_episode()
        observation = self.env.reset()
        self.recording.add_reset(observation)
        return observation

    def close(self):
        """
        Flush any buffered data to disk and close. It should get called automatically at program exit time, but
        you can free up memory by calling it explicitly when you're done
        """
        if self.recording is not None:
            self.recording.close()

class TraceRecording(object):
    _id_counter = 0
    def __init__(self, directory=None):
        """
        Create a TraceRecording, writing into directory
        """

        if directory is None:
            directory = os.path.join('/tmp', 'openai.gym.{}.{}'.format(time.time(), os.getpid()))
            os.mkdir(directory)

        self.directory = directory
        self.file_prefix = 'openaigym.trace.{}.{}'.format(self._id_counter, os.getpid())
        TraceRecording._id_counter += 1

        self.closed = False

        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        self.episode_id = 0

        self.buffered_step_count = 0
        self.buffer_batch_size = 100

        self.episodes_first = 0
        self.episodes = []
        self.batches = []

    def add_reset(self, observation):
        assert not self.closed
        self.end_episode()
        self.observations.append(observation)

    def add_step(self, action, observation, reward, done):
        assert not self.closed
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.buffered_step_count += 1

    def end_episode(self):
        """
        if len(observations) == 0, nothing has happened yet.
        If len(observations) == 1, then len(actions) == 0, and we have only called reset and done a null episode.
        """
        if len(self.observations) > 0:
            if len(self.episodes)==0:
                self.episodes_first = self.episode_id

            self.episodes.append({
                'actions': self.optimize_list_of_ndarrays(self.actions),
                'observations': self.optimize_list_of_ndarrays(self.observations),
                'rewards': self.optimize_list_of_ndarrays(self.rewards),
                'dones': self.optimize_list_of_ndarrays(self.dones),
            })
            self.actions = []
            self.observations = []
            self.rewards = []
            self.dones = []
            self.episode_id += 1

            if self.buffered_step_count >= self.buffer_batch_size:
                self.save_complete()

    def save_complete(self):
        """
        Save the latest batch and write a manifest listing all the batches.
        We save the arrays as raw binary, in a format compatible with np.load.
        We could possibly use numpy's compressed format, but the large observations we care about (VNC screens)
        don't compress much, only by 30%, and it's a goal to be able to read the files from C++ or a browser someday.
        """

        batch_fn = '{}.ep{:09}.json'.format(self.file_prefix, self.episodes_first)
        bin_fn = '{}.ep{:09}.bin'.format(self.file_prefix, self.episodes_first)

        with atomic_write.atomic_write(os.path.join(self.directory, batch_fn), False) as batch_f:
            with atomic_write.atomic_write(os.path.join(self.directory, bin_fn), True) as bin_f:

                def json_encode(obj):
                    if isinstance(obj, np.ndarray):
                        offset = bin_f.tell()
                        while offset%8 != 0:
                            bin_f.write(b'\x00')
                            offset += 1
                        obj.tofile(bin_f)
                        size = bin_f.tell() - offset
                        return {'__type': 'ndarray', 'shape': obj.shape, 'order': 'C', 'dtype': str(obj.dtype), 'npyfile': bin_fn, 'npyoff': offset, 'size': size}
                    return obj

                json.dump({'episodes': self.episodes}, batch_f, default=json_encode)

                bytes_per_step = float(bin_f.tell() + batch_f.tell()) / float(self.buffered_step_count)

        self.batches.append({
            'first': self.episodes_first,
            'len': len(self.episodes),
            'fn': batch_fn})

        manifest = {'batches': self.batches}
        manifest_fn = os.path.join(self.directory, '{}.manifest.json'.format(self.file_prefix))
        with atomic_write.atomic_write(os.path.join(self.directory, manifest_fn), False) as f:
            json.dump(manifest, f)

        # Adjust batch size, aiming for 5 MB per file.
        # This seems like a reasonable tradeoff between:
        #   writing speed (not too much overhead creating small files)
        #   local memory usage (buffering an entire batch before writing)
        #   random read access (loading the whole file isn't too much work when just grabbing one episode)
        self.buffer_batch_size = max(1, min(50000, int(5000000 / bytes_per_step + 1)))

        self.episodes = []
        self.episodes_first = None
        self.buffered_step_count = 0

    def close(self):
        """
        Flush any buffered data to disk and close. It should get called automatically at program exit time, but
        you can free up memory by calling it explicitly when you're done
        """
        if not self.closed:
            self.end_episode()
            if len(self.episodes) > 0:
                self.save_complete()
            self.closed = True
            logger.info('Wrote traces to %s', self.directory)

    @staticmethod
    def optimize_list_of_ndarrays(x):
        """
        Replace a list of ndarrays with a single ndarray with an extra dimension.
        Should return unchanged a list of other kinds of observations or actions, like Discrete or Tuple
        """
        if type(x) == np.ndarray:
            return x
        if len(x) == 0:
            return np.array([[]])
        if type(x[0]) == float or type(x[0]) == int:
            return np.array(x)
        if type(x[0]) == np.ndarray and len(x[0].shape) == 1:
            return np.array(x)
        return x