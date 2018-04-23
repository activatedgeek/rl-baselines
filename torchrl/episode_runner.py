import time
from .learners import BaseLearner


class EpisodeRunner:
    """
    EpisodeRunner is a utility wrapper to run episodes on a single Gym-like environment
    object. Each call to the `run()` method will run one episode for specified
    number of steps. Call `reset()` to reuse the same object again
    """
    def __init__(self, env, max_steps=100000000):
        """
        :param env: Environment with Gym-like API
        :param max_steps: Maximum number of steps per episode (useful for non-episodic environments)
        """
        self.env = env

        # Parameters
        self.max_steps = max_steps

        # Internal State
        self._state = None
        self._done = False

    def reset(self):
        """
        Reset internal state for the `run` method to be reused
        """
        self._state = None
        self._done = False

    def is_done(self):
        return self._done

    def run(self, learner, steps=None, render=False, fps=30, episode_id=None):
        """

        :param learner: An agent of type BaseLearner
        :param steps: Number of maximum steps in the current rollout
        :param render: Flag to render the environment, True or False
        :param fps: Rendering rate of the environment, frames per second if render is True
        :param episode_id: Unique identifier to identify the current transition's episode
        :return: batch of transitions
        """
        assert not self._done, 'EpisodeRunner has ended. Call .reset() to reuse.'

        assert isinstance(learner, BaseLearner),\
            '"learner" should inherit from "BaseLearner", found invalid type "{}"'.format(type(learner))

        steps = steps or self.max_steps

        if self._state is None:
            self._state = self.env.reset()
            if render:
                self.env.render()
                time.sleep(1. / fps)

        state_history = []
        action_history = []
        reward_history = []
        next_state_history = []
        done_history = []

        while not self._done and steps:
            action = learner.act(self._state)
            next_state, reward, self._done, info = self.env.step(action)

            state_history.append(self._state)
            action_history.append(action)
            reward_history.append(reward)
            next_state_history.append(next_state)
            done_history.append(int(self._done))

            self._state = next_state
            steps -= 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

        return state_history, action_history, reward_history, next_state_history, done_history
