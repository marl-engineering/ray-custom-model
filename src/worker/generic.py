import numpy as np


class GenericWorker:
    """ Gather experiences through interaction between agent and env

        Args:
            arg: [env]: instance of gymnasium / ray environment
            arg: [agent]: action selector

    """

    def __init__(self, env, agent) -> None:
        self._env = env
        self._agent = agent

        self._env_step = 0
        self._rollout_step = 0

        self._scores = []

    def reset(self) -> np.ndarray:
        obs, info = self._env.reset()
        self._rollout_step = 0
        return obs, info

    def get_mean_scores(self, n_last_scores=100) -> np.ndarray:
        return np.mean(self._scores[-n_last_scores:])

    def gather_experience(self):
        state, _ = self.reset()
        done = False
        episode_return = 0

        while not done:
            action = self._agent.choose_action(state)
            new_state, reward, terminated, truncated, info = self._env.step(action)

            episode_return += reward

            # Check if the episode is done either due to truncation or termination
            done = np.any([truncated, terminated])

            self._agent.store_transition(state, action, reward, new_state, done)

            state = new_state
            self._env_step += 1
            self._rollout_step += 1

        self._scores.append(episode_return)
