import ray
import gymnasium

from src.worker.generic import GenericWorker
from src.agent.ddqn_agent import DuellingDDQNAgent
from src.dataclass.agent import AgentConfig

# Initialize Ray
ray.init()

# Define a Ray remote function or actor
@ray.remote
class TuningActor:
    def __init__(self, env, agent):
        self.worker = GenericWorker(env=env, agent=agent)
        self.agent = agent

    def tune(self, n_rollouts=100_000):
        for rollout in range(n_rollouts):
            self.worker.gather_experience()
            self.agent.learn()

            if rollout % 100 == 0:
                mean_scores = self.worker.get_mean_scores()
                print(f"Rollout {rollout}/{n_rollouts}, Mean Score: {mean_scores}, Epsilon: {self.agent.epsilon}")

        return self.worker.get_mean_scores()

def instantiate_env(prefix="CartPole-v1"):
    env = gymnasium.make("CartPole-v1", render_mode="human")
    return env

if __name__ == "__main__":
    # Parameters
    num_parallel_runs = 4  # Number of parallel runs

    # Create multiple instances of environment and agent configurations
    envs = [instantiate_env() for _ in range(num_parallel_runs)]

    actions = envs[0].action_space.n
    observation_shape = envs[0].observation_space.shape

    agent_config = AgentConfig(
            n_actions=actions,
            input_dims=tuple(observation_shape),
            gamma=0.99,
            epsilon=1.0,
            lr=5e-4,
            mem_size=100_000,
            batch_size=64,
            eps_min=0.01,
            eps_dec=2e-4,
            replace=1_000,
        )

    agents = [DuellingDDQNAgent(config=agent_config) for _ in range(num_parallel_runs)]

    # Create Ray actors
    tuning_actors = [TuningActor.remote(envs[i], agents[i]) for i in range(num_parallel_runs)]

    # Start tuning in parallel and get results
    results = ray.get([actor.tune.remote() for actor in tuning_actors])

    # Process results
    print("Results from parallel runs:", results)

    # Shutdown Ray
    ray.shutdown()
