from enum import Enum
import gymnasium as gym
import numpy as np


class GameAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.001,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95,
    ):
        self.env = env

        if isinstance(env.action_space, gym.spaces.Box):
            nb_actions = env.action_space.shape[0]
        else:
            nb_actions = env.action_space.n

        print("action space =", nb_actions)
        print("state space =", env.observation_space)

        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.q_values = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def attach_env(self, env: gym.Env):
        self.env = env

    def set_hyperparameters(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
    ):
        self.alpha = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, done):
        old_q = self.get_q_value(state, action)

        if done:
            future_reward = 0.0
        else:
            future_reward = self.best_future_reward(next_state)

        new_q = old_q + self.alpha * (reward + self.gamma * future_reward - old_q)
        self.q_values[(state, action)] = new_q

    def best_future_reward(self, state):
        n_actions = self.env.action_space.n
        q_list = [self.get_q_value(state, action) for action in range(n_actions)]
        return max(q_list) if q_list else 0.0

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        n_actions = self.env.action_space.n
        q_list = [self.get_q_value(state, action) for action in range(n_actions)]
        max_q = max(q_list)

        best_actions = [action for action, q in enumerate(q_list) if q == max_q]
        return int(np.random.choice(best_actions))

    def get_determinist_action(self, state):
        n_actions = self.env.action_space.n
        q_list = [self.get_q_value(state, action) for action in range(n_actions)]
        max_q = max(q_list)

        best_actions = [action for action, q in enumerate(q_list) if q == max_q]
        return int(np.random.choice(best_actions))


class Game(Enum):
    frozen = "frozen_lake"
    frozen8 = "frozen_lake8"
    taxi = "taxi"
    cart = "cart_pole"

    def __str__(self):
        return self.value

    @staticmethod
    def getEnv(s, show):
        render_mode = "human" if show else None

        if s == Game.cart:
            return gym.make("CartPole-v1", render_mode=render_mode)
        elif s == Game.frozen8:
            return gym.make(
                "FrozenLake-v1",
                map_name="8x8",
                is_slippery=False,
                render_mode=render_mode
            )
        elif s == Game.taxi:
            return gym.make("Taxi-v3", render_mode=render_mode)
        else:
            return gym.make(
                "FrozenLake-v1",
                is_slippery=False,
                render_mode=render_mode
            )

    @staticmethod
    def discretizeState(state):
        cartPositionMin = -4.8
        cartPositionMax = 4.8
        cartVelocityMin = -3
        cartVelocityMax = 3
        poleAngleMin = -0.418
        poleAngleMax = 0.418
        poleAngleVelocityMin = -10
        poleAngleVelocityMax = 10
        numberOfBins = 25

        cartPositionBin = np.linspace(cartPositionMin, cartPositionMax, numberOfBins)
        cartVelocityBin = np.linspace(cartVelocityMin, cartVelocityMax, numberOfBins)
        poleAngleBin = np.linspace(poleAngleMin, poleAngleMax, numberOfBins)
        poleAngleVelocityBin = np.linspace(
            poleAngleVelocityMin,
            poleAngleVelocityMax,
            numberOfBins
        )

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(
            np.digitize(state[3], poleAngleVelocityBin) - 1,
            0
        )

        return (
            int(indexPosition),
            int(indexVelocity),
            int(indexAngle),
            int(indexAngularVelocity),
        )