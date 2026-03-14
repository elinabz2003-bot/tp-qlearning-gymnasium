from agent import Game, GameAgent
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os


def learn(args):
    learning_rate = 0.1
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / args.n_episodes
    final_epsilon = 0.01

    env = Game.getEnv(args.game, show=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=args.n_episodes)

    if os.path.isfile("agent-" + str(args.game) + ".dump"):
        with open("agent-" + str(args.game) + ".dump", "rb") as file:
            agent = pickle.load(file)

        agent.attach_env(env)
        agent.set_hyperparameters(
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )
        print("Q-values already learned:", len(agent.q_values))

    else:
        agent = GameAgent(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )

    for _ in tqdm(range(args.n_episodes)):
        state, info = env.reset()

        if args.game == Game.cart:
            state = Game.discretizeState(state)

        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if args.game == Game.cart:
                next_state = Game.discretizeState(next_state)

            done = terminated or truncated

            agent.update_q_value(state, action, reward, next_state, done)

            state = next_state

        agent.decay_epsilon()

    with open("agent-" + str(args.game) + ".dump", "wb") as file:
        pickle.dump(agent, file)

    return env


def moving_average(values, window):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return np.array([])
    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window), mode="valid") / window


def showStats(args, env):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    episode_window = min(100, max(1, args.n_episodes // 20))

    rewards = list(env.return_queue)
    lengths = list(env.length_queue)

    reward_moving_average = moving_average(rewards, episode_window)
    length_moving_average = moving_average(lengths, episode_window)

    axs[0].set_title("Episode rewards")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    plt.tight_layout()
    plt.savefig("agent-" + str(args.game) + ".png")
    plt.show()
    plt.close()


parser = argparse.ArgumentParser(description="Learn an agent to play using Q-learning.")

parser.add_argument("game", type=Game, choices=list(Game))
parser.add_argument(
    "-n",
    "--n_episodes",
    nargs="?",
    dest="n_episodes",
    type=int,
    default=1000,
    help="number of episodes for Q-training",
)

args = parser.parse_args()

env = learn(args)
showStats(args, env)