from agent import Game
import argparse
import sys
import os
import pickle


def launch(args):
    env = Game.getEnv(args.game, show=True)

    state, info = env.reset()

    if args.game == Game.cart:
        state = Game.discretizeState(state)

    agent = None
    if args.ai:
        filename = "agent-" + str(args.game) + ".dump"

        if not os.path.isfile(filename):
            sys.stderr.write("no agent available for " + str(args.game) + "\n")
            sys.exit(1)

        with open(filename, "rb") as file:
            agent = pickle.load(file)

        agent.attach_env(env)

    episode_over = False
    while not episode_over:
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.get_determinist_action(state)

        state, reward, terminated, truncated, info = env.step(action)

        if args.game == Game.cart:
            state = Game.discretizeState(state)

        episode_over = terminated or truncated

    env.close()


parser = argparse.ArgumentParser(description="Launch a game.")

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-r",
    "--random",
    dest="rand",
    action="store_true",
    help="choose randomly actions during the game [default]",
)
group.add_argument(
    "-i",
    "--ai",
    dest="ai",
    action="store_true",
    help="choose actions using an AI trained with Q-learning",
)

parser.add_argument("game", type=Game, choices=list(Game))
args = parser.parse_args()

launch(args)