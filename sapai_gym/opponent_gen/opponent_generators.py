from sapai import Player, Team

from sapai_gym.ai import baselines
from sapai_gym import SuperAutoPetsEnv

# TODO : Wrap the ai to create a generator


def _do_store_phase(env: SuperAutoPetsEnv, ai):
    env.player.start_turn()

    while True:
        actions = env._avail_actions()
        chosen_action = ai(env.player, actions)
        env.resolve_action(chosen_action)

        if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
            return


def opp_generator(num_turns, ai):
    opps = list()
    env = SuperAutoPetsEnv(None, valid_actions_only=True, manual_battles=True)
    while env.player.turn <= num_turns:
        _do_store_phase(env, ai)
        opps.append(Team.from_state(env.player.team.state))
    return opps


def random_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.random_agent)


def biggest_numbers_horizontal_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.biggest_numbers_horizontal_scaling_agent)