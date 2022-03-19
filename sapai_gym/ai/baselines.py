from sapai import *
from random import choice


def random_agent(player_to_act: Player, actions):
    """
    Returns a random action
    :param player_to_act: Not used in this function
    :param actions: Available actions
    :return: Action to play
    """
    return choice(actions)


def random_agent_max_spend(player_to_act: Player, actions):
    """
    A random agent that spends all of its money before ending the turn.
    :param player_to_act: Not used in this function
    :param actions: Available actions
    :return: Action to play
    """
    non_selling_actions = _filter_remove_by_action_name(actions, ["sell", "roll"])
    if len(non_selling_actions) == 1:
        return non_selling_actions[0]
    return choice(actions[1:])


def _get_action_str(action):
    method, *raw_args = action
    method_name = method.__name__
    args_str = ','.join(str(e) for e in raw_args)
    return method_name + "-" + args_str


def _map_buy_pet_action_to_shop_pet(player_to_act: Player, action):
    shop_index = action[1]
    assert isinstance(shop_index, int)
    return player_to_act.shop[shop_index].item


def _map_sell_pet_action_to_team_pet(player_to_act: Player, action):
    team_index = action[1]
    assert isinstance(team_index, int)
    return player_to_act.team[team_index].pet


def _feed_front_pet_actions(player_to_act: Player, actions):
    front_pet_index = 0
    for i in range(5):
        front_pet_index = i
        if not player_to_act.team[i].empty:
            break

    # TODO : MULTI FOODS DON'T WORK CURRENTLY. MAKE A CHANGE FOR THAT WHEN IT IS FIXED
    return [a for a in actions if a[2] == front_pet_index]


def _find_weakest_pet_on_team(player_to_act: Player, actions):
    sorted_list = sorted(actions, key=lambda a: _map_sell_pet_action_to_team_pet(player_to_act, a).attack + _map_sell_pet_action_to_team_pet(player_to_act, a).health)
    return sorted_list[0]


def _find_strongest_shop_pet(player_to_act: Player, actions):
    sorted_list = sorted(actions, key=lambda a: _map_buy_pet_action_to_shop_pet(player_to_act, a).attack + _map_buy_pet_action_to_shop_pet(player_to_act, a).health, reverse=True)
    return sorted_list[0]


def _filter_by_action_name(action_list: list[any], match_criteria: list[str]):
    return [a for a in action_list if a[0].__name__ in match_criteria]


def _filter_remove_by_action_name(action_list: list[any], match_criteria: list[str]):
    return [a for a in action_list if a[0].__name__ not in match_criteria]


def get_buy_food_action_front(player_to_act: Player, actions):
    # Buy food, target the front pet if it's a targeting food
    buy_food_actions = _filter_by_action_name(actions, ["buy_food"])
    if len(buy_food_actions) >= 1:
        # Remove sleeping pill from choices
        buy_food_actions_no_pill = [x for x in buy_food_actions if
                                    player_to_act.shop[x[1]].item.name != "food-sleeping-pill"]
        if len(buy_food_actions_no_pill) >= 1:
            return choice(_feed_front_pet_actions(player_to_act, buy_food_actions_no_pill))
    return None


def get_buy_food_action_everyone(player_to_act: Player, actions):
    # Buy food, target the front pet if it's a targeting food
    buy_food_actions = _filter_by_action_name(actions, ["buy_food"])
    if len(buy_food_actions) >= 1:
        # Remove sleeping pill from choices
        buy_food_actions_no_pill = [x for x in buy_food_actions if
                                    player_to_act.shop[x[1]].item.name != "food-sleeping-pill"]
        if len(buy_food_actions_no_pill) >= 1:
            return choice(buy_food_actions_no_pill)
    return None


def _biggest_numbers(player_to_act: Player, actions, buy_food_method):
    if len(actions) == 1:
        return actions[0]

    buy_pet_actions = _filter_by_action_name(actions, ["buy_pet"])
    can_buy_pet = len(buy_pet_actions) >= 1
    if can_buy_pet:
        buy_strongest_shop_pet_action = _find_strongest_shop_pet(player_to_act, buy_pet_actions)

    # If team isn't full, buy the pet with the biggest numbers
    if len(player_to_act.team) < 5 and can_buy_pet:
        return buy_strongest_shop_pet_action

    # Upgrade existing pets if possible
    upgrade_actions = _filter_by_action_name(actions, ["buy_combine", "combine"])
    if len(upgrade_actions) >= 1:
        return choice(upgrade_actions)

    # If there is a pet in the shop with a bigger number than a pet on the team, replace the weakest pet on the team
    # with the strongest pet from the shop by selling the weakest pet
    sell_actions = _filter_by_action_name(actions, ["sell"])
    if len(sell_actions) >= 1 and can_buy_pet:
        strongest_shop_pet = _map_buy_pet_action_to_shop_pet(player_to_act, buy_strongest_shop_pet_action)
        strongest_shop_pet_score = strongest_shop_pet.attack + strongest_shop_pet.health
        sell_weakest_team_pet_action = _find_weakest_pet_on_team(player_to_act, sell_actions)
        weakest_team_pet = _map_sell_pet_action_to_team_pet(player_to_act, sell_weakest_team_pet_action)
        weakest_team_pet_score = weakest_team_pet.attack + weakest_team_pet.health
        if strongest_shop_pet_score > weakest_team_pet_score:
            return sell_weakest_team_pet_action

    # Buy food, target the front pet if it's a targeting food
    buy_food_actions = _filter_by_action_name(actions, ["buy_food"])
    if len(buy_food_actions) >= 1:
        buy_food_action = buy_food_method(player_to_act, actions)
        if buy_food_action:
            return buy_food_action

    # Re-roll
    re_roll_action = _filter_by_action_name(actions, ["roll"])
    assert len(re_roll_action) <= 1
    if len(re_roll_action) == 1:
        return re_roll_action[0]

    # End turn
    return actions[0]


def biggest_numbers_vert(player_to_act: Player, actions):
    """
    Always increase the total (health+attack) of the team. When buying food, feeds the first pet.
    :param player_to_act: Player to choose action for
    :param actions: Available actions
    :return: Action to play
    """
    return _biggest_numbers(player_to_act, actions, get_buy_food_action_front)


def biggest_numbers_horizontal(player_to_act: Player, actions):
    """
    Always increase the total (health+attack) of the team. When buying food, feeds pets randomly.
    :param player_to_act: Player to choose action for
    :param actions: Available actions
    :return: Action to play
    """
    return _biggest_numbers(player_to_act, actions, get_buy_food_action_everyone)
