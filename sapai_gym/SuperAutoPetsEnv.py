import gym
from gym import spaces
import numpy as np
from typing import Optional
import itertools
from sklearn.preprocessing import OneHotEncoder

from sapai import Player, Pet, Food, Battle


class SuperAutoPetsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_ACTIONS = 63
    ACTION_BASE_NUM = {
        "end_turn": 0,
        "buy_pet": 1,
        "buy_food": 7,
        "buy_combine": 17,
        "combine": 47,
        "sell": 57,
        "roll": 62
    }
    # Max turn limit to prevent infinite loops
    MAX_TURN = 25
    BAD_ACTION_PENALTY = -0.1

    # Max number of pets that can be on a team
    MAX_TEAM_PETS = 5
    # Max number of pets that can be in a shop
    MAX_SHOP_PETS = 6
    # Max number of foods that can be in a shop
    MAX_SHOP_FOODS = 2
    ALL_PETS = ["pet-ant", "pet-beaver", "pet-beetle", "pet-bluebird", "pet-cricket", "pet-duck", "pet-fish", "pet-horse", "pet-ladybug", "pet-mosquito", "pet-otter", "pet-pig", "pet-sloth", "pet-bat", "pet-crab", "pet-dodo", "pet-dog", "pet-dromedary", "pet-elephant", "pet-flamingo", "pet-hedgehog", "pet-peacock", "pet-rat", "pet-shrimp", "pet-spider", "pet-swan", "pet-tabby-cat", "pet-badger", "pet-blowfish", "pet-caterpillar", "pet-camel", "pet-hatching-chick", "pet-giraffe", "pet-kangaroo", "pet-owl", "pet-ox", "pet-puppy", "pet-rabbit", "pet-sheep", "pet-snail", "pet-tropical-fish", "pet-turtle", "pet-whale", "pet-bison", "pet-buffalo", "pet-deer", "pet-dolphin", "pet-hippo", "pet-llama", "pet-lobster", "pet-monkey", "pet-penguin", "pet-poodle", "pet-rooster", "pet-skunk", "pet-squirrel", "pet-worm", "pet-chicken", "pet-cow", "pet-crocodile", "pet-eagle", "pet-goat", "pet-microbe", "pet-parrot", "pet-rhino", "pet-scorpion", "pet-seal", "pet-shark", "pet-turkey", "pet-cat", "pet-boar", "pet-dragon", "pet-fly", "pet-gorilla", "pet-leopard", "pet-mammoth", "pet-octopus", "pet-sauropod", "pet-snake", "pet-tiger", "pet-tyrannosaurus", "pet-zombie-cricket", "pet-bus", "pet-zombie-fly", "pet-dirty-rat", "pet-chick", "pet-ram", "pet-butterfly", "pet-bee"]
    ALL_FOODS = ["food-apple", "food-honey", "food-cupcake", "food-meat-bone", "food-sleeping-pill", "food-garlic", "food-salad-bowl", "food-canned-food", "food-pear", "food-chili", "food-chocolate", "food-sushi", "food-melon", "food-mushroom", "food-pizza", "food-steak", "food-milk"]
    ALL_STATUSES = ["status-weak", "status-coconut-shield", "status-honey-bee", "status-bone-attack", "status-garlic-armor", "status-splash-attack", "status-melon-armor", "status-extra-life", "status-steak-attack", "status-poison-attack"]

    def __init__(self, opponent_generator, valid_actions_only, manual_battles=False):
        """
        Create a gym for Super Auto Pets.
        :param opponent_generator: Function that generates the opponents to play against when a shop turn is ended. This
        function should take one param (an int) that is the number of turns to generate opponents for. It should return
        a list of opponents, starting from turn 1
        :param valid_actions_only: bool. If set to true, will raise an exception when an invalid action is pass to step.
        This is helpful when action masks are used
        :param manual_battles: bool. If set to true, battles will not be manually executed. The caller is responsible
        for starting the next turn after a turn is ended. This is helpful when battles are irrelevant to task at hand,
        or when battles are manually controlled (eg. in an arena with multiple agents)
        """
        super(SuperAutoPetsEnv, self).__init__()

        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        len_obs_space = (len(self.ALL_PETS) + 2 + len(self.ALL_STATUSES)) * 11 + (len(self.ALL_FOODS) + 1) * 2 + 5
        print(f"Observation space of {len_obs_space}")
        self.observation_space = spaces.Box(low=0, high=1, shape=(len_obs_space,), dtype=np.uint8)
        self.reward_range = (0, 10)

        self.player = Player()
        self.just_froze = False
        self.just_reordered = False

        self.opponent_generator = opponent_generator
        self.valid_actions_only = valid_actions_only
        self.manual_battles = manual_battles

        # Initialization. Initial values assigned in reset
        self.opponents = None
        self.bad_action_reward_sum = 0

        self.reset()

    def step(self, action):
        if not isinstance(action, int):
            # Convert np int to python int
            action = action.item()
        if not self._is_valid_action(action):
            if self.valid_actions_only:
                raise RuntimeError(f"Environment tried to play invalid action {action}. Valid actions are {self._avail_actions().keys()}")
            # Teach agent to play valid actions
            self.bad_action_reward_sum += self.BAD_ACTION_PENALTY
        else:
            # Resolve action
            player_to_act = self.player
            action_to_play = self._avail_actions()[action]
            action_name = self._get_action_name(action_to_play).split(".")[-1]
            action_method = getattr(player_to_act, action_name)
            action_method(*action_to_play[1:])

            # If turn is ended, play an opponent
            if action_name == "end_turn" and not self.manual_battles:
                opponent = self.opponents[self.player.turn - 1]
                battle_result = Battle(self.player.team, opponent).battle()
                self._player_fight_outcome(battle_result)
                self.player.start_turn()

        obs = self._encode_state()
        reward = self.get_reward()
        done = self.is_done()
        info = dict()
        # info["player_info"] = self.player

        return obs, reward, done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        self.player = Player()
        self.just_froze = False
        self.just_reordered = False

        if not self.manual_battles:
            self.opponents = self.opponent_generator(25)

        self.bad_action_reward_sum = 0

        return self._encode_state()

    def render(self, mode='human', close=False):
        print(self.player)
        print(f"just_froze: {self.just_froze}")
        print(f"just_reordered: {self.just_reordered}")

    def is_done(self):
        # Cap games at 25 turns to prevent infinite loops
        return self.player.wins >= 10 or self.player.lives <= 0 or self.player.turn >= 25

    def get_reward(self):
        assert 0 <= self.player.wins <= 10
        if self.valid_actions_only:
            assert self.bad_action_reward_sum == 0
        return self.player.wins / 10 + self.bad_action_reward_sum

    def _avail_end_turn(self):
        action_dict = dict()
        action_num = self.ACTION_BASE_NUM["end_turn"]
        action_dict[action_num] = (self.player.end_turn,)
        return action_dict

    def _avail_buy_pets(self):
        action_dict = dict()
        if len(self.player.team) == 5:
            # Cannot buy for full team
            return action_dict
        pet_index = 0
        for shop_idx, shop_slot in enumerate(self.player.shop):
            if shop_slot.slot_type == "pet":
                if shop_slot.cost <= self.player.gold:
                    action_num = self.ACTION_BASE_NUM["buy_pet"] + pet_index
                    action_dict[action_num] = (self.player.buy_pet, shop_idx)
                pet_index += 1
        return action_dict

    def _avail_buy_foods(self):
        action_dict = dict()
        if len(self.player.team) == 0:
            return action_dict
        food_index = 0
        for shop_idx, shop_slot in enumerate(self.player.shop):
            if shop_slot.slot_type == "food":
                if shop_slot.cost <= self.player.gold:
                    for team_idx, team_slot in enumerate(self.player.team):
                        if team_slot.empty:
                            continue
                        action_num = self.ACTION_BASE_NUM["buy_food"] + (food_index * self.MAX_TEAM_PETS) + team_idx
                        action_dict[action_num] = (self.player.buy_food, shop_idx, team_idx)
                food_index += 1
        return action_dict

    def _avail_buy_combine(self):
        action_dict = dict()
        team_names = dict()
        if len(self.player.team) == 0:
            return action_dict

        # Find pet names on team
        for team_idx, slot in enumerate(self.player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(team_idx)

        # Search through pets in the shop
        shop_pet_index = 0
        for shop_idx, shop_slot in enumerate(self.player.shop):
            if shop_slot.slot_type == "pet":
                # Can't combine if pet not already on team
                if shop_slot.item.name not in team_names:
                    continue

                if shop_slot.cost <= self.player.gold:
                    for team_idx in team_names[shop_slot.item.name]:
                        action_num = self.ACTION_BASE_NUM["buy_combine"] + (shop_pet_index * self.MAX_TEAM_PETS) + team_idx
                        action_dict[action_num] = (self.player.buy_combine, shop_idx, team_idx)
                shop_pet_index += 1

        return action_dict

    def _avail_team_combine(self):
        action_dict = dict()

        if len(self.player.team) <= 1:
            return action_dict

        team_names = {}
        for slot_idx, slot in enumerate(self.player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(slot_idx)

        for key, value in team_names.items():
            if len(value) == 1:
                continue

            for idx0, idx1 in itertools.combinations(value, r=2):
                indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
                action_num = self.ACTION_BASE_NUM["combine"] + indexes.index((idx0, idx1))
                action_dict[action_num] = (self.player.combine, idx0, idx1)

        return action_dict

    def _avail_sell(self):
        action_dict = dict()
        for team_idx, slot in enumerate(self.player.team):
            if slot.empty:
                continue
            action_num = self.ACTION_BASE_NUM["sell"] + team_idx
            action_dict[action_num] = (self.player.sell, team_idx)
        return action_dict

    def _avail_roll(self):
        action_dict = dict()
        if self.player.gold > 1:
            action_dict[self.ACTION_BASE_NUM["roll"]] = (self.player.roll,)
        return action_dict

    @staticmethod
    def _get_action_name(input_action):
        return str(input_action[0].__name__)

    # Maps an integer representation of the action to the action
    def _avail_actions(self):
        end_turn_actions = self._avail_end_turn()
        buy_pet_actions = self._avail_buy_pets()
        buy_food_actions = self._avail_buy_foods()
        buy_combine_actions = self._avail_buy_combine()
        team_combine_actions = self._avail_team_combine()
        sell_actions = self._avail_sell()
        roll_actions = self._avail_roll()
        # TODO : REORDERING
        # TODO : FREEZE SHOP ITEMS

        # Verify no duplicates or incorrectly indexed actions
        total_action_len = len(end_turn_actions) + len(buy_pet_actions) + len(buy_food_actions) + len(buy_combine_actions) + len(team_combine_actions) + len(sell_actions) + len(roll_actions)
        all_avail_actions = end_turn_actions | buy_pet_actions | buy_food_actions | buy_combine_actions | team_combine_actions | sell_actions | roll_actions
        assert total_action_len == len(all_avail_actions)

        return all_avail_actions

    def _is_valid_action(self, action: int) -> bool:
        return action in self._avail_actions().keys()

    def action_masks(self):
        masks = [False] * self.MAX_ACTIONS
        for a in self._avail_actions().keys():
            masks[a] = True
        return masks

    def _player_fight_outcome(self, outcome: int):
        if outcome == 0:
            self.player.lf_winner = True
            self.player.wins += 1
        elif outcome == 2:
            self.player.lf_winner = False
        elif outcome == 1:
            self.player.lf_winner = False
            if self.player.turn <= 2:
                self.player.lives -= 1
            elif self.player.turn <= 4:
                self.player.lives -= 2
            else:
                self.player.lives -= 3
            self.player.lives = max(self.player.lives, 0)

    def _encode_pets(self, pets):
        arrays_to_concat = list()
        for pet in pets:
            if pet.name == "pet-none":
                arrays_to_concat.append(np.zeros((len(self.ALL_PETS),)))
                arrays_to_concat.append(np.zeros((2,)))
                arrays_to_concat.append(np.zeros((len(self.ALL_STATUSES),)))
            else:
                arrays_to_concat.append(self._encode_single(pet.name, self.ALL_PETS))
                arrays_to_concat.append(np.array([pet.attack / 50, pet.health / 50]))
                if pet.status == "none":
                    arrays_to_concat.append(np.zeros((len(self.ALL_STATUSES),)))
                else:
                    arrays_to_concat.append(self._encode_single(pet.status, self.ALL_STATUSES))
        return arrays_to_concat

    def _encode_foods(self, foods):
        arrays_to_concat = list()
        for food_tuple in foods:
            (food, cost) = food_tuple
            if food.name == "food-none":
                arrays_to_concat.append(np.zeros((len(self.ALL_FOODS),)))
                arrays_to_concat.append(np.zeros((1,)))
            else:
                arrays_to_concat.append(self._encode_single(food.name, self.ALL_FOODS))
                arrays_to_concat.append(np.array([cost / 3]))
        return arrays_to_concat

    def _get_shop_foods(self):
        food_slots = []
        for slot in self.player.shop.shop_slots:
            if slot.slot_type == "food":
                food_slots.append((slot.item, slot.cost))
        return food_slots

    def _encode_state(self):
        # Encode team
        encoded_team_pets = self._encode_pets([p.pet for p in self.player.team])

        # Encode shop
        shop_pets = self.player.shop.pets
        shop_foods = self._get_shop_foods()

        # Pad to the maximum number of pets and foods that can be in a shop
        while len(shop_pets) < 6:
            shop_pets.append(Pet("pet-none"))
        while len(shop_foods) < 2:
            shop_foods.append((Food("food-none"), 0))

        encoded_shop_pets = self._encode_pets(shop_pets)
        encoded_shop_foods = self._encode_foods(shop_foods)

        # Other player stats
        # Assumptions: Treat max gold as 20. Treat max turn as 25. Treat max cans as 10.
        other_stats = np.array([self.player.wins / 10, self.player.lives / 10, min(self.player.gold, 20) / 20, min(self.player.turn, 25) / 25, min(self.player.shop.can, 10) / 10])

        all_lists = list()
        all_lists.extend(encoded_team_pets)
        all_lists.extend(encoded_shop_pets)
        all_lists.extend(encoded_shop_foods)
        all_lists.append(other_stats)
        return np.concatenate(all_lists)

    @staticmethod
    def _encode_single(value, category):
        np_array = np.array([[value]])
        encoder = OneHotEncoder(categories=[category], sparse=False)
        onehot_encoded = encoder.fit_transform(np_array)
        collapsed = np.sum(onehot_encoded, axis=0)
        return collapsed
