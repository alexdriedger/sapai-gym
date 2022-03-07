import gym
from gym import spaces
import numpy as np
from typing import Optional
from sklearn.preprocessing import OneHotEncoder

from sapai import Player, Pet, Food, Battle
from sapai.agents import CombinatorialSearch


class SuperAutoPetsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_ACTIONS = 100
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

    def __init__(self, opponent_generator):
        super(SuperAutoPetsEnv, self).__init__()

        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        len_obs_space = (len(self.ALL_PETS) + 3) * 11 + (len(self.ALL_FOODS) + 1) * 2 + 5
        self.observation_space = spaces.Box(low=0, high=1, shape=(len_obs_space,), dtype=np.uint8)
        self.reward_range = (0, 10)

        self.player = Player()
        self.just_froze = False
        self.just_reordered = False

        self.opponent_generator = opponent_generator
        self.opponents = opponent_generator(25)

        self.bad_action_reward_sum = 0

    def step(self, action):
        if not self._is_valid_action(action):
            # Teach agent to play valid actions
            self.bad_action_reward_sum += self.BAD_ACTION_PENALTY
        else:
            # Resolve action
            player_to_act = self.player
            action_to_play = self._map_int_to_action(action)
            action_name = self._get_action_name(action_to_play).split(".")[-1]
            action_method = getattr(player_to_act, action_name)
            action_method(*action_to_play[1:])

            # If turn is ended, play an opponent
            if action_name == "end_turn":
                opponent = self.opponents[self.player.turn - 1]
                battle_result = Battle(self.player.team, opponent).battle()
                self._player_fight_outcome(battle_result)

        obs = self._encode_state()
        reward = self.get_reward()
        done = self.is_done()
        info = self.player

        return obs, reward, done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        self.player = Player()
        self.just_froze = False
        self.just_reordered = False

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
        return self.player.wins

    @staticmethod
    def _avail_roll(player_to_check: Player):
        action_list = []
        if player_to_check.gold > 1:
            action_list.append((player_to_check.roll,))
        return action_list

    @staticmethod
    def _avail_end_turn(player_to_check: Player):
        return [(player_to_check.end_turn,)]

    @staticmethod
    def _get_action_name(input_action):
        return str(input_action[0].__name__)

    def _avail_actions(self):
        # TODO : REMOVE AND JUST GENERATE ACTION NUMBERS DIRECTLY
        player = self.player
        cs = CombinatorialSearch()

        action_list = []
        action_list += self._avail_end_turn(player)
        action_list += CombinatorialSearch.avail_buy_pets(cs, player)
        action_list += CombinatorialSearch.avail_buy_food(cs, player)
        action_list += CombinatorialSearch.avail_buy_combine(cs, player)
        action_list += CombinatorialSearch.avail_team_combine(cs, player)
        action_list += CombinatorialSearch.avail_sell(cs, player)
        # action_list += CombinatorialSearch.avail_sell_buy(cs, player)
        action_list += self._avail_roll(player)
        # TODO : RE-ENABLE REORDERING
        # if not self.just_reordered:
        #     action_list += CombinatorialSearch.avail_team_order(cs, player)
        # TODO : FREEZE LIST
        return action_list

    def _is_valid_action(self, action: int) -> bool:
        # TODO : REFACTOR TO JUST GENERATE ACTION NUMBERS DIRECTLY
        # Get all avail actions and map to integer representation
        avail_action_ints = [self._map_action_to_int(a) for a in self._avail_actions()]
        return action in avail_action_ints

    # WARNING : These map actions from CombinatorialSearch to actions in `_map_int_to_action`. Mapping multiple times
    # will not produce the same result. TODO This should be fixed in the future
    def _map_action_to_int(self, action) -> int:
        if self._get_action_name(action) == "end_turn":
            return self.ACTION_BASE_NUM["end_turn"]
        if self._get_action_name(action) == "buy_pet":
            pet = self.player.shop[action[1]].item
            for index, p in enumerate(self.player.shop.pets):
                if pet == p:
                    return self.ACTION_BASE_NUM["buy_pet"] + index
            print(f"Index of pet was {action[1]}")
            raise RuntimeError(f"Unknown buy pet action {action}\n{self.player}")
        if self._get_action_name(action) == "buy_food":
            food = self.player.shop[action[1]].item
            for index, f in enumerate(self.player.shop.foods):
                if food == f:
                    return self.ACTION_BASE_NUM["buy_food"] + (index * self.MAX_TEAM_PETS) + action[2]
            raise RuntimeError(f"Unknown buy food action {action}\n{self.player}")
        if self._get_action_name(action) == "buy_combine":
            pet = self.player.shop[action[1]].item
            for index, p in enumerate(self.player.shop.pets):
                if pet == p:
                    return self.ACTION_BASE_NUM["buy_combine"] + (index * self.MAX_TEAM_PETS) + action[2]
            raise RuntimeError(f"Unknown buy combine pet action {action}\n{self.player}")
        if self._get_action_name(action) == "combine":
            indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            ind = indexes.index((action[1], action[2]))
            return self.ACTION_BASE_NUM["combine"] + ind
        if self._get_action_name(action) == "sell":
            return self.ACTION_BASE_NUM["sell"] + action[1]
        if self._get_action_name(action) == "roll":
            return self.ACTION_BASE_NUM["roll"]
        raise RuntimeError(f"Action {self._get_action_name(action)} did not map to an action int in the game")

    def _map_int_to_action(self, action_int):
        if self.ACTION_BASE_NUM["end_turn"] <= action_int < self.ACTION_BASE_NUM["buy_pet"]:
            return self.player.end_turn,
        if self.ACTION_BASE_NUM["buy_pet"] <= action_int < self.ACTION_BASE_NUM["buy_food"]:
            # Buy one of the 6 possible pets
            index = action_int - self.ACTION_BASE_NUM["buy_pet"]
            pet_to_buy = self.player.shop.pets[index]
            return self.player.buy_pet, pet_to_buy
        if self.ACTION_BASE_NUM["buy_food"] <= action_int < self.ACTION_BASE_NUM["buy_combine"]:
            base_index = action_int - self.ACTION_BASE_NUM["buy_food"]
            food_index = base_index // self.MAX_TEAM_PETS
            food_to_buy = self.player.shop.foods[food_index]
            pet_index = base_index % self.MAX_TEAM_PETS
            return self.player.buy_food, food_to_buy, pet_index
        if self.ACTION_BASE_NUM["buy_combine"] <= action_int < self.ACTION_BASE_NUM["combine"]:
            base_index = action_int - self.ACTION_BASE_NUM["buy_combine"]
            shop_pet_index = base_index // self.MAX_TEAM_PETS
            pet_to_buy = self.player.shop.pets[shop_pet_index]
            team_pet_index = base_index % self.MAX_TEAM_PETS
            return self.player.buy_combine, pet_to_buy, team_pet_index
        if self.ACTION_BASE_NUM["combine"] <= action_int < self.ACTION_BASE_NUM["sell"]:
            base_index = action_int - self.ACTION_BASE_NUM["combine"]
            indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            team_pet_index_1, team_pet_index_2 = indexes[base_index]
            return self.player.combine, team_pet_index_1, team_pet_index_2
        if self.ACTION_BASE_NUM["sell"] <= action_int < self.ACTION_BASE_NUM["roll"]:
            base_index = action_int - self.ACTION_BASE_NUM["sell"]
            return self.player.sell, base_index
        if self.ACTION_BASE_NUM["roll"] <= action_int:
            return self.player.roll,
        raise RuntimeError(f"Action int {action_int} did not map to an action in the game")

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
