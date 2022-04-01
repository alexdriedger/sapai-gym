from unittest import TestCase

from sapai import Team
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator


class TestOpponentGenerators(TestCase):
    def test_random_generator(self):
        opponents = random_opp_generator(25)

    def test_biggest_numbers_horizontal_opp_generator(self):
        opponents = biggest_numbers_horizontal_opp_generator(25)
        scores = [self.map_team_to_total_attack_and_health(team) for team in opponents]
        sorted_scores = sorted(scores)

        # Check that the team is always getting stronger
        self.assertEqual(scores, sorted_scores)

    @staticmethod
    def map_team_to_total_attack_and_health(team: Team):
        total = 0
        for pet_slot in team:
            pet = pet_slot.pet
            if pet.name != "pet-none":
                total += pet._attack
                total += pet._health
        return total
