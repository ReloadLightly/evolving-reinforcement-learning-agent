"""Selection, mutation, and injection must not leak shared policy state."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from erl.evolution import evolve_population


def actors():
    result = []
    for value in range(5):
        actor = nn.Linear(128, 32)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.fill_(float(value))
        result.append(actor)
    return result[:4], result[4]


def same_parameters(first, second):
    return all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


class EvolutionTests(unittest.TestCase):
    def evolve(self, population, learner, inject=False, fitness=None):
        return evolve_population(
            population, fitness or [1.0, 4.0, 2.0, 3.0],
            np.random.default_rng(7), np.random.default_rng(19), learner,
            inject, 1, ["initial_0", "initial_1", "initial_2", "initial_3"],
        )

    def test_exact_elite_and_tournament_winners(self):
        population, learner = actors()
        fitness = [1.0, 4.0, 2.0, 3.0]
        children, ids, events = self.evolve(population, learner, fitness=fitness)
        self.assertTrue(same_parameters(children[0], population[1]))
        self.assertEqual(events[0]["type"], "elite")
        self.assertEqual(events[0]["parent_id"], "initial_1")
        self.assertEqual(len(set(ids)), 4)
        for event in events[1:]:
            maximum = max(fitness[index] for index in event["tournament_indices"])
            self.assertEqual(fitness[event["parent_index"]], maximum)
            self.assertEqual(event["parent_fitness"], maximum)

    def test_masked_mutation_changes_weights_and_biases_at_declared_scale(self):
        population, learner = actors()
        children, _, events = self.evolve(population, learner)
        for child, event in zip(children[1:], events[1:]):
            parent = population[event["parent_index"]]
            change = torch.cat([(a - b).flatten() for a, b in zip(child.parameters(), parent.parameters())])
            changed = change[change != 0.0].detach().numpy()
            self.assertGreater(len(changed) / len(change), 0.07)
            self.assertLess(len(changed) / len(change), 0.13)
            self.assertGreater(np.std(changed), 0.017)
            self.assertLess(np.std(changed), 0.023)
            self.assertGreater(torch.count_nonzero(child.bias - parent.bias).item(), 0)
            self.assertAlmostEqual(float(torch.linalg.vector_norm(change.double())),
                                   event["mutation_l2"], places=12)

    def test_originals_learner_and_children_have_independent_storage(self):
        population, learner = actors()
        original_population = copy.deepcopy(population)
        original_learner = copy.deepcopy(learner)
        children, _, _ = self.evolve(population, learner, inject=True)
        self.assertTrue(same_parameters(children[3], learner))
        for first, second in zip(population, original_population):
            self.assertTrue(same_parameters(first, second))
        all_policies = [*population, learner, *children]
        addresses = [next(policy.parameters()).data_ptr() for policy in all_policies]
        self.assertEqual(len(set(addresses)), len(addresses))
        with torch.no_grad():
            for parameter in children[3].parameters():
                parameter.add_(100.0)
        self.assertTrue(same_parameters(learner, original_learner))
        self.assertTrue(same_parameters(children[0], population[1]))

    def test_injection_is_fixed_slot_and_does_not_consume_different_rng_draws(self):
        population, learner = actors()
        outcomes, states = [], []
        for inject in (False, True):
            selection_rng, mutation_rng = np.random.default_rng(7), np.random.default_rng(19)
            result = evolve_population(population, [1., 4., 2., 3.], selection_rng,
                                       mutation_rng, learner, inject, 1, ["a", "b", "c", "d"])
            outcomes.append(result)
            states.append((selection_rng.bit_generator.state, mutation_rng.bit_generator.state))
        self.assertEqual(states[0], states[1])
        for slot in range(3):
            self.assertTrue(same_parameters(outcomes[0][0][slot], outcomes[1][0][slot]))
        self.assertEqual([event["type"] for event in outcomes[1][2]],
                         ["elite", "mutation", "mutation", "injection"])

    def test_evolution_leaves_global_torch_rng_untouched(self):
        population, learner = actors()
        state = torch.random.get_rng_state().clone()
        self.evolve(population, learner, inject=True)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_ties_are_reproducible_and_population_ids_advance(self):
        population, learner = actors()
        first = self.evolve(population, learner, fitness=[1.] * 4)
        second = self.evolve(population, learner, fitness=[1.] * 4)
        self.assertEqual(first[2], second[2])
        next_result = evolve_population(first[0], [1.] * 4, np.random.default_rng(1),
                                        np.random.default_rng(2), learner, False, 2, first[1])
        self.assertFalse(set(first[1]).intersection(next_result[1]))

    def test_nonfinite_fitness_and_reused_generation_are_rejected(self):
        population, learner = actors()
        with self.assertRaises(ValueError):
            self.evolve(population, learner, fitness=[0.0, 1.0, np.nan, 3.0])
        first = self.evolve(population, learner)
        with self.assertRaises(ValueError):
            evolve_population(first[0], [1.] * 4, np.random.default_rng(1),
                              np.random.default_rng(2), learner, False, 1, first[1])


if __name__ == "__main__":
    unittest.main()
