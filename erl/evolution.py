"""A small, auditable evolutionary population for the CPU ERL experiment."""

from __future__ import annotations

import copy
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import nn


MUTATION_PROBABILITY = 0.1
MUTATION_SIGMA = 0.02
POPULATION_SIZE = 4
INJECTION_SLOT = 3


def _winner(indices: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> int:
    """Choose the best contender, breaking genuine fitness ties with ``rng``."""
    best = indices[fitness[indices] == np.max(fitness[indices])]
    # A duplicated contender is not a tie between distinct individuals.
    best = np.unique(best)
    return int(best[0] if len(best) == 1 else rng.choice(best))


def _mutated_copy(actor: nn.Module, rng: np.random.Generator):
    child = copy.deepcopy(actor)
    selected = 0
    count = 0
    squared_change = 0.0
    with torch.no_grad():
        for parameter in child.parameters():
            shape = tuple(parameter.shape)
            mask = rng.random(shape) < MUTATION_PROBABILITY
            delta = rng.normal(0.0, MUTATION_SIGMA, shape) * mask
            # NumPy supplies every random draw; torch's training RNG is untouched.
            change = torch.as_tensor(delta, dtype=parameter.dtype, device=parameter.device)
            before = parameter.clone()
            parameter.add_(change)
            actual_change = (parameter - before).double()
            squared_change += float(torch.sum(actual_change.square()).item())
            selected += int(np.count_nonzero(mask))
            count += int(parameter.numel())
    return child, squared_change ** 0.5, selected, count


def evolve_population(
    population: list[nn.Module],
    fitness: list[float],
    selection_rng: np.random.Generator,
    mutation_rng: np.random.Generator,
    learner: nn.Module,
    inject: bool,
    generation: int,
    lineage_ids: list[str],
) -> tuple[list[nn.Module], list[str], list[dict[str, Any]]]:
    """Create four independent actors: one elite and three mutated offspring.

    Each offspring uses a size-two tournament sampled with replacement. On an
    injection generation, destination slot 3 is replaced by a parameter copy of
    the updated learner. The discarded mutation is still performed, preserving
    the mutation RNG schedule. No optimizer or critic is copied into a population
    actor. ``generation`` labels the new destination generation and must increase
    between calls so that lineage identifiers remain unique.
    """
    if len(population) != POPULATION_SIZE or len(fitness) != POPULATION_SIZE:
        raise ValueError("This protocol requires exactly four actors and four fitness values")
    if len(lineage_ids) != POPULATION_SIZE or len(set(lineage_ids)) != POPULATION_SIZE:
        raise ValueError("Provide four distinct lineage IDs")
    if not isinstance(generation, Integral) or isinstance(generation, bool) or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    scores = np.asarray(fitness, dtype=np.float64)
    if scores.shape != (POPULATION_SIZE,) or not np.isfinite(scores).all():
        raise ValueError("Fitness values must be four finite scalars")
    new_ids = [f"g{int(generation):06d}_s{slot}" for slot in range(POPULATION_SIZE)]
    if set(new_ids).intersection(lineage_ids):
        raise ValueError("generation must label a new generation, not the source generation")

    elite = _winner(np.arange(POPULATION_SIZE), scores, selection_rng)
    new_population = [copy.deepcopy(population[elite])]
    events: list[dict[str, Any]] = [{
        "generation": int(generation), "slot": 0, "individual_id": new_ids[0],
        "parent_id": lineage_ids[elite], "parent_index": elite, "type": "elite",
        "fitness": float(scores[elite]), "parent_fitness": float(scores[elite]),
        "mutation_l2": 0.0, "mutation_selected_fraction": 0.0,
    }]
    for slot in range(1, POPULATION_SIZE):
        contenders = selection_rng.integers(0, POPULATION_SIZE, size=2)
        parent = _winner(contenders, scores, selection_rng)
        child, magnitude, selected, count = _mutated_copy(population[parent], mutation_rng)
        new_population.append(child)
        events.append({
            "generation": int(generation), "slot": slot, "individual_id": new_ids[slot],
            "parent_id": lineage_ids[parent], "parent_index": parent, "type": "mutation",
            "fitness": float(scores[parent]), "parent_fitness": float(scores[parent]),
            "mutation_l2": magnitude,
            "mutation_selected_fraction": selected / count if count else 0.0,
            "mutation_selected_parameters": selected, "total_parameters": count,
            "tournament_indices": [int(i) for i in contenders],
        })
    if inject:
        discarded = events[INJECTION_SLOT]
        new_population[INJECTION_SLOT] = copy.deepcopy(learner)
        events[INJECTION_SLOT] = {
            "generation": int(generation), "slot": INJECTION_SLOT,
            "individual_id": new_ids[INJECTION_SLOT],
            "parent_id": f"learner_g{int(generation):06d}", "parent_index": None,
            "type": "injection", "fitness": None, "parent_fitness": None,
            "mutation_l2": 0.0, "mutation_selected_fraction": 0.0,
            "discarded_offspring_parent_id": discarded["parent_id"],
            "discarded_mutation_l2": discarded["mutation_l2"],
        }
    return new_population, new_ids, events
