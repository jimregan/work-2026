"""Provenance graph invariants over a set of transformations.

The provenance graph is not stored as an explicit edge list: it is derived
from `Transformation.input_ids` / `output_ids`. An edge exists from an input
artifact to an output artifact whenever some transformation consumes the
former to produce the latter. This module checks the one invariant that
must hold regardless of backend: the derived graph is acyclic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from corpus_build.model.identity import LayerId
from corpus_build.model.transformation import Transformation


class CycleError(ValueError):
    """Raised when a set of transformations would make an artifact its own
    ancestor."""

    def __init__(self, cycle: tuple[LayerId, ...]) -> None:
        self.cycle = cycle
        path = " -> ".join(f"{n.layer}:{n.local_id}" for n in cycle)
        super().__init__(f"provenance cycle detected: {path}")


def _adjacency(
    transformations: Iterable[Transformation],
) -> dict[LayerId, list[LayerId]]:
    graph: dict[LayerId, list[LayerId]] = {}
    for t in transformations:
        for source in t.input_ids:
            graph.setdefault(source, [])
            for target in t.output_ids:
                graph[source].append(target)
        for target in t.output_ids:
            graph.setdefault(target, [])
    return graph


def find_cycle(transformations: Iterable[Transformation]) -> tuple[LayerId, ...] | None:
    """Return a cycle as a tuple of artifact ids if one exists, else None."""

    graph = _adjacency(transformations)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[LayerId, int] = dict.fromkeys(graph, WHITE)
    stack: list[LayerId] = []

    def visit(node: LayerId) -> tuple[LayerId, ...] | None:
        color[node] = GRAY
        stack.append(node)
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                start = stack.index(neighbor)
                return tuple(stack[start:]) + (neighbor,)
            if color[neighbor] == WHITE:
                found = visit(neighbor)
                if found is not None:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for node in graph:
        if color[node] == WHITE:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None


def assert_acyclic(transformations: Iterable[Transformation]) -> None:
    cycle = find_cycle(list(transformations))
    if cycle is not None:
        raise CycleError(cycle)


@dataclass(frozen=True)
class ProvenanceGraph:
    """The transitive ancestry of one artifact: every transformation and
    every artifact that contributed to it, directly or indirectly."""

    root: LayerId
    transformations: tuple[Transformation, ...]
    artifacts: tuple[LayerId, ...]
