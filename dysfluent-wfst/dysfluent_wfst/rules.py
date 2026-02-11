"""Optional cdrewrite rule loading and composition."""

from __future__ import annotations

import importlib.util
from typing import Optional

import pynini


def build_sigma_star(syms: pynini.SymbolTable) -> pynini.Fst:
    """Build sigma_star: the closure over all symbols in the table.

    This is the universal language over the symbol set, needed as the
    ``sigma_star`` argument to ``pynini.cdrewrite``.
    """
    symbol_fsts = []
    for idx in range(syms.num_symbols()):
        sym = syms.find(idx)
        if sym == "<eps>" or sym == "":
            continue
        symbol_fsts.append(pynini.escape(sym))

    sigma = pynini.union(*symbol_fsts)
    return sigma.closure().optimize()


def load_rules_module(path: str):
    """Dynamically import a Python module that exports build_rules().

    The module must define:
        ``build_rules(sigma_star: pynini.Fst) -> pynini.Fst``

    Args:
        path: Filesystem path to the Python module.

    Returns:
        The imported module object.
    """
    spec = importlib.util.spec_from_file_location("_rules_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rules module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_rules"):
        raise AttributeError(
            f"Rules module {path} must export build_rules(sigma_star)"
        )
    return module


def compile_rules(
    rules_path: Optional[str],
    syms: pynini.SymbolTable,
) -> Optional[pynini.Fst]:
    """Compile phonetic rules from a Python module.

    Args:
        rules_path: Path to a Python module exporting
            ``build_rules(sigma_star) -> pynini.Fst``.
            If None, returns None (no rules applied).
        syms: Symbol table for building sigma_star.

    Returns:
        A composed rules transducer, or None if no rules_path given.
    """
    if rules_path is None:
        return None

    sigma_star = build_sigma_star(syms)
    module = load_rules_module(rules_path)
    rules_fst = module.build_rules(sigma_star)
    return rules_fst.optimize()
