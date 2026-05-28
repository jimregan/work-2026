import sys
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "pynini" not in sys.modules:
    class _FakeWeight:
        @staticmethod
        def zero(_weight_type):
            return 0

    fake_pynini = types.SimpleNamespace(
        NO_STATE_ID=-1,
        Fst=object,
        SymbolTable=object,
        Weight=_FakeWeight,
        compose=lambda left, right: None,
        union=lambda *fsts: fsts[0] if fsts else None,
    )
    sys.modules["pynini"] = fake_pynini
