import math

from dysfluent_wfst.ref_fst import dynamic_error_probability


def test_dynamic_error_probability_matches_paper_formula():
    err0 = 0.01
    distance = 2

    probability = dynamic_error_probability(err0, distance)

    assert probability == (
        err0
        * (1 / math.sqrt(2 * math.pi))
        * math.exp(-(distance ** 2) / 2)
    )
