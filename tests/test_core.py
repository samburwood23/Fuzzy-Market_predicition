import numpy as np
import pytest

from fuzzy_analytics.core import FuzzyInferenceSystem, FuzzyRule, FuzzySet, FuzzyVariable
from fuzzy_analytics.core import membership_functions as mf


class TestMembershipFunctions:
    def test_triangular_shape(self):
        f = mf.triangular(0.0, 1.0, 2.0)
        assert f(-1.0) == 0.0
        assert f(0.5) == pytest.approx(0.5)
        assert f(1.0) == 1.0
        assert f(1.5) == pytest.approx(0.5)
        assert f(3.0) == 0.0

    def test_shoulders_include_their_edge(self):
        left = mf.trapezoidal(0.0, 0.0, 1.0, 2.0)
        right = mf.trapezoidal(0.0, 1.0, 2.0, 2.0)
        assert left(0.0) == 1.0
        assert left(-0.1) == 0.0
        assert right(2.0) == 1.0
        assert right(2.1) == 0.0

    def test_accepts_arrays(self):
        f = mf.triangular(0.0, 1.0, 2.0)
        np.testing.assert_allclose(f(np.array([0.0, 0.5, 1.0, 2.0])), [0.0, 0.5, 1.0, 0.0])

    def test_scalar_returns_float(self):
        assert isinstance(mf.gaussian(0.0, 1.0)(0.3), float)

    def test_rejects_unordered_points(self):
        with pytest.raises(ValueError):
            mf.triangular(2.0, 1.0, 0.0)

    def test_sigmoid_is_stable_for_extreme_inputs(self):
        f = mf.sigmoid(0.0, 10.0)
        assert f(1e6) == pytest.approx(1.0)
        assert f(-1e6) == pytest.approx(0.0)
        assert f(0.0) == pytest.approx(0.5)


def _simple_system():
    x = FuzzyVariable("x", (0.0, 10.0))
    x.add_set("low", FuzzySet(mf.trapezoidal(0.0, 0.0, 2.0, 5.0)))
    x.add_set("high", FuzzySet(mf.trapezoidal(5.0, 8.0, 10.0, 10.0)))
    y = FuzzyVariable("y", (0.0, 1.0))
    y.add_set("low", FuzzySet(mf.triangular(0.0, 0.2, 0.4)))
    y.add_set("high", FuzzySet(mf.triangular(0.6, 0.8, 1.0)))
    rules = [FuzzyRule([(x, "low")], (y, "low")), FuzzyRule([(x, "high")], (y, "high"))]
    return FuzzyInferenceSystem([x], y, rules, universe_resolution=501), x, y


class TestRulesAndInference:
    def test_rule_rejects_unknown_set(self):
        x = FuzzyVariable("x", (0.0, 1.0))
        x.add_set("low", FuzzySet(mf.triangular(0.0, 0.0, 1.0)))
        with pytest.raises(ValueError, match="hgih"):
            FuzzyRule([(x, "hgih")], (x, "low"))

    def test_rule_string(self):
        _, x, y = _simple_system()
        assert str(FuzzyRule([(x, "low")], (y, "high"), weight=0.5)) == "IF x is low THEN y is high (weight 0.5)"

    def test_centroid_of_symmetric_output(self):
        system, _, _ = _simple_system()
        assert system.evaluate({"x": 0.0}) == pytest.approx(0.2, abs=1e-3)
        assert system.evaluate({"x": 10.0}) == pytest.approx(0.8, abs=1e-3)

    def test_out_of_domain_inputs_are_clipped(self):
        system, _, _ = _simple_system()
        assert system.evaluate({"x": 50.0}) == pytest.approx(system.evaluate({"x": 10.0}))

    def test_missing_input(self):
        system, _, _ = _simple_system()
        with pytest.raises(ValueError, match="Missing input"):
            system.evaluate({})

    def test_non_finite_input(self):
        system, _, _ = _simple_system()
        with pytest.raises(ValueError, match="finite"):
            system.evaluate({"x": float("nan")})

    def test_no_rule_fired_returns_midpoint(self):
        x = FuzzyVariable("x", (0.0, 10.0))
        x.add_set("mid", FuzzySet(mf.triangular(4.0, 5.0, 6.0)))
        y = FuzzyVariable("y", (0.0, 1.0))
        y.add_set("high", FuzzySet(mf.triangular(0.6, 0.8, 1.0)))
        system = FuzzyInferenceSystem([x], y, [FuzzyRule([(x, "mid")], (y, "high"))])
        assert system.evaluate({"x": 0.0}) == 0.5

    def test_rule_activations(self):
        system, _, _ = _simple_system()
        strengths = [s for _, s in system.rule_activations({"x": 0.0})]
        assert strengths == [1.0, 0.0]
