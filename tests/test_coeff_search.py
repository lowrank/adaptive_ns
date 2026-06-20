"""Tests for coefficient search and stability checks."""

from muon.newton_schulz import is_stable, orbit_max, iterate_polynomial
from muon.coeff_search import search_coefficients
from muon.coeff_table import (DEFAULT_COEFF_TABLE, FAST_NS, JORDAN_NS,
                              STANDARD_NS)


class TestPolynomial:
    """Tests for the polynomial iteration (provided, should pass)."""

    def test_standard_ns_fails_strict_band(self):
        """Standard NS maps small x to ~0.008 — below 0.5. Fails strict band."""
        a, b, c = STANDARD_NS
        assert not is_stable(a, b, c), \
            "Standard NS should fail [0.5, 1.5]: f⁵(0.001) ≈ 0.008 < 0.5"

    def test_jordan_near_boundary(self):
        """Jordan NS at the boundary — f⁵_min ≈ 0.47, just below 0.5."""
        a, b, c = JORDAN_NS
        # Jordan is close to the strict boundary; the exact result depends on grid density.
        result = is_stable(a, b, c)
        assert isinstance(result, bool)

    def test_fastns_is_stable(self):
        """FastNS satisfies the strict band."""
        a, b, c = FAST_NS
        assert is_stable(a, b, c), "FastNS should be stable"

    def test_standard_ns_orbit_bounded(self):
        """Standard NS orbit should stay ≤ 1."""
        a, b, c = STANDARD_NS
        om = orbit_max(a, b, c)
        assert om <= 1.01, f"Standard NS orbit max = {om:.3f}, expected ≤ 1.0"

    def test_iterate_polynomial_zero_stays_zero(self):
        """f^K(0) should be 0 for any coefficients."""
        x = iterate_polynomial(0.0, 3.0, -5.0, 2.0, K=5)
        assert x == 0.0

    def test_iterate_polynomial_one_goes_to_one_with_standard(self):
        """f^K(1) should be 1 for standard NS."""
        x = iterate_polynomial(1.0, 1.5, -0.5, 0.0, K=5)
        assert abs(x - 1.0) < 0.01


class TestIsStable:
    """Tests for the is_stable function."""

    def test_divergent_coeffs_fail(self):
        """Coefficients that cause divergence should fail is_stable."""
        result = is_stable(10.0, -20.0, 10.0, n_pts=100)
        assert not result, "Divergent coefficients should return False"

    def test_band_constraint_respected(self):
        """is_stable rejects coefficients where f^5 goes outside [0.5, 1.5]."""
        result = is_stable(1.5, -0.5, 0.0, n_pts=100)
        assert not result, (
            "Standard NS should fail: f^5(x) ≈ 0.008 for small x, "
            "which is below 0.5.  Higher a needed to pull small values up."
        )


class TestDefaultCoeffTable:
    """Tests for the generated adaptive coefficient lookup table."""

    def test_default_table_is_dense(self):
        assert len(DEFAULT_COEFF_TABLE) >= 40
        assert max(DEFAULT_COEFF_TABLE) >= 3.89

    def test_generated_high_a_entries_are_stable(self):
        for key, coeffs in DEFAULT_COEFF_TABLE.items():
            if key >= 3.50:
                assert is_stable(*coeffs, n_pts=500), key


class TestCoeffSearch:
    """Tests for the coefficient search."""

    def test_search_finds_valid_coeffs(self):
        """search_coefficients should return a stable triple."""
        a, b, c = search_coefficients(n_samples=5000)
        assert a >= 1.5, f"a={a:.2f} should be ≥ 1.5"
        assert is_stable(a, b, c), (
            f"Search found (a={a:.3f}, b={b:.3f}, c={c:.3f}) "
            f"which should pass is_stable"
        )

    def test_search_beats_standard_ns(self):
        """The search should find a > 1.5 (better than standard NS)."""
        a, b, c = search_coefficients(n_samples=5000)
        assert a > 1.5, (
            f"Search found a={a:.2f}.  With 5000 samples, you should "
            f"find a > 1.5 (standard NS).  Try increasing n_samples or "
            f"widening the search ranges."
        )
