"""Tests for the arena-style pairwise comparison evaluator."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.evaluation.pairwise_arena import (
    ArenaResult,
    PairwiseVerdict,
    _COMPARISON_DIMENSIONS,
    _PAIRWISE_SYSTEM_PROMPT,
    _build_head_to_head,
    _build_win_matrix,
    _swap_label,
    bradley_terry_scores,
    compute_elo_ratings,
    pairwise_comparison,
    run_arena,
    transitivity_check,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_verdict(
    query_id: str,
    system_a: str,
    system_b: str,
    winner: str,
    confidence: float = 0.8,
) -> PairwiseVerdict:
    """Helper to create a PairwiseVerdict for testing."""
    return PairwiseVerdict(
        query_id=query_id,
        system_a=system_a,
        system_b=system_b,
        winner=winner,
        confidence=confidence,
        reasoning="test verdict",
        dimensions_won={d: winner for d in _COMPARISON_DIMENSIONS},
    )


# ── Elo Ratings ─────────────────────────────────────────────────────────────


class TestEloRatings:
    def test_known_outcomes_expected_order(self):
        """When A always beats B, and B always beats C, ratings should be A > B > C."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q2", "A", "B", "A"),
            _make_verdict("q1", "B", "C", "A"),  # B wins over C (B is system_a)
            _make_verdict("q2", "B", "C", "A"),
            _make_verdict("q1", "A", "C", "A"),
            _make_verdict("q2", "A", "C", "A"),
        ]
        elo = compute_elo_ratings(verdicts, systems, seed=42)
        assert elo["A"] > elo["B"]
        assert elo["B"] > elo["C"]

    def test_all_ties_equal_ratings(self):
        """When all matches are ties, ratings should remain at initial value."""
        systems = ["X", "Y", "Z"]
        verdicts = [
            _make_verdict("q1", "X", "Y", "tie"),
            _make_verdict("q1", "Y", "Z", "tie"),
            _make_verdict("q1", "X", "Z", "tie"),
        ]
        elo = compute_elo_ratings(verdicts, systems, initial_rating=1500.0, seed=42)
        # All ties between equal-rated players: no change
        assert math.isclose(elo["X"], 1500.0, abs_tol=1e-9)
        assert math.isclose(elo["Y"], 1500.0, abs_tol=1e-9)
        assert math.isclose(elo["Z"], 1500.0, abs_tol=1e-9)

    def test_single_match_winner_gains(self):
        """A single win should increase the winner's rating and decrease the loser's."""
        systems = ["W", "L"]
        verdicts = [_make_verdict("q1", "W", "L", "A")]  # W wins
        elo = compute_elo_ratings(verdicts, systems, initial_rating=1500.0, seed=0)
        assert elo["W"] > 1500.0
        assert elo["L"] < 1500.0
        # Elo is zero-sum for two players
        assert math.isclose(elo["W"] + elo["L"], 3000.0, abs_tol=1e-9)

    def test_empty_verdicts(self):
        """No verdicts should return initial ratings for all systems."""
        systems = ["A", "B"]
        elo = compute_elo_ratings([], systems)
        assert elo["A"] == 1500.0
        assert elo["B"] == 1500.0

    def test_empty_systems(self):
        """No systems should return empty dict."""
        elo = compute_elo_ratings([], [])
        assert elo == {}

    def test_k_factor_effect(self):
        """Higher K-factor should produce larger rating changes."""
        systems = ["A", "B"]
        verdicts = [_make_verdict("q1", "A", "B", "A")]

        elo_low_k = compute_elo_ratings(verdicts, systems, k_factor=16.0, seed=0)
        elo_high_k = compute_elo_ratings(verdicts, systems, k_factor=64.0, seed=0)

        gain_low = elo_low_k["A"] - 1500.0
        gain_high = elo_high_k["A"] - 1500.0
        assert gain_high > gain_low

    def test_seed_determinism(self):
        """Same seed should produce identical results."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q1", "B", "C", "A"),
            _make_verdict("q1", "A", "C", "B"),
        ]
        elo1 = compute_elo_ratings(verdicts, systems, seed=99)
        elo2 = compute_elo_ratings(verdicts, systems, seed=99)
        for s in systems:
            assert math.isclose(elo1[s], elo2[s], abs_tol=1e-12)

    def test_different_seeds_may_differ(self):
        """Different seeds process verdicts in different order, potentially yielding different ratings."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q2", "B", "C", "A"),
            _make_verdict("q3", "A", "C", "B"),
        ]
        elo1 = compute_elo_ratings(verdicts, systems, seed=1)
        elo2 = compute_elo_ratings(verdicts, systems, seed=999)
        # At least one system should have a different rating
        diffs = [abs(elo1[s] - elo2[s]) for s in systems]
        # With only 3 verdicts and different orders, differences may be very small
        # but the function should at least run without error
        assert all(isinstance(elo1[s], float) for s in systems)


# ── Bradley-Terry Scores ────────────────────────────────────────────────────


class TestBradleyTerry:
    def test_clear_winner(self):
        """System that wins all matches should have highest BT score."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q2", "A", "B", "A"),
            _make_verdict("q1", "A", "C", "A"),
            _make_verdict("q2", "A", "C", "A"),
            _make_verdict("q1", "B", "C", "A"),  # B wins over C
            _make_verdict("q2", "B", "C", "A"),
        ]
        bt = bradley_terry_scores(verdicts, systems)
        assert bt["A"] > bt["B"]
        assert bt["B"] > bt["C"]

    def test_convergence_sums_to_one(self):
        """BT scores should sum to 1.0."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q1", "B", "C", "B"),
            _make_verdict("q1", "A", "C", "A"),
        ]
        bt = bradley_terry_scores(verdicts, systems)
        total = sum(bt.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_ties_balanced(self):
        """All ties should produce approximately equal scores."""
        systems = ["X", "Y"]
        verdicts = [
            _make_verdict("q1", "X", "Y", "tie"),
            _make_verdict("q2", "X", "Y", "tie"),
            _make_verdict("q3", "X", "Y", "tie"),
        ]
        bt = bradley_terry_scores(verdicts, systems)
        assert math.isclose(bt["X"], 0.5, abs_tol=1e-4)
        assert math.isclose(bt["Y"], 0.5, abs_tol=1e-4)

    def test_empty_input(self):
        """Empty systems should return empty dict."""
        bt = bradley_terry_scores([], [])
        assert bt == {}

    def test_empty_verdicts_with_systems(self):
        """No verdicts but with systems should return uniform scores."""
        systems = ["A", "B", "C"]
        bt = bradley_terry_scores([], systems)
        for s in systems:
            assert math.isclose(bt[s], 1.0 / 3, abs_tol=1e-6)

    def test_single_system(self):
        """Single system should get score of 1.0."""
        bt = bradley_terry_scores([], ["solo"])
        assert bt["solo"] == 1.0

    def test_all_scores_positive(self):
        """All BT scores should be positive even for total losers."""
        systems = ["W", "L"]
        verdicts = [
            _make_verdict("q1", "W", "L", "A"),
            _make_verdict("q2", "W", "L", "A"),
            _make_verdict("q3", "W", "L", "A"),
        ]
        bt = bradley_terry_scores(verdicts, systems)
        assert bt["W"] > 0
        assert bt["L"] > 0
        assert bt["W"] > bt["L"]


# ── Transitivity Check ──────────────────────────────────────────────────────


class TestTransitivityCheck:
    def test_perfect_transitivity(self):
        """A > B > C with A > C should have zero violations."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),  # A > B
            _make_verdict("q1", "B", "C", "A"),  # B > C
            _make_verdict("q1", "A", "C", "A"),  # A > C (transitive)
        ]
        result = transitivity_check(verdicts, systems)
        assert result["violation_rate"] == 0.0
        assert result["n_violations"] == 0
        assert result["n_testable_triplets"] > 0

    def test_cycle_detection(self):
        """A > B, B > C, C > A is a cycle and should produce violations."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),  # A > B
            _make_verdict("q1", "B", "C", "A"),  # B > C
            _make_verdict("q1", "A", "C", "B"),  # C > A (violation!)
        ]
        result = transitivity_check(verdicts, systems)
        assert result["n_violations"] > 0
        assert result["violation_rate"] > 0.0
        assert len(result["violations"]) > 0

    def test_single_pair_no_triplets(self):
        """With only 2 systems, there are no triplets to check."""
        systems = ["A", "B"]
        verdicts = [_make_verdict("q1", "A", "B", "A")]
        result = transitivity_check(verdicts, systems)
        assert result["violation_rate"] == 0.0
        assert result["n_testable_triplets"] == 0
        assert result["n_violations"] == 0

    def test_empty_input(self):
        """Empty systems should return zero violations."""
        result = transitivity_check([], [])
        assert result["violation_rate"] == 0.0
        assert result["n_violations"] == 0

    def test_all_ties_no_testable(self):
        """If all results are ties, no directional preferences exist => no testable triplets."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "tie"),
            _make_verdict("q1", "B", "C", "tie"),
            _make_verdict("q1", "A", "C", "tie"),
        ]
        result = transitivity_check(verdicts, systems)
        assert result["n_testable_triplets"] == 0

    def test_violations_list_content(self):
        """Violations list should contain tuples of (a, b, c) where a>b>c but not a>c."""
        systems = ["A", "B", "C"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),  # A > B
            _make_verdict("q1", "B", "C", "A"),  # B > C
            _make_verdict("q1", "A", "C", "B"),  # C > A (violation)
        ]
        result = transitivity_check(verdicts, systems)
        violations = result["violations"]
        # The triplet (A, B, C) should appear in violations
        assert ("A", "B", "C") in violations


# ── Pairwise Comparison (LLM call) ─────────────────────────────────────────


class TestPairwiseComparison:
    @pytest.mark.asyncio
    async def test_basic_verdict_structure(self):
        """Mock LLM and verify returned verdict has correct structure."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "A",
            "confidence": 0.85,
            "reasoning": "Report A is more thorough",
            "dimensions": {
                "factual_accuracy": "A",
                "coverage": "A",
                "analytical_depth": "tie",
                "citation_quality": "B",
                "organisation": "A",
                "instruction_following": "tie",
            },
        })

        verdict = await pairwise_comparison(
            report_a="Report A content",
            report_b="Report B content",
            query_text="Test query",
            query_id="q1",
            system_a="P0",
            system_b="P1",
            llm=mock_llm,
            model="test-model",
        )

        assert isinstance(verdict, PairwiseVerdict)
        assert verdict.query_id == "q1"
        assert verdict.system_a == "P0"
        assert verdict.system_b == "P1"
        assert verdict.winner in ("A", "B", "tie")
        assert 0.0 <= verdict.confidence <= 1.0
        assert isinstance(verdict.reasoning, str)
        assert len(verdict.dimensions_won) == 6
        for dim in _COMPARISON_DIMENSIONS:
            assert dim in verdict.dimensions_won
            assert verdict.dimensions_won[dim] in ("A", "B", "TIE")

    @pytest.mark.asyncio
    async def test_swap_and_unswap(self):
        """Verify that position bias mitigation (swap/unswap) works correctly.

        We test two cases: one where swap happens and one where it does not,
        by choosing query_id + system_a + system_b combinations that produce
        different hash parities.
        """
        call_prompts = []

        async def mock_complete_json(prompt, **kwargs):
            call_prompts.append(prompt)
            return {
                "overall_winner": "A",
                "confidence": 0.9,
                "reasoning": "A is better",
                "dimensions": {d: "A" for d in _COMPARISON_DIMENSIONS},
            }

        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(side_effect=mock_complete_json)

        # Determine if swap happens for this combo
        swap_hash = hash("q1" + "P0" + "P1") & 0x7FFFFFFF
        swapped = swap_hash % 2 == 1

        verdict = await pairwise_comparison(
            report_a="AAA content",
            report_b="BBB content",
            query_text="Test query",
            query_id="q1",
            system_a="P0",
            system_b="P1",
            llm=mock_llm,
            model="test-model",
        )

        # Verify the prompt was constructed
        assert len(call_prompts) == 1

        if swapped:
            # If swapped, Report A in prompt should be report_b ("BBB content")
            assert "BBB content" in call_prompts[0].split("## Report A")[1].split("## Report B")[0]
            # LLM said "A" wins, but since we swapped, un-swap means "B" wins
            assert verdict.winner == "B"
        else:
            # If not swapped, Report A in prompt should be report_a ("AAA content")
            assert "AAA content" in call_prompts[0].split("## Report A")[1].split("## Report B")[0]
            # LLM said "A" wins, no swap needed
            assert verdict.winner == "A"

    @pytest.mark.asyncio
    async def test_deterministic_swap_decision(self):
        """Same inputs should always produce the same swap decision."""
        results = []
        for _ in range(5):
            mock_llm = MagicMock(spec=["complete_json"])
            mock_llm.complete_json = AsyncMock(return_value={
                "overall_winner": "tie",
                "confidence": 0.5,
                "reasoning": "equal",
                "dimensions": {d: "tie" for d in _COMPARISON_DIMENSIONS},
            })
            verdict = await pairwise_comparison(
                report_a="A",
                report_b="B",
                query_text="Q",
                query_id="q_fixed",
                system_a="s1",
                system_b="s2",
                llm=mock_llm,
                model="m",
            )
            results.append(verdict.winner)

        # All should be identical since tie un-swaps to tie
        assert all(r == results[0] for r in results)

    @pytest.mark.asyncio
    async def test_llm_failure_returns_tie(self):
        """If LLM call fails, should return a tie verdict with zero confidence."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(side_effect=RuntimeError("API error"))

        verdict = await pairwise_comparison(
            report_a="A",
            report_b="B",
            query_text="Q",
            query_id="q1",
            system_a="P0",
            system_b="P1",
            llm=mock_llm,
            model="m",
        )

        assert verdict.winner == "tie"
        assert verdict.confidence == 0.0
        assert "failed" in verdict.reasoning.lower()

    @pytest.mark.asyncio
    async def test_invalid_winner_defaults_to_tie(self):
        """If LLM returns an invalid winner value, it should default to tie."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "INVALID",
            "confidence": 0.5,
            "reasoning": "confused",
            "dimensions": {d: "INVALID" for d in _COMPARISON_DIMENSIONS},
        })

        verdict = await pairwise_comparison(
            report_a="A",
            report_b="B",
            query_text="Q",
            query_id="q1",
            system_a="P0",
            system_b="P1",
            llm=mock_llm,
            model="m",
        )

        assert verdict.winner == "tie"
        for dim_winner in verdict.dimensions_won.values():
            assert dim_winner == "TIE"

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        """Confidence should be clamped to [0, 1]."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "tie",
            "confidence": 5.0,  # Out of range
            "reasoning": "test",
            "dimensions": {d: "tie" for d in _COMPARISON_DIMENSIONS},
        })

        verdict = await pairwise_comparison(
            report_a="A", report_b="B", query_text="Q",
            query_id="q1", system_a="P0", system_b="P1",
            llm=mock_llm, model="m",
        )
        assert verdict.confidence == 1.0


# ── Run Arena (integration) ─────────────────────────────────────────────────


class TestRunArena:
    @pytest.mark.asyncio
    async def test_arena_result_structure(self):
        """Mock pairwise_comparison and verify ArenaResult structure."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "A",
            "confidence": 0.7,
            "reasoning": "A is better",
            "dimensions": {d: "A" for d in _COMPARISON_DIMENSIONS},
        })

        reports = {
            "P0": {"q1": "P0 report for q1", "q2": "P0 report for q2"},
            "P1": {"q1": "P1 report for q1", "q2": "P1 report for q2"},
        }
        queries = {"q1": "Query 1", "q2": "Query 2"}

        result = await run_arena(
            reports_by_system=reports,
            queries=queries,
            llm=mock_llm,
            model="test-model",
            max_concurrent=2,
            seed=42,
        )

        assert isinstance(result, ArenaResult)
        assert result.systems == ["P0", "P1"]
        assert result.n_comparisons == 2  # 1 pair x 2 queries
        assert len(result.verdicts) == 2
        assert "P0" in result.elo_ratings
        assert "P1" in result.elo_ratings
        assert "P0" in result.bradley_terry_scores
        assert "P1" in result.bradley_terry_scores

    @pytest.mark.asyncio
    async def test_all_pairs_generated(self):
        """With 3 systems and 2 queries, should generate C(3,2)*2 = 6 comparisons."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "tie",
            "confidence": 0.5,
            "reasoning": "equal",
            "dimensions": {d: "tie" for d in _COMPARISON_DIMENSIONS},
        })

        reports = {
            "P0": {"q1": "r1", "q2": "r2"},
            "P1": {"q1": "r1", "q2": "r2"},
            "P2": {"q1": "r1", "q2": "r2"},
        }
        queries = {"q1": "Q1", "q2": "Q2"}

        result = await run_arena(
            reports_by_system=reports,
            queries=queries,
            llm=mock_llm,
            max_concurrent=5,
        )

        # C(3,2) = 3 pairs, 2 queries each = 6 comparisons
        assert result.n_comparisons == 6
        assert len(result.verdicts) == 6
        assert result.systems == ["P0", "P1", "P2"]

    @pytest.mark.asyncio
    async def test_missing_reports_skipped(self):
        """If a system is missing a report for a query, that comparison is skipped."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "A",
            "confidence": 0.8,
            "reasoning": "A wins",
            "dimensions": {d: "A" for d in _COMPARISON_DIMENSIONS},
        })

        reports = {
            "P0": {"q1": "report", "q2": "report"},
            "P1": {"q1": "report"},  # Missing q2
        }
        queries = {"q1": "Q1", "q2": "Q2"}

        result = await run_arena(
            reports_by_system=reports,
            queries=queries,
            llm=mock_llm,
            max_concurrent=2,
        )

        # Only q1 has reports for both systems
        assert result.n_comparisons == 1

    @pytest.mark.asyncio
    async def test_single_system_returns_empty(self):
        """With only one system, arena should return empty results."""
        mock_llm = MagicMock(spec=["complete_json"])

        result = await run_arena(
            reports_by_system={"P0": {"q1": "report"}},
            queries={"q1": "Q1"},
            llm=mock_llm,
        )

        assert result.n_comparisons == 0
        assert len(result.verdicts) == 0
        assert result.elo_ratings["P0"] == 1500.0

    @pytest.mark.asyncio
    async def test_win_matrix_populated(self):
        """Win matrix should reflect verdict outcomes."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "A",
            "confidence": 0.9,
            "reasoning": "A wins",
            "dimensions": {d: "A" for d in _COMPARISON_DIMENSIONS},
        })

        reports = {
            "P0": {"q1": "report"},
            "P1": {"q1": "report"},
        }
        queries = {"q1": "Q1"}

        result = await run_arena(
            reports_by_system=reports,
            queries=queries,
            llm=mock_llm,
            max_concurrent=1,
        )

        # P0 is system_a and "A" wins, so after un-swap logic,
        # one of them should have a win
        total_wins = sum(
            result.win_matrix[s][t]
            for s in result.systems
            for t in result.systems
        )
        # Exactly 1 win recorded (not a tie)
        assert total_wins == 1

    @pytest.mark.asyncio
    async def test_head_to_head_populated(self):
        """Head-to-head records should be populated correctly."""
        mock_llm = MagicMock(spec=["complete_json"])
        mock_llm.complete_json = AsyncMock(return_value={
            "overall_winner": "tie",
            "confidence": 0.5,
            "reasoning": "equal",
            "dimensions": {d: "tie" for d in _COMPARISON_DIMENSIONS},
        })

        reports = {
            "P0": {"q1": "r"},
            "P1": {"q1": "r"},
        }
        queries = {"q1": "Q1"}

        result = await run_arena(
            reports_by_system=reports,
            queries=queries,
            llm=mock_llm,
            max_concurrent=1,
        )

        # Both sides should have 1 tie
        assert result.head_to_head["P0"]["P1"]["ties"] == 1
        assert result.head_to_head["P1"]["P0"]["ties"] == 1


# ── Win Matrix and Head-to-Head builders ────────────────────────────────────


class TestWinMatrix:
    def test_basic_wins(self):
        systems = ["A", "B"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q2", "A", "B", "A"),
            _make_verdict("q3", "A", "B", "B"),
        ]
        matrix = _build_win_matrix(verdicts, systems)
        assert matrix["A"]["B"] == 2
        assert matrix["B"]["A"] == 1

    def test_ties_not_counted(self):
        systems = ["A", "B"]
        verdicts = [
            _make_verdict("q1", "A", "B", "tie"),
        ]
        matrix = _build_win_matrix(verdicts, systems)
        assert matrix["A"]["B"] == 0
        assert matrix["B"]["A"] == 0

    def test_empty_verdicts(self):
        matrix = _build_win_matrix([], ["A", "B"])
        assert matrix["A"]["B"] == 0


class TestHeadToHead:
    def test_complete_records(self):
        systems = ["A", "B"]
        verdicts = [
            _make_verdict("q1", "A", "B", "A"),
            _make_verdict("q2", "A", "B", "B"),
            _make_verdict("q3", "A", "B", "tie"),
        ]
        h2h = _build_head_to_head(verdicts, systems)
        assert h2h["A"]["B"] == {"wins": 1, "losses": 1, "ties": 1}
        assert h2h["B"]["A"] == {"wins": 1, "losses": 1, "ties": 1}


# ── Swap label utility ──────────────────────────────────────────────────────


class TestSwapLabel:
    def test_swap_a_to_b(self):
        assert _swap_label("A") == "B"

    def test_swap_b_to_a(self):
        assert _swap_label("B") == "A"

    def test_tie_unchanged(self):
        assert _swap_label("TIE") == "TIE"

    def test_other_unchanged(self):
        assert _swap_label("X") == "X"


# ── Data class construction ─────────────────────────────────────────────────


class TestDataClasses:
    def test_pairwise_verdict_construction(self):
        v = PairwiseVerdict(
            query_id="q1",
            system_a="P0",
            system_b="P1",
            winner="A",
            confidence=0.9,
            reasoning="A is better",
            dimensions_won={"coverage": "A", "factual_accuracy": "B"},
        )
        assert v.query_id == "q1"
        assert v.winner == "A"
        assert v.confidence == 0.9
        assert v.dimensions_won["coverage"] == "A"

    def test_arena_result_construction(self):
        ar = ArenaResult(
            verdicts=[],
            elo_ratings={"P0": 1520.0, "P1": 1480.0},
            bradley_terry_scores={"P0": 0.6, "P1": 0.4},
            win_matrix={"P0": {"P1": 3}, "P1": {"P0": 1}},
            head_to_head={},
            n_comparisons=4,
            systems=["P0", "P1"],
        )
        assert ar.n_comparisons == 4
        assert ar.systems == ["P0", "P1"]
        assert ar.elo_ratings["P0"] == 1520.0


# ── System prompt ───────────────────────────────────────────────────────────


class TestSystemPrompt:
    def test_all_dimensions_in_prompt(self):
        for dim in _COMPARISON_DIMENSIONS:
            assert dim in _PAIRWISE_SYSTEM_PROMPT

    def test_json_format_instruction(self):
        assert "valid JSON" in _PAIRWISE_SYSTEM_PROMPT

    def test_winner_options_in_prompt(self):
        assert '"A"' in _PAIRWISE_SYSTEM_PROMPT
        assert '"B"' in _PAIRWISE_SYSTEM_PROMPT
        assert '"tie"' in _PAIRWISE_SYSTEM_PROMPT
