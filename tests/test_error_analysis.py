"""Comprehensive tests for the error analysis module.

Tests cover:
- ErrorInstance construction
- ErrorProfile construction and counts
- aggregate_error_profiles groups by pattern
- aggregate_error_profiles computes correct distributions
- generate_error_report produces valid markdown
- categorize_errors with mock data
"""

from __future__ import annotations

import pytest

from deep_research.evaluation.error_analysis import (
    ErrorInstance,
    ErrorProfile,
    PatternErrorProfile,
    aggregate_error_profiles,
    categorize_errors,
    generate_error_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_errors():
    """A list of sample ErrorInstance objects."""
    return [
        ErrorInstance(
            category="hallucination",
            severity="critical",
            description="Report claims a study that does not exist",
            section="Introduction",
            evidence="Smith et al. (2024) is not findable",
        ),
        ErrorInstance(
            category="citation_fabrication",
            severity="critical",
            description="Citation [3] points to non-existent URL",
            section="References",
            evidence="[3] https://example.com/nonexistent",
        ),
        ErrorInstance(
            category="missing_coverage",
            severity="moderate",
            description="Report does not discuss economic implications",
            section="Analysis",
            evidence="No section on economics",
        ),
        ErrorInstance(
            category="synthesis_failure",
            severity="minor",
            description="Conclusion repeats introduction verbatim",
            section="Conclusion",
        ),
    ]


@pytest.fixture
def sample_profile(sample_errors):
    """An ErrorProfile built from sample errors."""
    return ErrorProfile(
        pattern="p4_perspective_storm",
        query_id="q1",
        errors=sample_errors,
        total_errors=len(sample_errors),
        by_category={
            "hallucination": 1,
            "citation_fabrication": 1,
            "missing_coverage": 1,
            "synthesis_failure": 1,
        },
        by_severity={"critical": 2, "moderate": 1, "minor": 1},
        critical_count=2,
    )


@pytest.fixture
def multi_pattern_profiles():
    """Multiple ErrorProfiles across different patterns."""
    p4_errors = [
        ErrorInstance(category="hallucination", severity="critical",
                     description="Fake study", section="Intro"),
        ErrorInstance(category="hallucination", severity="critical",
                     description="Another fake", section="Body"),
        ErrorInstance(category="missing_coverage", severity="moderate",
                     description="Gap", section="Analysis"),
    ]
    p4_profile = ErrorProfile(
        pattern="p4_perspective_storm", query_id="q1",
        errors=p4_errors, total_errors=3,
        by_category={"hallucination": 2, "missing_coverage": 1},
        by_severity={"critical": 2, "moderate": 1},
        critical_count=2,
    )

    p3_errors = [
        ErrorInstance(category="synthesis_failure", severity="minor",
                     description="Poor structure", section="Body"),
        ErrorInstance(category="synthesis_failure", severity="minor",
                     description="Repetitive", section="Conclusion"),
    ]
    p3_profile = ErrorProfile(
        pattern="p3_meridian", query_id="q1",
        errors=p3_errors, total_errors=2,
        by_category={"synthesis_failure": 2},
        by_severity={"minor": 2},
        critical_count=0,
    )

    p4_errors_2 = [
        ErrorInstance(category="hallucination", severity="critical",
                     description="Yet another", section="Intro"),
    ]
    p4_profile_2 = ErrorProfile(
        pattern="p4_perspective_storm", query_id="q2",
        errors=p4_errors_2, total_errors=1,
        by_category={"hallucination": 1},
        by_severity={"critical": 1},
        critical_count=1,
    )

    return [p4_profile, p3_profile, p4_profile_2]


# ---------------------------------------------------------------------------
# 1. ErrorInstance construction
# ---------------------------------------------------------------------------

class TestErrorInstance:
    def test_construction(self):
        err = ErrorInstance(
            category="hallucination",
            severity="critical",
            description="Made up a source",
            section="Introduction",
            evidence="No such paper exists",
        )
        assert err.category == "hallucination"
        assert err.severity == "critical"
        assert err.section == "Introduction"

    def test_default_evidence(self):
        err = ErrorInstance(
            category="factual_error",
            severity="moderate",
            description="Wrong date",
            section="Timeline",
        )
        assert err.evidence == ""

    def test_all_categories(self):
        """Verify all expected categories can be instantiated."""
        categories = [
            "hallucination", "citation_fabrication", "topic_drift",
            "factual_error", "missing_coverage", "synthesis_failure",
            "source_quality", "attribution_error",
        ]
        for cat in categories:
            err = ErrorInstance(category=cat, severity="minor",
                              description="test", section="test")
            assert err.category == cat


# ---------------------------------------------------------------------------
# 2. ErrorProfile construction and counts
# ---------------------------------------------------------------------------

class TestErrorProfile:
    def test_construction(self, sample_profile):
        assert sample_profile.pattern == "p4_perspective_storm"
        assert sample_profile.query_id == "q1"
        assert sample_profile.total_errors == 4
        assert sample_profile.critical_count == 2

    def test_by_category_sums(self, sample_profile):
        total = sum(sample_profile.by_category.values())
        assert total == sample_profile.total_errors

    def test_by_severity_sums(self, sample_profile):
        total = sum(sample_profile.by_severity.values())
        assert total == sample_profile.total_errors

    def test_empty_profile(self):
        profile = ErrorProfile(
            pattern="p0_baseline", query_id="q_empty",
            errors=[], total_errors=0,
            by_category={}, by_severity={}, critical_count=0,
        )
        assert profile.total_errors == 0
        assert profile.critical_count == 0


# ---------------------------------------------------------------------------
# 3. aggregate_error_profiles groups by pattern
# ---------------------------------------------------------------------------

class TestAggregateGrouping:
    def test_groups_by_pattern(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        assert "p4_perspective_storm" in result
        assert "p3_meridian" in result
        assert len(result) == 2

    def test_correct_report_counts(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        assert result["p4_perspective_storm"].n_reports == 2
        assert result["p3_meridian"].n_reports == 1


# ---------------------------------------------------------------------------
# 4. aggregate_error_profiles computes correct distributions
# ---------------------------------------------------------------------------

class TestAggregateDistributions:
    def test_category_distribution(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        p4 = result["p4_perspective_storm"]
        # P4 has 3 hallucination + 1 missing_coverage = 4 total
        assert p4.category_distribution["hallucination"] == pytest.approx(3 / 4)
        assert p4.category_distribution["missing_coverage"] == pytest.approx(1 / 4)

    def test_severity_distribution(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        p3 = result["p3_meridian"]
        # P3 has 2 minor errors
        assert p3.severity_distribution["minor"] == pytest.approx(1.0)

    def test_avg_errors_per_report(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        p4 = result["p4_perspective_storm"]
        # 4 total errors across 2 reports = 2.0 avg
        assert p4.avg_errors_per_report == pytest.approx(2.0)

    def test_most_common_errors(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        p4 = result["p4_perspective_storm"]
        # hallucination should be most common
        assert p4.most_common_errors[0][0] == "hallucination"
        assert p4.most_common_errors[0][1] == 3

    def test_failure_modes_populated(self, multi_pattern_profiles):
        result = aggregate_error_profiles(multi_pattern_profiles)
        for pattern, profile in result.items():
            assert len(profile.failure_modes) > 0

    def test_empty_profiles(self):
        result = aggregate_error_profiles([])
        assert result == {}


# ---------------------------------------------------------------------------
# 5. generate_error_report produces valid markdown
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_has_header(self, multi_pattern_profiles):
        profiles = aggregate_error_profiles(multi_pattern_profiles)
        report = generate_error_report(profiles)
        assert "# Error Analysis Report" in report

    def test_report_has_summary_table(self, multi_pattern_profiles):
        profiles = aggregate_error_profiles(multi_pattern_profiles)
        report = generate_error_report(profiles)
        assert "## Summary" in report
        assert "| Pattern |" in report

    def test_report_has_per_pattern_sections(self, multi_pattern_profiles):
        profiles = aggregate_error_profiles(multi_pattern_profiles)
        report = generate_error_report(profiles)
        assert "## p4_perspective_storm" in report
        assert "## p3_meridian" in report

    def test_report_has_category_table(self, multi_pattern_profiles):
        profiles = aggregate_error_profiles(multi_pattern_profiles)
        report = generate_error_report(profiles)
        assert "Error Categories" in report
        assert "| Category |" in report

    def test_report_has_failure_modes(self, multi_pattern_profiles):
        profiles = aggregate_error_profiles(multi_pattern_profiles)
        report = generate_error_report(profiles)
        assert "Failure Modes" in report

    def test_empty_report(self):
        report = generate_error_report({})
        assert "# Error Analysis Report" in report


# ---------------------------------------------------------------------------
# 6. categorize_errors with mock data
# ---------------------------------------------------------------------------

class TestCategorizeErrors:
    @pytest.mark.asyncio
    async def test_from_judge_verdicts(self):
        report_text = "# Test Report\n\n## Introduction\nSome content.\n\n## Analysis\nMore content."
        verdicts = [
            {"dimension": "factual_accuracy", "verdict": "NOT_SATISFIED",
             "reasoning": "Contains unsupported claims"},
            {"dimension": "coverage", "verdict": "SATISFIED",
             "reasoning": "Good coverage"},
            {"dimension": "citation_quality", "verdict": "NOT_SATISFIED",
             "reasoning": "Citations are fabricated"},
        ]

        profile = await categorize_errors(
            report_text=report_text,
            pattern="p4_perspective_storm",
            query_id="q_test",
            judge_verdicts=verdicts,
        )

        assert profile.pattern == "p4_perspective_storm"
        assert profile.query_id == "q_test"
        # Only NOT_SATISFIED verdicts become errors
        assert profile.total_errors >= 2
        # Should have factual_error and citation_fabrication
        categories = [e.category for e in profile.errors]
        assert "factual_error" in categories
        assert "citation_fabrication" in categories

    @pytest.mark.asyncio
    async def test_from_citation_verification(self):
        report_text = "# Report\n\n## Body\nText [1] and [2].\n\n## References\n[1] Source"
        citation_verification = {
            "flagged_claims": ["Claim about X is not supported by [2]"],
            "accuracy_rate": 0.4,
        }

        profile = await categorize_errors(
            report_text=report_text,
            pattern="p3_meridian",
            query_id="q_cite",
            citation_verification=citation_verification,
        )

        assert profile.total_errors >= 1
        categories = [e.category for e in profile.errors]
        assert "citation_fabrication" in categories

    @pytest.mark.asyncio
    async def test_heuristic_short_report(self):
        """A very short report should trigger a synthesis_failure error."""
        report_text = "# Short\n\nToo short."

        profile = await categorize_errors(
            report_text=report_text,
            pattern="p0_baseline",
            query_id="q_short",
        )

        categories = [e.category for e in profile.errors]
        assert "synthesis_failure" in categories

    @pytest.mark.asyncio
    async def test_no_errors_on_clean_report(self):
        """A normal-length report with no judge issues should have minimal errors."""
        report_text = "# Good Report\n\n## Abstract\n" + ("Good content here now. " * 200) + \
                      "\n\n## References\n[1] Real Source"

        profile = await categorize_errors(
            report_text=report_text,
            pattern="p4_perspective_storm",
            query_id="q_clean",
        )

        # Should have no or very few errors (maybe dangling refs)
        assert profile.critical_count == 0

    @pytest.mark.asyncio
    async def test_empty_verdicts(self):
        report_text = "# Report\n\n" + ("Content. " * 200)
        profile = await categorize_errors(
            report_text=report_text,
            pattern="p1_iterative_rag",
            query_id="q_empty_verdicts",
            judge_verdicts=[],
        )
        assert profile.pattern == "p1_iterative_rag"

    @pytest.mark.asyncio
    async def test_dimension_mapping(self):
        """Verify dimension-to-category mapping for all known dimensions."""
        dimensions_expected = {
            "factual_accuracy": "factual_error",
            "citation_quality": "citation_fabrication",
            "coverage": "missing_coverage",
            "analytical_depth": "synthesis_failure",
            "organisation": "synthesis_failure",
            "instruction_following": "topic_drift",
        }
        report_text = "# Report\n\n" + ("Content. " * 200)

        for dim, expected_cat in dimensions_expected.items():
            verdicts = [
                {"dimension": dim, "verdict": "NOT_SATISFIED", "reasoning": "Test"}
            ]
            profile = await categorize_errors(
                report_text=report_text,
                pattern="p0_baseline",
                query_id=f"q_{dim}",
                judge_verdicts=verdicts,
            )
            categories = [e.category for e in profile.errors]
            assert expected_cat in categories, f"Dimension {dim} should map to {expected_cat}"
