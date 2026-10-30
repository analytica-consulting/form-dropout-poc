"""Metrics vs fill ground truth — uplift over naive (no form removal)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from form_dropout_poc.render import is_fragile_field_value


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    m, n = len(reference), len(hypothesis)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n] / float(m)


@dataclass
class FieldResult:
    field_id: str
    expected: str
    naive: str
    naive_subtract: str
    edge_dropout: str
    fragile: bool = False

    @property
    def naive_ok(self) -> bool:
        return self.naive == self.expected

    @property
    def edge_ok(self) -> bool:
        return self.edge_dropout == self.expected

    @property
    def uplift(self) -> bool:
        return (not self.naive_ok) and self.edge_ok

    @property
    def subtract_ok(self) -> bool:
        return self.naive_subtract == self.expected


@dataclass
class SampleResult:
    sample_id: str
    font_id: str
    ink_mode: str
    font_detected: str
    font_correct: bool
    reg_reproj_mean_px: float
    mean_overlap: float
    fields: list[FieldResult] = field(default_factory=list)


def _aggregate_fields(fields: list[FieldResult]) -> dict:
    n = len(fields) or 1
    naive_fail = [f for f in fields if not f.naive_ok]
    nf = len(naive_fail) or 1
    uplift = sum(1 for f in fields if f.uplift)
    rescue = sum(1 for f in naive_fail if f.edge_ok)
    return {
        "field_count": len(fields),
        "exact_naive": sum(1 for f in fields if f.naive_ok) / n,
        "exact_edge": sum(1 for f in fields if f.edge_ok) / n,
        "exact_subtract": sum(1 for f in fields if f.subtract_ok) / n,
        "mean_cer_naive": sum(character_error_rate(f.expected, f.naive) for f in fields) / n,
        "mean_cer_edge": sum(character_error_rate(f.expected, f.edge_dropout) for f in fields) / n,
        "mean_cer_subtract": sum(
            character_error_rate(f.expected, f.naive_subtract) for f in fields
        )
        / n,
        "uplift_fields": uplift,
        "uplift_rate": uplift / n,
        "rescue_rate_on_naive_fail": rescue / nf,
        "edge_beats_subtract_on_rescues": sum(
            1 for f in naive_fail if f.edge_ok and not f.subtract_ok
        ),
    }


def aggregate_results(samples: list[SampleResult]) -> dict:
    if not samples:
        return {}
    all_fields = [f for s in samples for f in s.fields]
    fragile_fields = [f for f in all_fields if f.fragile]
    agg = _aggregate_fields(all_fields)
    agg["total_samples"] = len(samples)
    agg["font_detection_accuracy"] = sum(1 for s in samples if s.font_correct) / len(samples)
    agg["mean_reg_reproj_px"] = sum(s.reg_reproj_mean_px for s in samples) / len(samples)
    agg["cer_delta_naive_minus_edge"] = agg["mean_cer_naive"] - agg["mean_cer_edge"]
    if fragile_fields:
        frag = _aggregate_fields(
            [
                FieldResult(
                    f.field_id,
                    f.expected,
                    f.naive,
                    f.naive_subtract,
                    f.edge_dropout,
                    fragile=True,
                )
                for f in fragile_fields
            ]
        )
        agg["fragile"] = {
            "uplift_fields": frag["uplift_fields"],
            "uplift_rate": frag["uplift_rate"],
            "rescue_rate_on_naive_fail": frag["rescue_rate_on_naive_fail"],
            "mean_cer_naive": frag["mean_cer_naive"],
            "mean_cer_edge": frag["mean_cer_edge"],
        }

    by_ink: dict[str, list] = {}
    for s in samples:
        by_ink.setdefault(s.ink_mode, []).extend(s.fields)
    agg["by_ink_mode"] = {k: _aggregate_fields(v) for k, v in by_ink.items()}

    overlaps = [s.mean_overlap for s in samples]
    if overlaps:
        q33 = float(np.percentile(overlaps, 33))
        q66 = float(np.percentile(overlaps, 66))
        by_overlap: dict[str, list] = {"low": [], "mid": [], "high": []}
        for s in samples:
            if s.mean_overlap <= q33:
                key = "low"
            elif s.mean_overlap <= q66:
                key = "mid"
            else:
                key = "high"
            by_overlap[key].extend(s.fields)
        agg["by_overlap_bucket"] = {k: _aggregate_fields(v) for k, v in by_overlap.items() if v}
        agg["overlap_bucket_thresholds"] = {"p33": q33, "p66": q66}

    reg_errors = [s.reg_reproj_mean_px for s in samples]
    if len(reg_errors) >= 4:
        qs = [float(np.percentile(reg_errors, p)) for p in (25, 50, 75)]
        by_reg: dict[str, list] = {"q1": [], "q2": [], "q3": [], "q4": []}
        for s in samples:
            r = s.reg_reproj_mean_px
            if r <= qs[0]:
                key = "q1"
            elif r <= qs[1]:
                key = "q2"
            elif r <= qs[2]:
                key = "q3"
            else:
                key = "q4"
            by_reg[key].extend(s.fields)
        agg["by_reg_error_quartile"] = {k: _aggregate_fields(v) for k, v in by_reg.items() if v}
        agg["reg_error_quartile_px"] = {"p25": qs[0], "p50": qs[1], "p75": qs[2]}

    agg["per_sample"] = []
    for s in samples:
        sub = _aggregate_fields(s.fields)
        sub["sample_id"] = s.sample_id
        sub["ink_mode"] = s.ink_mode
        sub["font_correct"] = s.font_correct
        sub["reg_reproj_mean_px"] = s.reg_reproj_mean_px
        sub["mean_overlap"] = s.mean_overlap
        agg["per_sample"].append(sub)
    return agg
