"""failure_mapper.py

Deterministic Failure Attribution Engine for VGCL Phase 0.

Constructs execution-based failure clusters from empirical base model evaluation records
without subjective LLM-judge bias or semantic error inference.
"""

import enum
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class FailureCategory(str, enum.Enum):
    BOUNDS_VIOLATION = "BOUNDS_VIOLATION"
    TERMINATION_FAILURE = "TERMINATION_FAILURE"
    CORRECTNESS_FAILURE_MICRO = "CORRECTNESS_FAILURE_MICRO"
    SCALE_DIVERGENCE = "SCALE_DIVERGENCE"
    RESOURCE_FAILURE = "RESOURCE_FAILURE"
    FORMAT_CONTRACT_FAILURE = "FORMAT_CONTRACT_FAILURE"
    STATE_INCONSISTENCY = "STATE_INCONSISTENCY"
    UNKNOWN_DETERMINISTIC_FAILURE = "UNKNOWN_DETERMINISTIC_FAILURE"


# Stable cluster identifiers for experimental intervention units
CLUSTER_MAP: Dict[FailureCategory, str] = {
    FailureCategory.CORRECTNESS_FAILURE_MICRO: "FC_01_MICRO_CORRECTNESS",
    FailureCategory.SCALE_DIVERGENCE: "FC_02_SCALE_DIVERGENCE",
    FailureCategory.TERMINATION_FAILURE: "FC_03_TERMINATION",
    FailureCategory.BOUNDS_VIOLATION: "FC_04_BOUNDS_VIOLATION",
    FailureCategory.RESOURCE_FAILURE: "FC_05_RESOURCE_FAILURE",
    FailureCategory.FORMAT_CONTRACT_FAILURE: "FC_06_FORMAT_CONTRACT",
    FailureCategory.STATE_INCONSISTENCY: "FC_07_STATE_INCONSISTENCY",
    FailureCategory.UNKNOWN_DETERMINISTIC_FAILURE: "FC_08_UNKNOWN",
}


@dataclass
class FailureRecord:
    problem_id: str
    prompt: str
    generated_code: str
    execution_result: str
    failing_input: Optional[str]
    expected_output: Optional[str]
    actual_output: Optional[str]
    exception_type: str
    execution_time: float
    memory_usage: float
    execution_trace_statistics: Dict[str, Any]
    seed: int
    model_version: str
    prompt_hash: str
    failure_category: FailureCategory
    cluster_id: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["failure_category"] = self.failure_category.value
        return res


def compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def classify_failure(
    exception_type: str,
    error_msg: str,
    execution_time: float,
    memory_usage_mb: float,
    is_small_scale: bool = True,
    is_large_scale: bool = False,
    small_passed: bool = False,
    large_failed: bool = False,
    state_mutation_detected: bool = False,
) -> Tuple[FailureCategory, str]:
    """Deterministically categorizes failure based strictly on observable execution behavior."""
    error_upper = error_msg.upper()
    exc_upper = exception_type.upper()

    if state_mutation_detected or "STATE MUTATION" in error_upper:
        cat = FailureCategory.STATE_INCONSISTENCY
    elif "MEMORY LIMIT" in error_upper or "MEMORYERROR" in exc_upper or memory_usage_mb > 512.0:
        cat = FailureCategory.RESOURCE_FAILURE
    elif "TIMEOUT" in error_upper or "TIMED OUT" in error_upper or execution_time > 4.0 or "RECURSIONERROR" in exc_upper:
        cat = FailureCategory.TERMINATION_FAILURE
    elif "INDEXERROR" in exc_upper or "OUT OF BOUNDS" in error_upper or "KEYERROR" in exc_upper:
        cat = FailureCategory.BOUNDS_VIOLATION
    elif "TYPEERROR" in exc_upper or "VALUEERROR" in exc_upper or "SYNTAXERROR" in exc_upper or "FORMAT" in error_upper:
        cat = FailureCategory.FORMAT_CONTRACT_FAILURE
    elif small_passed and (large_failed or is_large_scale or "SCALE" in error_upper):
        cat = FailureCategory.SCALE_DIVERGENCE
    elif "MISMATCH" in error_upper or "PHASE A" in error_upper or is_small_scale:
        cat = FailureCategory.CORRECTNESS_FAILURE_MICRO
    else:
        cat = FailureCategory.UNKNOWN_DETERMINISTIC_FAILURE

    cluster_id = CLUSTER_MAP[cat]
    return cat, cluster_id


class FailureMapper:
    """Attribution engine that runs probing models against problem suites and maps failures to stable clusters."""

    def __init__(self, model_version: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_version = model_version
        self.records: List[FailureRecord] = []

    def record_failure(
        self,
        problem_id: str,
        prompt: str,
        generated_code: str,
        execution_result: str,
        failing_input: Optional[str] = None,
        expected_output: Optional[str] = None,
        actual_output: Optional[str] = None,
        exception_type: str = "ExecutionFailure",
        execution_time: float = 0.0,
        memory_usage: float = 0.0,
        execution_trace_statistics: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        is_small_scale: bool = True,
        is_large_scale: bool = False,
        small_passed: bool = False,
        large_failed: bool = False,
        state_mutation_detected: bool = False,
    ) -> FailureRecord:
        prompt_hash = compute_prompt_hash(prompt)
        trace_stats = execution_trace_statistics or {}

        cat, cluster_id = classify_failure(
            exception_type=exception_type,
            error_msg=execution_result,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            is_small_scale=is_small_scale,
            is_large_scale=is_large_scale,
            small_passed=small_passed,
            large_failed=large_failed,
            state_mutation_detected=state_mutation_detected,
        )

        record = FailureRecord(
            problem_id=problem_id,
            prompt=prompt,
            generated_code=generated_code,
            execution_result=execution_result,
            failing_input=failing_input,
            expected_output=expected_output,
            actual_output=actual_output,
            exception_type=exception_type,
            execution_time=execution_time,
            memory_usage=memory_usage,
            execution_trace_statistics=trace_stats,
            seed=seed,
            model_version=self.model_version,
            prompt_hash=prompt_hash,
            failure_category=cat,
            cluster_id=cluster_id,
        )
        self.records.append(record)
        return record

    def save_records(self, filepath: str = "data/failure_records.json") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = [r.to_dict() for r in self.records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_cluster_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for r in self.records:
            summary[r.cluster_id] = summary.get(r.cluster_id, 0) + 1
        return summary

    def verify_no_evaluation_leakage(self, eval_problem_ids: List[str]) -> bool:
        """Enforces training_problem_ids ∩ evaluation_problem_ids == empty (Leakage Guard)."""
        train_ids = {r.problem_id for r in self.records}
        eval_ids = set(eval_problem_ids)
        intersection = train_ids.intersection(eval_ids)
        if intersection:
            raise ValueError(f"CRITICAL EVALUATION LEAKAGE DETECTED: Overlapping problem IDs: {intersection}")
        return True


if __name__ == "__main__":
    mapper = FailureMapper()
    rec = mapper.record_failure(
        problem_id="PROB_001",
        prompt="Solve sliding window maximum",
        generated_code="def solve(nums, k): return []",
        execution_result="Phase A Mismatch: Optimal=[] vs Brute=[3, 3, 5]",
        failing_input="[1,3,-1,-3,5,3,6,7], 3",
        expected_output="[3, 3, 5, 5, 6, 7]",
        actual_output="[]",
        exception_type="MismatchError",
        execution_time=0.05,
        memory_usage=12.4,
    )
    print("Recorded failure:", rec.cluster_id, rec.failure_category.value)
    print("Summary:", mapper.get_cluster_summary())
