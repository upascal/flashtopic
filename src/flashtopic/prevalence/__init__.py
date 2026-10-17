"""Debiased document-level topic prevalence (PPI / ACC) with valid CIs.

Pipeline: build a RunContext from the clustering → define Topics → cheap per-document
surrogate (membership + cosine score) → stratified gold sample → LLM/human judge →
debiased prevalence estimate. The estimator is pure-numpy and backend-free.
"""

from __future__ import annotations

from flashtopic.prevalence.estimate import PrevalenceResult, estimate_prevalence
from flashtopic.prevalence.data import RunContext
from flashtopic.prevalence.topics import Topic, topics_from_labels
from flashtopic.prevalence.prototypes import (
    TopicSurrogate,
    build_surrogate,
    candidate_false_negatives,
)
from flashtopic.prevalence.sampling import SampleDesign, design_sample
from flashtopic.prevalence.codebook import Codebook, build_codebook, render
from flashtopic.prevalence.judge import Verdict, judge_design, run_judge, verdicts_to_y
from flashtopic.prevalence.classifier import (
    Answer,
    DocClassifier,
    JevClassifier,
    MockClassifier,
    Question,
    classify_docs,
    run_classify,
)
from flashtopic.prevalence.questions import QuestionSet, TopicContext, build_question_set
from flashtopic.prevalence.features import FeatureMatrix, assemble
from flashtopic.prevalence.surrogate import FittedSurrogate, choose_surrogate, fit_cross_fitted
from flashtopic.prevalence.pilot import PilotResult, run_pilot, run_pilot_sync, write_report

__all__ = [
    "PrevalenceResult",
    "estimate_prevalence",
    "RunContext",
    "Topic",
    "topics_from_labels",
    "TopicSurrogate",
    "build_surrogate",
    "candidate_false_negatives",
    "SampleDesign",
    "design_sample",
    "Codebook",
    "build_codebook",
    "render",
    "Verdict",
    "judge_design",
    "run_judge",
    "verdicts_to_y",
    "Answer",
    "DocClassifier",
    "JevClassifier",
    "MockClassifier",
    "Question",
    "classify_docs",
    "run_classify",
    "QuestionSet",
    "TopicContext",
    "build_question_set",
    "FeatureMatrix",
    "assemble",
    "FittedSurrogate",
    "choose_surrogate",
    "fit_cross_fitted",
    "PilotResult",
    "run_pilot",
    "run_pilot_sync",
    "write_report",
]
