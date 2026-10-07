"""The small, centralized boundary to the unchanged stage-one implementation."""
from stages.stage1.veriserve.common import (
    atomic_json as atomic_json,
    read_json as read_json,
    stable_hash as stable_hash,
    append_jsonl as append_jsonl,
)


def prm_components():
    from stages.stage1.veriserve.prm import ProcessRewardModel, SYSTEM, STEP_RE, split_steps
    return ProcessRewardModel, SYSTEM, STEP_RE, split_steps


def stage1_feedback(diagnosis, hint, checkpoint_step):
    from stages.stage1.veriserve.model import feedback_text
    return feedback_text(diagnosis, hint, checkpoint_step)
