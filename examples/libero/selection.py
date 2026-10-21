"""Explicit, reproducible task/state selection for reduced LIBERO diagnostics."""


def _validate(ids, total, label):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{label} must be nonempty and unique")
    if any(not 0 <= index < total for index in ids):
        raise ValueError(f"{label} must be in [0, {total})")
    return list(ids)


def select_tasks(total, task_ids=None, max_tasks=None):
    if task_ids is not None and max_tasks is not None:
        raise ValueError("Use task_ids or max_tasks, not both")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    ids = list(range(min(total, max_tasks))) if max_tasks is not None else list(range(total))
    return _validate(ids if task_ids is None else task_ids, total, "task_ids")


def select_initial_states(total, trials, state_ids=None):
    ids = list(range(trials)) if state_ids is None else state_ids
    if trials <= 0 or len(ids) != trials:
        raise ValueError("initial_state_ids must contain one state per trial")
    return _validate(ids, total, "initial_state_ids")
