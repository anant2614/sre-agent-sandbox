"""Tests for task definitions and graders."""

from __future__ import annotations

import pytest

from demo.run_demo import HeuristicAgent, RandomAgent
from tasks import (
    TASKS,
    TASK_EASY,
    TASK_HARD,
    TASK_MEDIUM,
    _configure_env_for_task,
    evaluate_agent,
    grade,
    make_env,
)


class TestTaskCatalogue:
    def test_three_tasks_registered(self) -> None:
        assert len(TASKS) == 3

    def test_task_ids(self) -> None:
        assert set(TASKS.keys()) == {
            "sre_single_fault",
            "sre_mixed_faults",
            "sre_high_chaos",
        }

    def test_difficulties_cover_all_levels(self) -> None:
        difficulties = {t.difficulty for t in TASKS.values()}
        assert difficulties == {"easy", "medium", "hard"}

    def test_eval_seeds_are_unique_per_task(self) -> None:
        for task in TASKS.values():
            assert len(task.eval_seeds) == len(set(task.eval_seeds))

    def test_eval_seeds_match_n_eval_episodes(self) -> None:
        for task in TASKS.values():
            assert len(task.eval_seeds) == task.n_eval_episodes

    def test_reward_range_ordered(self) -> None:
        for task in TASKS.values():
            worst, best = task.reward_range
            assert worst < best, f"{task.task_id}: worst >= best"


class TestGrader:
    def test_perfect_score(self) -> None:
        _, best = TASK_EASY.reward_range
        score = grade(TASK_EASY, best)
        assert 0.0 < score < 1.0 and score > 0.999

    def test_worst_score(self) -> None:
        worst, _ = TASK_EASY.reward_range
        score = grade(TASK_EASY, worst)
        assert 0.0 < score < 1.0 and score < 0.001

    def test_midpoint(self) -> None:
        worst, best = TASK_EASY.reward_range
        mid = (worst + best) / 2
        assert abs(grade(TASK_EASY, mid) - 0.5) < 1e-9

    def test_clamp_above(self) -> None:
        _, best = TASK_MEDIUM.reward_range
        score = grade(TASK_MEDIUM, best + 1000)
        assert 0.0 < score < 1.0 and score > 0.999

    def test_clamp_below(self) -> None:
        worst, _ = TASK_MEDIUM.reward_range
        score = grade(TASK_MEDIUM, worst - 1000)
        assert 0.0 < score < 1.0 and score < 0.001

    def test_score_in_range(self) -> None:
        for task in TASKS.values():
            worst, best = task.reward_range
            mid = (worst + best) / 2
            score = grade(task, mid)
            assert 0.0 < score < 1.0


class TestMakeEnv:
    def test_easy_env_max_steps(self) -> None:
        env = make_env(TASK_EASY)
        assert env._max_steps == 100

    def test_medium_env_fault_probability(self) -> None:
        env = make_env(TASK_MEDIUM)
        assert env._fault_probability == 0.3

    def test_hard_env_max_steps(self) -> None:
        env = make_env(TASK_HARD)
        assert env._max_steps == 300


class TestConfigureEnvForTask:
    def test_easy_restricts_fault_types(self) -> None:
        env = make_env(TASK_EASY)
        env.reset(seed=0)
        _configure_env_for_task(env, TASK_EASY)
        assert env._chaos._allowed_fault_types == ["bad_config"]

    def test_medium_allows_all_fault_types(self) -> None:
        env = make_env(TASK_MEDIUM)
        env.reset(seed=0)
        _configure_env_for_task(env, TASK_MEDIUM)
        assert env._chaos._allowed_fault_types is None

    def test_hard_lowers_latency_threshold(self) -> None:
        env = make_env(TASK_HARD)
        env.reset(seed=0)
        _configure_env_for_task(env, TASK_HARD)
        assert env._chaos._latency_timeout_threshold == 350.0


class TestEvaluateAgent:
    def test_evaluate_returns_expected_keys(self) -> None:
        agent = RandomAgent(seed=0)
        result = evaluate_agent(TASK_EASY, agent)
        assert set(result.keys()) == {
            "task_id",
            "difficulty",
            "scores",
            "mean_score",
            "rewards",
            "mean_reward",
        }

    def test_evaluate_returns_correct_episode_count(self) -> None:
        agent = RandomAgent(seed=0)
        result = evaluate_agent(TASK_EASY, agent)
        assert len(result["scores"]) == TASK_EASY.n_eval_episodes
        assert len(result["rewards"]) == TASK_EASY.n_eval_episodes

    def test_scores_in_valid_range(self) -> None:
        agent = RandomAgent(seed=0)
        result = evaluate_agent(TASK_EASY, agent)
        for s in result["scores"]:
            assert 0.0 <= s <= 1.0

    def test_heuristic_beats_random_on_easy(self) -> None:
        random_result = evaluate_agent(TASK_EASY, RandomAgent(seed=0))
        heuristic_result = evaluate_agent(TASK_EASY, HeuristicAgent())
        assert heuristic_result["mean_score"] > random_result["mean_score"]

    def test_heuristic_beats_random_on_medium(self) -> None:
        random_result = evaluate_agent(TASK_MEDIUM, RandomAgent(seed=0))
        heuristic_result = evaluate_agent(TASK_MEDIUM, HeuristicAgent())
        assert heuristic_result["mean_score"] > random_result["mean_score"]

    def test_reproducible_scores(self) -> None:
        agent = HeuristicAgent()
        r1 = evaluate_agent(TASK_EASY, agent)
        r2 = evaluate_agent(TASK_EASY, agent)
        assert r1["scores"] == r2["scores"]
        assert r1["rewards"] == r2["rewards"]
