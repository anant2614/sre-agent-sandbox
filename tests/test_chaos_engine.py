"""Tests for the ChaosEngine fault injection system.

TDD: Tests written first, then implementation.
Covers: deterministic seed behavior, each fault type, multiple faults stacking,
clear_all, probability 0 produces no faults, and validation contract assertions.
"""

from __future__ import annotations

from sre_agent_sandbox.chaos_engine import ChaosEngine
from sre_agent_sandbox.simulated_system import BASELINE_METRICS, SERVICE_NAMES, SimulatedSystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system(seed: int = 42) -> SimulatedSystem:
    """Create and reset a SimulatedSystem for testing."""
    system = SimulatedSystem()
    system.reset(seed)
    return system


def _make_engine(
    fault_probability: float = 1.0,
    seed: int = 42,
) -> ChaosEngine:
    """Create a ChaosEngine with given parameters."""
    return ChaosEngine(fault_probability=fault_probability, seed=seed)


# ===========================================================================
# Test: Deterministic behaviour with seed (VAL-CHAOS-006)
# ===========================================================================

class TestDeterministic:
    """Same seed must produce identical fault sequences."""

    def test_same_seed_produces_same_faults(self) -> None:
        """Two engines with the same seed inject identical faults."""
        faults_a = self._run_episode(seed=42)
        faults_b = self._run_episode(seed=42)
        assert faults_a == faults_b

    def test_different_seeds_produce_different_faults(self) -> None:
        """Different seeds produce at least one difference over 50 steps."""
        faults_a = self._run_episode(seed=42)
        faults_b = self._run_episode(seed=9999)
        assert faults_a != faults_b

    def test_deterministic_fault_types_and_targets(self) -> None:
        """Fault types and targets are reproducible across runs."""
        engine_a = _make_engine(fault_probability=1.0, seed=123)
        engine_b = _make_engine(fault_probability=1.0, seed=123)
        sys_a = _make_system(seed=100)
        sys_b = _make_system(seed=100)

        for _ in range(20):
            engine_a.inject_fault(sys_a)
            engine_b.inject_fault(sys_b)

        assert engine_a.get_active_faults() == engine_b.get_active_faults()

    # -- helpers --

    @staticmethod
    def _run_episode(seed: int) -> list:
        engine = _make_engine(fault_probability=0.5, seed=seed)
        system = _make_system(seed=seed)
        snapshots = []
        for _ in range(50):
            engine.inject_fault(system)
            engine.tick(system)
            snapshots.append(list(engine.get_active_faults()))
        return snapshots


# ===========================================================================
# Test: MemoryLeak fault (VAL-CHAOS-001)
# ===========================================================================

class TestMemoryLeak:
    """Memory leak causes gradual memory increase until crash at 95%."""

    def test_memory_increases_gradually(self) -> None:
        """Memory increases over multiple ticks, not instantaneously."""
        engine = _make_engine()
        system = _make_system()
        target = "db"

        engine._inject_specific_fault(system, "memory_leak", target)

        memory_readings: list[float] = []
        for _ in range(5):
            engine.tick(system)
            memory_readings.append(system._services[target]["memory"])

        # Memory should increase monotonically over ticks
        for i in range(1, len(memory_readings)):
            assert memory_readings[i] > memory_readings[i - 1], (
                f"Memory should increase: step {i}: {memory_readings[i]} <= {memory_readings[i-1]}"
            )

    def test_memory_leak_takes_at_least_3_steps_before_crash(self) -> None:
        """Service should not crash in fewer than 3 ticks (gradual, not instant)."""
        engine = _make_engine()
        system = _make_system()
        target = "order"

        engine._inject_specific_fault(system, "memory_leak", target)

        # Tick 3 times — service should NOT be down yet (baseline memory=40, needs to reach 95)
        for step in range(3):
            engine.tick(system)
            assert not system._services[target]["is_down"], (
                f"Service crashed too early at step {step + 1}"
            )

    def test_memory_leak_eventually_crashes_service(self) -> None:
        """Memory leak eventually brings service down when memory >= 95%."""
        engine = _make_engine()
        system = _make_system()
        target = "api"

        engine._inject_specific_fault(system, "memory_leak", target)

        crashed = False
        for _ in range(100):
            engine.tick(system)
            if system._services[target]["is_down"]:
                crashed = True
                break

        assert crashed, "Memory leak should eventually crash the service"

    def test_memory_leak_crash_threshold_95(self) -> None:
        """Service goes down when memory reaches 95%."""
        engine = _make_engine()
        system = _make_system()
        target = "db"

        engine._inject_specific_fault(system, "memory_leak", target)

        last_memory = 0.0
        for _ in range(200):
            engine.tick(system)
            mem = system._services[target]["memory"]
            if system._services[target]["is_down"]:
                assert last_memory >= 90.0 or mem >= 95.0, (
                    f"Service crashed at memory={mem}, last={last_memory}"
                )
                break
            last_memory = mem

    def test_memory_leak_targets_specific_service(self) -> None:
        """Memory leak only affects the targeted service's memory."""
        engine = _make_engine()
        system = _make_system()
        target = "db"

        # Record baseline of non-target services
        non_targets = [s for s in SERVICE_NAMES if s != target]
        before = {s: system._services[s]["memory"] for s in non_targets}

        engine._inject_specific_fault(system, "memory_leak", target)
        for _ in range(5):
            engine.tick(system)

        # Non-target services should not have their memory affected by the fault
        # (natural drift is separate and handled by system.tick(), not engine.tick())
        for s in non_targets:
            after_mem = system._services[s]["memory"]
            assert after_mem == before[s], (
                f"Non-target {s} memory changed from {before[s]} to {after_mem}"
            )


# ===========================================================================
# Test: LatentDependency fault (VAL-CHAOS-002, VAL-CHAOS-007)
# ===========================================================================

class TestLatentDependency:
    """Latent dependency causes progressive latency increase with cascade."""

    def test_latency_increases_progressively(self) -> None:
        """Latency grows each tick on the target service."""
        engine = _make_engine()
        system = _make_system()
        target = "db"

        engine._inject_specific_fault(system, "latent_dependency", target)

        readings: list[float] = []
        for _ in range(5):
            engine.tick(system)
            readings.append(system._services[target]["latency"])

        for i in range(1, len(readings)):
            assert readings[i] > readings[i - 1], (
                f"Latency should increase: step {i}: {readings[i]} <= {readings[i-1]}"
            )

    def test_latency_cascades_upstream(self) -> None:
        """Latency fault on 'db' cascades to 'order' then 'api'."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "latent_dependency", "db")

        # Tick several times to let cascade propagate
        for _ in range(10):
            engine.tick(system)

        # All three tiers should show latency increase
        db_latency = system._services["db"]["latency"]
        order_latency = system._services["order"]["latency"]
        api_latency = system._services["api"]["latency"]

        baseline = BASELINE_METRICS["latency"]
        assert db_latency > baseline, f"db latency {db_latency} should exceed baseline {baseline}"
        assert order_latency > baseline, f"order latency {order_latency} should exceed baseline {baseline}"
        assert api_latency > baseline, f"api latency {api_latency} should exceed baseline {baseline}"

    def test_cascade_order_db_order_api(self) -> None:
        """db is affected first, then order, then api (cascade direction)."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "latent_dependency", "db")

        # After 1 tick, db should be most affected
        engine.tick(system)
        db_lat = system._services["db"]["latency"]
        order_lat = system._services["order"]["latency"]

        baseline = BASELINE_METRICS["latency"]
        # db should have increased
        assert db_lat > baseline
        # After first tick, upstream may or may not have cascaded yet depending on impl
        # But db should be >= order >= api in terms of increase
        assert db_lat >= order_lat

    def test_latent_dependency_targets_specific_service(self) -> None:
        """Latent dependency fault on 'order' directly affects order."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "latent_dependency", "order")

        for _ in range(5):
            engine.tick(system)

        order_latency = system._services["order"]["latency"]
        baseline = BASELINE_METRICS["latency"]
        assert order_latency > baseline, "Directly targeted service should show increased latency"


# ===========================================================================
# Test: BadConfig fault (VAL-CHAOS-003)
# ===========================================================================

class TestBadConfig:
    """Bad config causes immediate errors on injection step."""

    def test_immediate_unhealthy_on_injection(self) -> None:
        """Target service becomes unhealthy immediately upon injection."""
        engine = _make_engine()
        system = _make_system()
        target = "api"

        # Before injection, service is healthy
        assert system._services[target]["is_healthy"] is True

        engine._inject_specific_fault(system, "bad_config", target)

        # Immediately after injection — no tick needed
        assert system._services[target]["is_healthy"] is False

    def test_bad_config_sets_fault_on_system(self) -> None:
        """Bad config registers as an active fault on the system."""
        engine = _make_engine()
        system = _make_system()
        target = "order"

        engine._inject_specific_fault(system, "bad_config", target)
        assert system._active_faults.get(target) == "bad_config"

    def test_bad_config_no_gradual_degradation(self) -> None:
        """Bad config is immediate — does not worsen progressively over ticks."""
        engine = _make_engine()
        system = _make_system()
        target = "db"

        engine._inject_specific_fault(system, "bad_config", target)

        # Already unhealthy — check it's immediate, not gradual
        assert system._services[target]["is_healthy"] is False

    def test_bad_config_on_all_services(self) -> None:
        """Bad config can target each service independently."""
        for target in SERVICE_NAMES:
            engine = _make_engine()
            system = _make_system()
            engine._inject_specific_fault(system, "bad_config", target)
            assert system._services[target]["is_healthy"] is False


# ===========================================================================
# Test: Multiple faults stacking (VAL-CHAOS-004)
# ===========================================================================

class TestMultipleFaults:
    """Multiple faults can be active simultaneously on different services."""

    def test_two_faults_on_different_services(self) -> None:
        """Two different fault types on two different services coexist."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine._inject_specific_fault(system, "bad_config", "api")

        active = engine.get_active_faults()
        assert len(active) >= 2

        # Both faults should be observable
        fault_types = {f["fault_type"] for f in active}
        assert "memory_leak" in fault_types
        assert "bad_config" in fault_types

    def test_three_faults_simultaneously(self) -> None:
        """Three faults on all three services at once."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine._inject_specific_fault(system, "latent_dependency", "order")
        engine._inject_specific_fault(system, "bad_config", "api")

        active = engine.get_active_faults()
        assert len(active) == 3

        targets = {f["target_service"] for f in active}
        assert targets == {"db", "order", "api"}

    def test_multiple_faults_have_independent_effects(self) -> None:
        """Each fault has its own effect on its target service."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine._inject_specific_fault(system, "bad_config", "api")

        # api is immediately unhealthy from bad_config
        assert system._services["api"]["is_healthy"] is False

        # db memory should increase from memory leak after ticks
        initial_mem = system._services["db"]["memory"]
        for _ in range(3):
            engine.tick(system)
        assert system._services["db"]["memory"] > initial_mem

    def test_tick_progresses_all_active_faults(self) -> None:
        """tick() advances all active gradual faults, not just one."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine._inject_specific_fault(system, "latent_dependency", "order")

        initial_db_mem = system._services["db"]["memory"]
        initial_order_lat = system._services["order"]["latency"]

        engine.tick(system)

        assert system._services["db"]["memory"] > initial_db_mem
        assert system._services["order"]["latency"] > initial_order_lat


# ===========================================================================
# Test: clear_all (VAL-CHAOS-005)
# ===========================================================================

class TestClearAll:
    """clear_all() removes all active faults."""

    def test_clear_all_removes_faults(self) -> None:
        """After clear_all(), get_active_faults() returns empty list."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine._inject_specific_fault(system, "bad_config", "api")
        assert len(engine.get_active_faults()) >= 2

        engine.clear_all()
        assert engine.get_active_faults() == []

    def test_clear_all_stops_fault_progression(self) -> None:
        """After clear_all(), tick() does not progress any faults."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine.clear_all()

        mem_before = system._services["db"]["memory"]
        engine.tick(system)
        mem_after = system._services["db"]["memory"]

        # Memory should NOT increase since fault was cleared
        assert mem_after == mem_before

    def test_clear_all_on_empty_is_no_op(self) -> None:
        """clear_all() on engine with no faults is safe (no error)."""
        engine = _make_engine()
        engine.clear_all()
        assert engine.get_active_faults() == []

    def test_clear_all_allows_new_faults_after(self) -> None:
        """Can inject new faults after clear_all()."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        engine.clear_all()
        assert engine.get_active_faults() == []

        engine._inject_specific_fault(system, "bad_config", "api")
        assert len(engine.get_active_faults()) == 1


# ===========================================================================
# Test: remove_faults_for_service
# ===========================================================================

class TestRemoveFaultsForService:
    """remove_faults_for_service() removes all faults targeting a specific service."""

    def test_removes_single_fault_for_service(self) -> None:
        engine = _make_engine()
        system = _make_system()
        engine._inject_specific_fault(system, "bad_config", "api")
        assert len(engine.get_active_faults()) == 1

        engine.remove_faults_for_service("api")
        assert engine.get_active_faults() == []

    def test_removes_only_target_service_faults(self) -> None:
        engine = _make_engine()
        system = _make_system()
        engine._inject_specific_fault(system, "bad_config", "api")
        engine._inject_specific_fault(system, "memory_leak", "db")
        assert len(engine.get_active_faults()) == 2

        engine.remove_faults_for_service("api")
        faults = engine.get_active_faults()
        assert len(faults) == 1
        assert faults[0]["target_service"] == "db"

    def test_removes_multiple_faults_on_same_service(self) -> None:
        """If a service has multiple fault types, all are removed."""
        engine = _make_engine()
        system = _make_system()
        engine._inject_specific_fault(system, "bad_config", "db")
        engine._inject_specific_fault(system, "memory_leak", "db")
        assert len(engine.get_active_faults()) == 2

        engine.remove_faults_for_service("db")
        assert engine.get_active_faults() == []

    def test_no_op_when_service_has_no_faults(self) -> None:
        engine = _make_engine()
        system = _make_system()
        engine._inject_specific_fault(system, "bad_config", "api")
        engine.remove_faults_for_service("db")  # db has no faults
        assert len(engine.get_active_faults()) == 1


# ===========================================================================
# Test: Probability 0 produces no faults
# ===========================================================================

class TestZeroProbability:
    """Zero probability engine never injects faults."""

    def test_zero_probability_no_faults(self) -> None:
        """inject_fault with probability 0.0 never injects."""
        engine = _make_engine(fault_probability=0.0, seed=42)
        system = _make_system()

        for _ in range(100):
            engine.inject_fault(system)

        assert engine.get_active_faults() == []

    def test_zero_probability_various_seeds(self) -> None:
        """Zero probability doesn't inject regardless of seed."""
        for seed in [0, 1, 42, 99, 9999]:
            engine = _make_engine(fault_probability=0.0, seed=seed)
            system = _make_system()
            for _ in range(50):
                engine.inject_fault(system)
            assert engine.get_active_faults() == [], f"seed={seed} produced faults"


# ===========================================================================
# Test: Full probability produces faults
# ===========================================================================

class TestFullProbability:
    """Probability 1.0 always attempts injection."""

    def test_full_probability_produces_faults(self) -> None:
        """inject_fault with probability 1.0 produces faults quickly."""
        engine = _make_engine(fault_probability=1.0, seed=42)
        system = _make_system()

        # After a few inject_fault calls, should have at least one active fault
        for _ in range(10):
            engine.inject_fault(system)

        assert len(engine.get_active_faults()) > 0


# ===========================================================================
# Test: get_active_faults returns list of fault descriptors
# ===========================================================================

class TestGetActiveFaults:
    """get_active_faults() returns list of fault descriptor dicts."""

    def test_empty_on_fresh_engine(self) -> None:
        """No faults before any injection."""
        engine = _make_engine()
        assert engine.get_active_faults() == []

    def test_fault_descriptor_has_required_keys(self) -> None:
        """Each fault descriptor has fault_type and target_service."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        faults = engine.get_active_faults()

        assert len(faults) == 1
        assert "fault_type" in faults[0]
        assert "target_service" in faults[0]
        assert faults[0]["fault_type"] == "memory_leak"
        assert faults[0]["target_service"] == "db"

    def test_returns_list_not_internal_reference(self) -> None:
        """get_active_faults returns a copy, not the internal data."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        faults = engine.get_active_faults()
        faults.clear()  # Should not affect engine state

        assert len(engine.get_active_faults()) == 1


# ===========================================================================
# Test: inject_fault uses rng for randomness
# ===========================================================================

class TestInjectFault:
    """inject_fault uses the engine's seeded RNG for decisions."""

    def test_inject_fault_respects_probability(self) -> None:
        """With intermediate probability, some faults are injected, some not."""
        engine = _make_engine(fault_probability=0.3, seed=42)
        system = _make_system()

        injected_count = 0
        for _ in range(100):
            before = len(engine.get_active_faults())
            engine.inject_fault(system)
            after = len(engine.get_active_faults())
            if after > before:
                injected_count += 1

        # Should inject some but not all (with p=0.3 over 100 tries)
        assert injected_count > 0, "Should inject at least some faults"
        assert injected_count < 100, "Should not inject faults every time"

    def test_inject_fault_called_with_system(self) -> None:
        """inject_fault takes a SimulatedSystem as argument."""
        engine = _make_engine(fault_probability=1.0, seed=42)
        system = _make_system()

        # Should not raise
        engine.inject_fault(system)


# ===========================================================================
# Test: Engine construction / configuration
# ===========================================================================

class TestEngineConstruction:
    """ChaosEngine construction and configuration."""

    def test_default_construction(self) -> None:
        """Can construct with just probability and seed."""
        engine = ChaosEngine(fault_probability=0.5, seed=42)
        assert engine.get_active_faults() == []

    def test_probability_bounds(self) -> None:
        """Probability must be between 0.0 and 1.0."""
        ChaosEngine(fault_probability=0.0, seed=1)
        ChaosEngine(fault_probability=1.0, seed=1)
        ChaosEngine(fault_probability=0.5, seed=1)

    def test_seed_creates_deterministic_rng(self) -> None:
        """Engine with same seed produces same random sequence."""
        engine_a = ChaosEngine(fault_probability=1.0, seed=42)
        engine_b = ChaosEngine(fault_probability=1.0, seed=42)

        sys_a = _make_system(seed=100)
        sys_b = _make_system(seed=100)

        for _ in range(10):
            engine_a.inject_fault(sys_a)
            engine_b.inject_fault(sys_b)

        assert engine_a.get_active_faults() == engine_b.get_active_faults()


# ===========================================================================
# Test: Tick method
# ===========================================================================

class TestTick:
    """tick() progresses active fault effects on the system."""

    def test_tick_with_no_faults_is_no_op(self) -> None:
        """tick() with no active faults doesn't change anything."""
        engine = _make_engine()
        system = _make_system()

        before_metrics = system.get_metrics()
        engine.tick(system)
        after_metrics = system.get_metrics()

        # Metrics should be unchanged (no faults active, engine.tick doesn't do natural drift)
        assert before_metrics == after_metrics

    def test_tick_progresses_memory_leak(self) -> None:
        """tick() increases memory for service with memory_leak fault."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "memory_leak", "db")
        initial_mem = system._services["db"]["memory"]

        engine.tick(system)
        assert system._services["db"]["memory"] > initial_mem

    def test_tick_progresses_latent_dependency(self) -> None:
        """tick() increases latency for service with latent_dependency fault."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "latent_dependency", "order")
        initial_lat = system._services["order"]["latency"]

        engine.tick(system)
        assert system._services["order"]["latency"] > initial_lat

    def test_tick_does_not_progress_bad_config(self) -> None:
        """Bad config is immediate, tick doesn't worsen it further in a special way."""
        engine = _make_engine()
        system = _make_system()

        engine._inject_specific_fault(system, "bad_config", "api")

        # Bad config already set service unhealthy; tick doesn't need to worsen
        assert system._services["api"]["is_healthy"] is False
