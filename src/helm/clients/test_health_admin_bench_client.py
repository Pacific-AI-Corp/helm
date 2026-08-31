import json
import threading
import time

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, ADAPT_HEALTH_ADMIN_BENCH
from helm.benchmark.runner import execute_parallelism_for_run
from helm.benchmark.scenarios.health_admin_bench_constants import HAB_HARNESS_DEPLOYMENT, HAB_PROTOCOL
from helm.clients.health_admin_bench_client import HealthAdminBenchClient
from helm.common.request import Request


def test_execute_parallelism_for_hab_is_clamped_to_one():
    assert execute_parallelism_for_run(ADAPT_HEALTH_ADMIN_BENCH, 4) == 1
    assert execute_parallelism_for_run(ADAPT_HEALTH_ADMIN_BENCH, 1) == 1
    assert execute_parallelism_for_run(ADAPT_GENERATION, 4) == 4


def test_hab_client_serializes_concurrent_episodes(monkeypatch):
    client = HealthAdminBenchClient()
    current = 0
    max_seen = 0
    lock = threading.Lock()
    start_together = threading.Barrier(2)

    def fake_run(self, envelope):
        nonlocal current, max_seen
        with lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.08)
        with lock:
            current -= 1
        return {
            "task_id": envelope.get("task_id"),
            "passed": False,
            "score": 0.0,
            "max_points": 0.0,
            "percentage": 0.0,
            "steps": 0,
            "eval_results": [],
        }

    monkeypatch.setattr(HealthAdminBenchClient, "_run_episode", fake_run)
    prompt = json.dumps({"hab_protocol": HAB_PROTOCOL, "task_id": "emr-easy-1"})
    request = Request(prompt=prompt, model_deployment=HAB_HARNESS_DEPLOYMENT)

    results: list = []

    def worker():
        start_together.wait(timeout=5)
        results.append(client.make_request(request))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert max_seen == 1
    assert len(results) == 2
    assert all(result.success for result in results)
