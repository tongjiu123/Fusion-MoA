import unittest

from fusion_moa import MoAFramework, Request


class FakeClient:
    def __init__(self, name, response="", error=None):
        self.name = name
        self.response = response
        self.error = error
        self.calls = []

    async def complete(self, messages):
        self.calls.append(tuple(messages))
        if self.error is not None:
            raise self.error
        return self.response


class MoAFrameworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_path_does_not_call_analysts(self):
        executor = FakeClient("executor", "direct")
        analyst = FakeClient("analyst", "unused")
        framework = MoAFramework(
            executor=executor,
            analysts=[analyst],
            admission=lambda request: False,
        )

        result = await framework.run(
            Request.from_messages([{"role": "user", "content": "question"}])
        )

        self.assertEqual(result.output, "direct")
        self.assertFalse(result.used_collective)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(analyst.calls, [])

    async def test_collective_path_ignores_failed_analyst(self):
        executor = FakeClient("executor", "final")
        useful = FakeClient("useful", "candidate evidence")
        failed = FakeClient("failed", error=RuntimeError("unavailable"))
        framework = MoAFramework(executor=executor, analysts=[useful, failed])

        result = await framework.run(
            Request.from_messages([{"role": "user", "content": "question"}])
        )

        self.assertEqual(result.output, "final")
        self.assertTrue(result.used_collective)
        self.assertEqual(result.evidence_count, 1)
        self.assertEqual(len(executor.calls), 1)
        self.assertIn("candidate evidence", executor.calls[0][-1]["content"])

    async def test_selector_bounds_forwarded_evidence(self):
        executor = FakeClient("executor", "final")
        analysts = [
            FakeClient("first", "one"),
            FakeClient("second", "two"),
        ]
        framework = MoAFramework(
            executor=executor,
            analysts=analysts,
            selector=lambda packets: packets[:1],
        )

        result = await framework.run(
            Request.from_messages([{"role": "user", "content": "question"}])
        )

        self.assertEqual(result.evidence_count, 1)
        self.assertIn("one", executor.calls[0][-1]["content"])
        self.assertNotIn("two", executor.calls[0][-1]["content"])


if __name__ == "__main__":
    unittest.main()
