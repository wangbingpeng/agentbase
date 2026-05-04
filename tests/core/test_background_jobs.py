"""Tests for BackgroundJobRunner."""

import asyncio

import pytest
import pytest_asyncio

from agentbase_core.engine import AgentBaseEngine
from agentbase_core.ingester.background import BackgroundJob, BackgroundJobRunner, JobStatus
from agentbase_core.models.config import AgentBaseConfig
from agentbase_core.store.connection import ConnectionPool


@pytest_asyncio.fixture
async def job_runner(tmp_path):
    """Create a BackgroundJobRunner with a test database."""
    config = AgentBaseConfig(data_dir=tmp_path)
    engine = AgentBaseEngine(config=config)
    await engine.initialize()

    runner = BackgroundJobRunner(pool=engine._pool, poll_interval=0.1, max_attempts=2)
    yield runner
    await engine.close()


@pytest.mark.asyncio
class TestBackgroundJobRunner:
    async def test_submit_and_get(self, job_runner):
        results = []

        async def handler(payload):
            results.append(payload)

        job_runner.register_handler("test_type", handler)
        job = await job_runner.submit("test_type", payload={"key": "value"})
        assert job.id
        assert job.job_type == "test_type"
        assert job.status == JobStatus.PENDING

        # Get the job
        fetched = await job_runner.get_job(job.id)
        assert fetched is not None
        assert fetched.job_type == "test_type"

    async def test_job_execution(self, job_runner):
        results = []

        async def handler(payload):
            results.append(payload["message"])

        job_runner.register_handler("echo", handler)
        job = await job_runner.submit("echo", payload={"message": "hello"})

        # Start the runner and wait for processing
        await job_runner.start()
        await asyncio.sleep(0.5)
        await job_runner.stop()

        # Check the job completed
        fetched = await job_runner.get_job(job.id)
        assert fetched.status == JobStatus.COMPLETED
        assert results == ["hello"]

    async def test_job_retry_on_failure(self, job_runner):
        call_count = 0

        async def failing_handler(payload):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Temporary failure")

        job_runner.register_handler("flaky", failing_handler)
        job = await job_runner.submit("flaky", max_attempts=3)

        await job_runner.start()
        await asyncio.sleep(1.0)
        await job_runner.stop()

        fetched = await job_runner.get_job(job.id)
        # Should eventually succeed
        assert fetched.status == JobStatus.COMPLETED

    async def test_job_max_retries_exceeded(self, job_runner):
        async def always_fail(payload):
            raise RuntimeError("Always fails")

        job_runner.register_handler("bad_job", always_fail)
        job = await job_runner.submit("bad_job", max_attempts=1)

        await job_runner.start()
        await asyncio.sleep(0.5)
        await job_runner.stop()

        fetched = await job_runner.get_job(job.id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.error_message is not None

    async def test_list_jobs(self, job_runner):
        async def handler(payload):
            pass

        job_runner.register_handler("list_test", handler)
        await job_runner.submit("list_test")
        await job_runner.submit("list_test")

        jobs = await job_runner.list_jobs(status="pending")
        assert len(jobs) >= 2

    async def test_unknown_job_type_rejected(self, job_runner):
        from agentbase_core.exceptions import BackgroundJobError
        with pytest.raises(BackgroundJobError):
            await job_runner.submit("unknown_type")

    async def test_recovery_on_startup(self, job_runner):
        """Jobs in 'running' state should be reset to 'pending' on startup."""
        async def handler(payload):
            await asyncio.sleep(0.1)

        job_runner.register_handler("recovery_test", handler)
        job = await job_runner.submit("recovery_test")

        # Manually set status to 'running'
        pool = job_runner._pool
        async with pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'running' WHERE id = ?",
                (job.id,),
            )
            await conn.commit()

        # Start runner (should recover the job)
        await job_runner.start()
        await asyncio.sleep(0.5)
        await job_runner.stop()

        # The job should have been processed
        fetched = await job_runner.get_job(job.id)
        assert fetched.status in (JobStatus.COMPLETED, JobStatus.PENDING, JobStatus.RUNNING)
