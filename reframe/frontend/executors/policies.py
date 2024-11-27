# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import contextlib
import math
import sys
import time

import reframe.core.runtime as rt
import reframe.utility as util
from reframe.core.exceptions import (FailureLimitError,
                                     RunSessionTimeout,
                                     SkipTestError,
                                     TaskDependencyError,
                                     TaskExit,
                                     ForceExitError,
                                     AbortTaskError)
from reframe.core.logging import getlogger, level_from_str
from reframe.core.pipeline import (CompileOnlyRegressionTest,
                                   RunOnlyRegressionTest)
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)


def _get_partition_name(task, phase='run'):
    if (task.check.local or
        (phase == 'build' and task.check.build_locally)):
        return '_rfm_local'
    else:
        return task.check.current_partition.fullname


def _cleanup_all(tasks, *args, **kwargs):
    for task in tasks:
        if task.ref_count == 0:
            with contextlib.suppress(TaskExit):
                task.cleanup(*args, **kwargs)

    # Remove cleaned up tests
    tasks[:] = [t for t in tasks if t.ref_count]


def _print_perf(task):
    '''Get performance info of the current task.'''

    perfvars = task.testcase.check.perfvalues
    level = level_from_str(
        rt.runtime().get_option('general/0/perf_info_level')
    )
    for key, info in perfvars.items():
        name = key.split(':')[-1]
        getlogger().log(level,
                        f'P: {name}: {info[0]} {info[4]} '
                        f'(r:{info[1]}, l:{info[2]}, u:{info[3]})')


class _PollController:
    SLEEP_MIN = 0.1
    SLEEP_MAX = 10
    SLEEP_INC_RATE = 1.1

    def __init__(self):
        self._num_polls = 0
        self._sleep_duration = None
        self._t_init = None

    def reset_snooze_time(self):
        self._sleep_duration = self.SLEEP_MIN

    async def snooze(self):
        if self._num_polls == 0:
            self._t_init = time.time()

        t_elapsed = time.time() - self._t_init
        self._num_polls += 1
        poll_rate = self._num_polls / t_elapsed if t_elapsed else math.inf
        getlogger().debug2(
            f'Poll rate control: sleeping for {self._sleep_duration}s '
            f'(current poll rate: {poll_rate} polls/s)'
        )
        await asyncio.sleep(self._sleep_duration)
        self._sleep_duration = min(
            self._sleep_duration*self.SLEEP_INC_RATE, self.SLEEP_MAX
        )


class SerialExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks that have finished, but have not performed their cleanup phase
        self._retired_tasks = []
        self.task_listeners.append(self)

    def _runcase(self, case):
        super()._runcase(case)
        check, partition, _ = case
        task = RegressionTask(case, self.task_listeners)
        if check.is_dry_run():
            self.printer.status('DRY', task.info())
        else:
            self.printer.status('RUN', task.info())

        self._task_index[case] = task
        self.stats.add_task(task)
        try:
            # Do not run test if any of its dependencies has failed
            # NOTE: Restored dependencies are not in the task_index
            if any(self._task_index[c].failed
                   for c in case.deps if c in self._task_index):
                raise TaskDependencyError('dependencies failed')

            if any(self._task_index[c].skipped
                   for c in case.deps if c in self._task_index):

                # We raise the SkipTestError here and catch it immediately in
                # order for `skip()` to get the correct exception context.
                try:
                    raise SkipTestError('skipped due to skipped dependencies')
                except SkipTestError as e:
                    task.skip()
                    raise TaskExit from e
            task.setup(task.testcase.partition,
                       task.testcase.environ,
                       sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                       sched_options=self.sched_options)

            async def compile_async_serial():
                await task.compile()
                await task.compile_wait()

            async def run_async_serial():
                await task.run()

                sched = self.local_scheduler

                self._pollctl.reset_snooze_time()
                while True:
                    if not self.dry_run_mode:
                        await sched.poll(task.check.job)
                    if task.run_complete():
                        break

                    await self._pollctl.snooze()

                await task.run_wait()

            if task.check.is_local:
                # TODO: ssh scheduler
                asyncio.run(compile_async_serial())
                asyncio.run(run_async_serial())
            else:
                asyncio.run(compile_async_serial())
                asyncio.run(task.run())

                sched = partition.scheduler

                self._pollctl.reset_snooze_time()
                while True:
                    if not self.dry_run_mode:
                        asyncio.run(sched.poll(task.check.job))
                    if task.run_complete():
                        break

                    asyncio.run(self._pollctl.snooze())

                asyncio.run(task.run_wait())

            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            self._retired_tasks.append(task)
            task.finalize()
        except TaskExit:
            return
        except ABORT_REASONS as e:
            task.abort(e)
            if type(e) is KeyboardInterrupt:
                raise ForceExitError
            else:
                raise e
        except BaseException:
            task.fail(sys.exc_info())

    def on_task_setup(self, task):
        pass

    def on_task_run(self, task):
        pass

    def on_task_compile(self, task):
        pass

    def on_task_exit(self, task):
        pass

    def on_task_compile_exit(self, task):
        pass

    def on_task_skip(self, task):
        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')

    def on_task_abort(self, task):
        msg = f'{task.info()}'
        self.printer.status('ABORT', msg, just='right')

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {task.check.stagedir!r}')
        getlogger().verbose(f'==> {timings}')
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def on_task_success(self, task):
        msg = f'{task.info()}'
        self.printer.status('OK', msg, just='right')
        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().verbose(f'==> {timings}')

        # Update reference count of dependencies
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1

        _cleanup_all(self._retired_tasks, not self.keep_stage_files)
        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def _exit(self):
        # Clean up all remaining tasks
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)


class AsyncioExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()
        self._current_tasks = util.OrderedSet()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks per partition
        self._partition_tasks = {
            '_rfm_local': util.OrderedSet()
        }

        # Job limit per partition
        self._max_jobs = {
            '_rfm_local': rt.runtime().get_option('systems/0/max_local_jobs')
        }

        # Tasks that have finished, but have not performed their cleanup phase
        self._retired_tasks = []
        self.task_listeners.append(self)

    async def _runcase(self, case, task):
        # I added the task here as an argument because, I wanted to initialize it
        # outside, when I gather the tasks. If I gather the tasks and then I do asyncio
        # manage them, if one of them fails the others are not iformed, I had to code that
        # manually. There is a way to make everything stop if an exepction is raised but
        # I didn't know how to treat that raise Exception nicelly because I wouldn't be able
        # to abort the tasks which the execution has not yet started, I needed to do abortall
        # on all the tests, not only the ones which were initiated by the execution. Exit gracefully
        # the execuion loop aborting all the tasks
        super()._runcase(case)
        check, partition, _ = case
        # task = RegressionTask(case, self.task_listeners)
        if check.is_dry_run():
            self.printer.status('DRY', task.info())
        else:
            self.printer.status('RUN', task.info())

        self._partition_tasks.setdefault(partition.fullname, util.OrderedSet())
        self._max_jobs.setdefault(partition.fullname, partition.max_jobs)

        self._task_index[case] = task
        try:
            # Do not run test if any of its dependencies has failed
            # NOTE: Restored dependencies are not in the task_index
            if any(self._task_index[c].failed
                   for c in case.deps if c in self._task_index):
                raise TaskDependencyError('dependencies failed')

            if any(self._task_index[c].skipped
                   for c in case.deps if c in self._task_index):

                # We raise the SkipTestError here and catch it immediately in
                # order for `skip()` to get the correct exception context.
                try:
                    raise SkipTestError('skipped due to skipped dependencies')
                except SkipTestError as e:
                    task.skip()
                    raise TaskExit from e

            deps_status = await self.check_deps(task)
            if deps_status == "skipped":
                try:
                    raise SkipTestError('skipped due to skipped dependencies')
                except SkipTestError:
                    task.skip()
                    self._current_tasks.remove(task)
                    return 1
            elif deps_status == "succeded":
                if task.check.is_dry_run():
                    self.printer.status('DRY', task.info())
                else:
                    self.printer.status('RUN', task.info())
            elif deps_status == "failed":
                exc = TaskDependencyError('dependencies failed')
                task.fail((type(exc), exc, None))
                self._current_tasks.remove(task)
                return 1

            task.setup(task.testcase.partition,
                       task.testcase.environ,
                       sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                       sched_options=self.sched_options)
            partname = _get_partition_name(task, phase='build')
            max_jobs = self._max_jobs[partname]
            while len(self._partition_tasks[partname]) > max_jobs:
                await asyncio.sleep(2)
            await task.compile()
            self._partition_tasks[partname].add(task)
            await task.compile_wait()
            while len(self._partition_tasks[partname]) > max_jobs:
                await asyncio.sleep(2)
            await task.run()

            # Pick the right scheduler
            if task.check.local:
                sched = self.local_scheduler
            else:
                sched = partition.scheduler

            self._pollctl.reset_snooze_time()
            while True:
                if not self.dry_run_mode:
                    await sched.poll(task.check.job)

                if task.run_complete():
                    break

                await self._pollctl.snooze()

            await task.run_wait()
            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            self._retired_tasks.append(task)
            task.finalize()
            # The execution is not as controlled as in the pseudo asynchronous
            # we need to block those tasks until all the dependencies finish
        except TaskExit:
            self._current_tasks.remove(task)
            if task.check.current_partition:
                partname = task.check.current_partition.fullname
            else:
                partname = None

            # Remove tasks from the partition tasks if there
            with contextlib.suppress(KeyError):
                self._partition_tasks['_rfm_local'].remove(task)
                if partname:
                    self._partition_tasks[partname].remove(task)
            return
        except ABORT_REASONS as e:
            self._abortall(e)
            if type(e) is KeyboardInterrupt:
                raise ForceExitError
            else:
                raise e
        except BaseException:
            task.fail(sys.exc_info())
            self._current_tasks.remove(task)
            if task.check.current_partition:
                partname = task.check.current_partition.fullname
            else:
                partname = None

            # Remove tasks from the partition tasks if there
            with contextlib.suppress(KeyError):
                self._partition_tasks['_rfm_local'].remove(task)
                if partname:
                    self._partition_tasks[partname].remove(task)
            return

        self._current_tasks.remove(task)
        if task.check.current_partition:
            partname = task.check.current_partition.fullname
        else:
            partname = None

        # Remove tasks from the partition tasks if there
        with contextlib.suppress(KeyError):
            self._partition_tasks['_rfm_local'].remove(task)
            if partname:
                self._partition_tasks[partname].remove(task)

    async def check_deps(self, task):
        while not (self.deps_skipped(task) or self.deps_failed(task) or
                   self.deps_succeeded(task)):
            await asyncio.sleep(1)

        if self.deps_skipped(task):
            return "skipped"
        elif self.deps_failed(task):
            return "failed"
        elif self.deps_succeeded(task):
            return "succeeded"

    def deps_failed(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return any(self._task_index[c].failed
                   for c in task.testcase.deps if c in self._task_index)

    def deps_succeeded(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return all(self._task_index[c].succeeded
                   for c in task.testcase.deps if c in self._task_index)

    def deps_skipped(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return any(self._task_index[c].skipped
                   for c in task.testcase.deps if c in self._task_index)

    def _abortall(self, cause):
        '''Mark all tests as failures'''

        getlogger().debug2(f'Aborting all tasks due to {type(cause).__name__}')
        for task in self._current_tasks:
            with contextlib.suppress(FailureLimitError):
                task.abort(cause)

    def on_task_setup(self, task):
        pass

    def on_task_run(self, task):
        pass

    def on_task_compile(self, task):
        pass

    def on_task_exit(self, task):
        pass

    def on_task_compile_exit(self, task):
        pass

    def on_task_skip(self, task):
        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')

    def on_task_abort(self, task):
        msg = f'{task.info()}'
        self.printer.status('ABORT', msg, just='right')

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {task.check.stagedir!r}')
        getlogger().verbose(f'==> {timings}')
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def on_task_success(self, task):
        msg = f'{task.info()}'
        self.printer.status('OK', msg, just='right')
        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                        'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().verbose(f'==> {timings}')

        # Update reference count of dependencies
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1

        _cleanup_all(self._retired_tasks, not self.keep_stage_files)
        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def exit(self):
        # Clean up all remaining tasks
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)

    def execute(self, testcases):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        all_cases = []
        for t in testcases:
            task = RegressionTask(t, self.task_listeners)

            self.stats.add_task(task)
            self._current_tasks.add(task)
            all_cases.append(asyncio.ensure_future(self._runcase(t, task)))
        try:
            # Wait for tasks until the first failure
            loop.run_until_complete(self._execute_until_failure(all_cases))
        except (Exception, KeyboardInterrupt) as e:
            if type(e) in (ABORT_REASONS):
                loop.run_until_complete(self._cancel_gracefully(all_cases))
                try:
                    raise AbortTaskError
                except AbortTaskError as exc:
                    self._abortall(exc)
                raise e
            else:
                getlogger().info(f"Execution stopped due to an error: {e}")
        finally:
            loop.close()
        loop.close()
        self.exit()

    async def _execute_until_failure(self, all_cases):
        """Wait for tasks to complete or fail, stopping at the first failure."""
        while all_cases:
            done, all_cases = await asyncio.wait(
                all_cases, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    raise task.exception()  # Exit if aborted

    async def _cancel_gracefully(self, all_cases):
        for case in all_cases:
            case.cancel()
        await asyncio.gather(*all_cases, return_exceptions=True)
