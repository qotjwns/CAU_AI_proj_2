"""컨테이너 CPU 할당량에 맞춰 torch 스레드 수를 정한다.

컨테이너(Vast 등)에서는 nproc·os.cpu_count()가 호스트 코어 수를 보여 주지만 실제 사용량은
cgroup 할당량(cpu.max)으로 제한된다. torch가 호스트 기준으로 스레드를 띄우면 할당량을 넘어
스로틀링이 걸리고 수십 배 느려질 수 있다.
"""
import math
import os
from pathlib import Path

import torch


def cgroup_cpu_limit():
    """cgroup CPU 할당량(코어 수, 소수 가능). 제한이 없거나 읽을 수 없으면 None."""
    v2 = Path("/sys/fs/cgroup/cpu.max")
    try:
        if v2.exists():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def available_cpus():
    """실제로 쓸 수 있는 CPU 수: affinity와 cgroup 할당량 중 작은 값."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    limit = cgroup_cpu_limit()
    if limit is not None:
        count = min(count, max(1, math.floor(limit)))
    return count


def configure_cpu_threads(verbose=True):
    """torch 스레드 수를 사용 가능한 CPU 수 이하로 맞춘다.

    OMP_NUM_THREADS가 명시돼 있으면 사용자 설정으로 보고 건드리지 않는다. 설정한 스레드 수를 반환한다.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return torch.get_num_threads()
    target = available_cpus()
    current = torch.get_num_threads()
    if current > target:
        torch.set_num_threads(target)
        if verbose:
            print(f"torch 스레드 {current} → {target} (컨테이너 CPU 할당량 기준)")
    return torch.get_num_threads()
