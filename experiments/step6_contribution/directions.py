"""STEP 6 후속 실험의 다섯 비교 조건과 P 부분공간 기저: 수학적 정의와 인터페이스만 둔다.

이번 단계(6-1)에서는 실제 기저 학습, 차원 선택, 조건 비교를 하지 않는다. 합성 데이터로 성질만 시험한다.
벡터는 o_proj 이후 공간(hidden 차원 d)에 있다. U는 [d, k] 정규직교 기저다.
  Full N          c_N
  Parallel        U Uᵀ c_N
  Orthogonal      (I − U Uᵀ) c_N
  Random          c_N과 노름이 같은 무작위 방향 (벡터마다, 고정 seed 생성기)
  Mean Difference μ_N − μ_P (크기를 c_N에 맞출지는 사전 등록 때 정한다: scale="raw" 또는 "match")
"""
import torch

from src.core.feature_validation import require


def check_orthonormal(basis, tolerance=1e-6):
    gram = basis.T.double() @ basis.double()
    require(torch.allclose(gram, torch.eye(basis.shape[1], dtype=torch.float64), atol=tolerance),
            "The basis is not orthonormal")


def full(c_n):
    return c_n


def parallel(c_n, basis):
    return (c_n @ basis) @ basis.T


def orthogonal(c_n, basis):
    return c_n - parallel(c_n, basis)


def random_same_norm(c_n, generator):
    """c_N 각 벡터와 노름이 같은 등방 무작위 방향 (float64로 뽑아 c_N dtype으로)."""
    direction = torch.randn(c_n.shape, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return (direction * c_n.double().norm(dim=-1, keepdim=True)).to(c_n.dtype)


def mean_difference(mu_n, mu_p, c_n=None, scale="raw"):
    difference = mu_n - mu_p
    if scale == "raw":
        return difference
    require(scale == "match" and c_n is not None, "scale must be 'raw' or 'match' (with c_N)")
    norm = difference.double().norm(dim=-1, keepdim=True)
    return (difference.double() / norm * c_n.double().norm(dim=-1, keepdim=True)).to(c_n.dtype)


CONDITIONS = ("full", "parallel", "orthogonal", "random", "mean_difference")


def condition_vector(name, c_n, basis=None, generator=None, mu_n=None, mu_p=None, mean_scale="raw"):
    """다섯 조건의 개입 벡터 (OutputAddition의 v). c_N: [..., d]."""
    if name == "full":
        return full(c_n)
    if name in ("parallel", "orthogonal"):
        require(basis is not None, "A P-subspace basis is required")
        return parallel(c_n, basis) if name == "parallel" else orthogonal(c_n, basis)
    if name == "random":
        require(generator is not None, "A seeded generator is required")
        return random_same_norm(c_n, generator)
    if name == "mean_difference":
        return mean_difference(mu_n, mu_p, c_n, mean_scale).expand_as(c_n)
    raise ValueError(f"Unknown condition {name}")


def fit_basis(vectors, rule, centered):
    """P 기여 벡터 [n, d]로 정규직교 기저 U [d, k]를 만든다 (SVD).

    rule: ("variance", 0.90 | 0.95) = 누적 설명 비율이 처음으로 기준 이상이 되는 가장 작은 k, ("rank", k) = 고정 k.
    centered=False면 원점 기준(2차 모멘트) 분해라서 선형 사영 U Uᵀ c와 맞는다.
    centered=True면 평균을 뺀 공분산 분해이고, 평균 방향은 기저에 자동으로 들어가지 않는다 (설계 문서 참고).
    """
    data = vectors.double()
    mean = data.mean(dim=0) if centered else torch.zeros(data.shape[1], dtype=torch.float64)
    _, singular, right = torch.linalg.svd(data - mean, full_matrices=False)
    energy = singular ** 2
    share = torch.cumsum(energy, dim=0) / energy.sum()
    kind, value = rule
    if kind == "variance":
        require(0 < value <= 1, "The variance threshold must be in (0, 1]")
        k = int((share < value - 1e-12).sum().item()) + 1
    elif kind == "rank":
        require(1 <= int(value) <= len(singular), "Invalid fixed rank")
        k = int(value)
    else:
        raise ValueError(f"Unknown rule {rule}")
    basis = right[:k].T.contiguous()
    return {"basis": basis, "k": k, "explained": float(share[k - 1]), "mean": mean, "singular_values": singular,
            "centered": centered, "rule": list(rule)}
