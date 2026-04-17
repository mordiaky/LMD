"""Triton CUDA kernels for LMD operations.

These kernels provide GPU-accelerated implementations of:
- Pairwise cosine similarity
- Memory coupling computation
- Density estimation
- Void probing
- Fused memory stepping
"""


import torch
import triton
import triton.language as tl

# ============================================================================
# Pairwise Cosine Similarity Kernel
# ============================================================================

@triton.jit
def _cosine_sim_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute cosine similarity between rows of A and rows of B.

    Out[i,j] = cos_sim(A[i], B[j]) = (A[i] @ B[j]) / (||A[i]|| * ||B[j]||)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block starting indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators
    dot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    norm_a = tl.zeros((BLOCK_M,), dtype=tl.float32)
    norm_b = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=a_mask,
            other=0.0
        )

        # Load B block
        b_mask = (offs_n[:, None] < N) & ((k + offs_k[None, :]) < K)
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + (k + offs_k[None, :]) * stride_bk,
            mask=b_mask,
            other=0.0
        )

        # Accumulate dot product: dot += A @ B.T
        dot += tl.dot(a, tl.trans(b))

        # Accumulate norms
        norm_a += tl.sum(a * a, axis=1)
        norm_b += tl.sum(b * b, axis=1)

    # Compute cosine similarity
    norm_a = tl.sqrt(norm_a + 1e-8)
    norm_b = tl.sqrt(norm_b + 1e-8)

    cos_sim = dot / (norm_a[:, None] * norm_b[None, :])

    # Store result
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        cos_sim,
        mask=out_mask
    )


def batch_cosine_similarity(A: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Compute pairwise cosine similarity between embeddings.

    Args:
        A: (M, K) tensor of embeddings
        B: (N, K) tensor of embeddings (if None, B=A for self-similarity)

    Returns:
        (M, N) tensor of cosine similarities
    """
    if B is None:
        B = A

    assert A.is_cuda, "Input must be on CUDA device"
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape[1] == B.shape[1]

    M, K = A.shape
    N = B.shape[0]

    # Output tensor
    Out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(64, K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _cosine_sim_kernel[grid](
        A, B, Out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return Out


# ============================================================================
# Coupling Computation Kernel
# ============================================================================

@triton.jit
def _coupling_kernel(
    Emb_ptr, Energy_ptr, Valence_ptr, Coupling_ptr,
    N, K,
    stride_en, stride_ek,
    coupling_strength,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute pairwise memory coupling.

    Coupling[i,j] = strength * cos_sim(Emb[i], Emb[j]) * Energy[i] * Energy[j] * valence_resonance
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # Skip lower triangle (symmetric matrix)
    if pid_j < pid_i:
        return

    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize
    dot = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    norm_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    norm_j = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Compute cosine similarity
    for k in range(0, K, BLOCK_K):
        mask_i = (offs_i[:, None] < N) & ((k + offs_k[None, :]) < K)
        emb_i = tl.load(
            Emb_ptr + offs_i[:, None] * stride_en + (k + offs_k[None, :]) * stride_ek,
            mask=mask_i, other=0.0
        )

        mask_j = (offs_j[:, None] < N) & ((k + offs_k[None, :]) < K)
        emb_j = tl.load(
            Emb_ptr + offs_j[:, None] * stride_en + (k + offs_k[None, :]) * stride_ek,
            mask=mask_j, other=0.0
        )

        dot += tl.dot(emb_i, tl.trans(emb_j))
        norm_i += tl.sum(emb_i * emb_i, axis=1)
        norm_j += tl.sum(emb_j * emb_j, axis=1)

    cos_sim = dot / (tl.sqrt(norm_i[:, None] + 1e-8) * tl.sqrt(norm_j[None, :] + 1e-8))

    # Load energies
    energy_i = tl.load(Energy_ptr + offs_i, mask=offs_i < N, other=0.0)
    energy_j = tl.load(Energy_ptr + offs_j, mask=offs_j < N, other=0.0)

    # Load valences
    valence_i = tl.load(Valence_ptr + offs_i, mask=offs_i < N, other=0.0)
    valence_j = tl.load(Valence_ptr + offs_j, mask=offs_j < N, other=0.0)

    # Valence resonance: similar emotions couple more strongly
    valence_diff = tl.abs(valence_i[:, None] - valence_j[None, :])
    valence_resonance = 1.0 - 0.5 * valence_diff

    # Compute coupling
    coupling = coupling_strength * cos_sim * energy_i[:, None] * energy_j[None, :] * valence_resonance

    # Store (symmetric, so store both i,j and j,i)
    mask_out = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(
        Coupling_ptr + offs_i[:, None] * N + offs_j[None, :],
        coupling,
        mask=mask_out
    )

    # Store transpose for lower triangle
    if pid_i != pid_j:
        tl.store(
            Coupling_ptr + offs_j[None, :] * N + offs_i[:, None],
            tl.trans(coupling),
            mask=(offs_j[None, :] < N) & (offs_i[:, None] < N)
        )


def batch_coupling(
    embeddings: torch.Tensor,
    energies: torch.Tensor,
    valences: torch.Tensor,
    coupling_strength: float = 0.1
) -> torch.Tensor:
    """Compute pairwise memory coupling matrix.

    Args:
        embeddings: (N, K) tensor of memory embeddings
        energies: (N,) tensor of memory energies
        valences: (N,) tensor of memory valences
        coupling_strength: Base coupling strength

    Returns:
        (N, N) coupling matrix
    """
    assert embeddings.is_cuda
    N, K = embeddings.shape

    Coupling = torch.zeros((N, N), device=embeddings.device, dtype=torch.float32)

    BLOCK_N = 16
    BLOCK_K = min(64, K)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(N, BLOCK_N))

    _coupling_kernel[grid](
        embeddings, energies, valences, Coupling,
        N, K,
        embeddings.stride(0), embeddings.stride(1),
        coupling_strength,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return Coupling


# ============================================================================
# Density Estimation Kernel
# ============================================================================

@triton.jit
def _density_kernel(
    Query_ptr, Points_ptr, Density_ptr,
    N_queries, N_points, K,
    stride_qn, stride_qk,
    stride_pn, stride_pk,
    bandwidth,
    BLOCK_Q: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Estimate density at query points using Gaussian kernel.

    density[i] = sum_j exp(-||query[i] - point[j]||^2 / (2 * bandwidth^2))
    """
    pid = tl.program_id(0)

    offs_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)

    # Accumulate density
    density = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for p_start in range(0, N_points, BLOCK_P):
        offs_p = p_start + tl.arange(0, BLOCK_P)

        # Compute squared distance
        sq_dist = tl.zeros((BLOCK_Q, BLOCK_P), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            # Load query block
            q_mask = (offs_q[:, None] < N_queries) & (offs_k[None, :] < K)
            q = tl.load(
                Query_ptr + offs_q[:, None] * stride_qn + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0
            )

            # Load points block
            p_mask = (offs_p[:, None] < N_points) & (offs_k[None, :] < K)
            p = tl.load(
                Points_ptr + offs_p[:, None] * stride_pn + offs_k[None, :] * stride_pk,
                mask=p_mask, other=0.0
            )

            # Accumulate (q - p)^2
            diff = q[:, None, :] - p[None, :, :]  # (BLOCK_Q, BLOCK_P, BLOCK_K)
            sq_dist += tl.sum(diff * diff, axis=2)

        # Gaussian kernel
        kernel = tl.exp(-sq_dist / (2.0 * bandwidth * bandwidth))

        # Mask invalid points
        valid_mask = offs_p[None, :] < N_points
        kernel = tl.where(valid_mask, kernel, 0.0)

        # Accumulate
        density += tl.sum(kernel, axis=1)

    # Normalize
    density = density / N_points

    # Store
    tl.store(
        Density_ptr + offs_q,
        density,
        mask=offs_q < N_queries
    )


def density_estimation(
    queries: torch.Tensor,
    points: torch.Tensor,
    bandwidth: float = 0.5
) -> torch.Tensor:
    """Estimate density at query points.

    Args:
        queries: (N_q, K) query points
        points: (N_p, K) reference points
        bandwidth: Gaussian kernel bandwidth

    Returns:
        (N_q,) density at each query point
    """
    assert queries.is_cuda
    N_q, K = queries.shape
    N_p = points.shape[0]

    density = torch.empty(N_q, device=queries.device, dtype=torch.float32)

    BLOCK_Q = 32
    BLOCK_P = 64
    BLOCK_K = min(32, K)

    grid = (triton.cdiv(N_q, BLOCK_Q),)

    _density_kernel[grid](
        queries, points, density,
        N_q, N_p, K,
        queries.stride(0), queries.stride(1),
        points.stride(0), points.stride(1),
        bandwidth,
        BLOCK_Q=BLOCK_Q,
        BLOCK_P=BLOCK_P,
        BLOCK_K=BLOCK_K,
    )

    return density


# ============================================================================
# Pairwise Distance Kernel
# ============================================================================

@triton.jit
def _pairwise_dist_kernel(
    A_ptr, B_ptr, Dist_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_dm, stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute pairwise L2 distances."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    sq_dist = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0
        )

        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=b_mask, other=0.0
        )

        diff = a[:, None, :] - b[None, :, :]
        sq_dist += tl.sum(diff * diff, axis=2)

    dist = tl.sqrt(sq_dist + 1e-8)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Dist_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn,
        dist,
        mask=out_mask
    )


def pairwise_distances(A: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Compute pairwise L2 distances.

    Args:
        A: (M, K) tensor
        B: (N, K) tensor (if None, B=A)

    Returns:
        (M, N) distance matrix
    """
    if B is None:
        B = A

    assert A.is_cuda
    M, K = A.shape
    N = B.shape[0]

    Dist = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(32, K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _pairwise_dist_kernel[grid](
        A, B, Dist,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Dist.stride(0), Dist.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return Dist


# ============================================================================
# Void Probe Density Kernel
# ============================================================================

def void_probe_density(
    probes: torch.Tensor,
    known_points: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """Compute void density for probes (lower = more void-like).

    Uses k-nearest neighbors distance as density proxy.

    Args:
        probes: (N_p, K) probe points
        known_points: (N_k, K) known memory embeddings
        k: Number of nearest neighbors

    Returns:
        (N_p,) density scores (lower = more void-like)
    """
    # Compute distances
    distances = pairwise_distances(probes, known_points)

    # Get k-nearest distances
    k = min(k, distances.shape[1])
    knn_dists, _ = distances.topk(k, dim=1, largest=False)

    # Mean kNN distance as density proxy (inverted)
    density = 1.0 / (knn_dists.mean(dim=1) + 0.1)

    return density


# ============================================================================
# Fused Memory Step Kernel
# ============================================================================

@triton.jit
def _memory_step_kernel(
    Emb_ptr, Energy_ptr, Phase_ptr,
    Coupling_ptr, Grad_ptr,
    N, K,
    dt, noise_scale, energy_decay,
    stride_en, stride_ek,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused memory evolution step.

    Updates:
    - Embedding: emb += dt * (coupling_force + noise)
    - Energy: energy -= dt * decay * (1 - activity)
    - Phase: phase += dt * phase_velocity
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load energy
    energy = tl.load(Energy_ptr + offs_n, mask=offs_n < N, other=0.0)
    phase = tl.load(Phase_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Compute coupling force from gradient
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        # Load embedding
        emb = tl.load(
            Emb_ptr + offs_n[:, None] * stride_en + offs_k[None, :] * stride_ek,
            mask=mask, other=0.0
        )

        # Load gradient (from coupling)
        grad = tl.load(
            Grad_ptr + offs_n[:, None] * stride_en + offs_k[None, :] * stride_ek,
            mask=mask, other=0.0
        )

        # Add noise
        # Note: Triton doesn't have built-in random, using deterministic pattern
        noise = tl.sin(emb * 12.9898 + phase[:, None] * 78.233) * noise_scale

        # Update embedding
        new_emb = emb + dt * (grad + noise)

        # Normalize (per-row)
        norm = tl.sqrt(tl.sum(new_emb * new_emb, axis=1, keep_dims=True) + 1e-8)
        new_emb = new_emb / norm

        tl.store(
            Emb_ptr + offs_n[:, None] * stride_en + offs_k[None, :] * stride_ek,
            new_emb,
            mask=mask
        )

    # Update energy (decay based on low activity)
    activity = tl.abs(tl.load(Grad_ptr + offs_n * K, mask=offs_n < N, other=0.0))
    new_energy = energy - dt * energy_decay * (1.0 - activity)
    new_energy = tl.maximum(new_energy, 0.0)  # Clamp to non-negative
    tl.store(Energy_ptr + offs_n, new_energy, mask=offs_n < N)

    # Update phase
    phase_velocity = 0.1 * (1.0 + energy)  # Higher energy = faster phase
    new_phase = phase + dt * phase_velocity
    tl.store(Phase_ptr + offs_n, new_phase, mask=offs_n < N)


def memory_step_fused(
    embeddings: torch.Tensor,
    energies: torch.Tensor,
    phases: torch.Tensor,
    coupling_grad: torch.Tensor,
    dt: float = 0.01,
    noise_scale: float = 0.01,
    energy_decay: float = 0.01
) -> None:
    """Fused memory evolution step (in-place).

    Updates embeddings, energies, and phases in a single kernel launch.

    Args:
        embeddings: (N, K) memory embeddings (modified in-place)
        energies: (N,) memory energies (modified in-place)
        phases: (N,) narrative phases (modified in-place)
        coupling_grad: (N, K) gradient from coupling field
        dt: Time step
        noise_scale: Noise magnitude
        energy_decay: Energy decay rate
    """
    assert embeddings.is_cuda
    N, K = embeddings.shape

    BLOCK_N = 32
    BLOCK_K = min(64, K)

    grid = (triton.cdiv(N, BLOCK_N),)

    _memory_step_kernel[grid](
        embeddings, energies, phases,
        None, coupling_grad,  # Coupling computed separately
        N, K,
        dt, noise_scale, energy_decay,
        embeddings.stride(0), embeddings.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
