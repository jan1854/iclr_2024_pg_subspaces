import math

import torch

from pg_subspaces.sb3_utils.common.parameters import project_orthonormal

ORIG_DIM = 5
SUBSPACE_DIM = 2


def create_random_orthogonal_matrix():
    random_vectors = torch.randn(ORIG_DIM, SUBSPACE_DIM)
    sub, _ = torch.linalg.qr(random_vectors)
    return sub


def create_jl_lemma_matrix():
    return (2 * torch.randint(0, 2, size=(ORIG_DIM, SUBSPACE_DIM)) - 1) / math.sqrt(
        SUBSPACE_DIM
    )


random_vectors = torch.randn(ORIG_DIM, 1000)
subspace_orthogonal = create_random_orthogonal_matrix()
projected_orthogonal = project_orthonormal(
    random_vectors, subspace_orthogonal, result_in_orig_space=False
)
print(
    f"Random orthogonal: {(projected_orthogonal.norm(dim=0) ** 2 / random_vectors.norm(dim=0) ** 2).mean().item()}"
)

print(
    f"Diff. criterion: {(1 - torch.norm(project_orthonormal(random_vectors, subspace_orthogonal, result_in_orig_space=True) - random_vectors, dim=0) ** 2 / torch.norm(random_vectors, dim=0) ** 2).mean()}"
)

subspace_jl = create_jl_lemma_matrix()
projected_jl = project_orthonormal(
    random_vectors, subspace_jl, result_in_orig_space=False
)
print(
    f"Random JL lemma: {(projected_jl.norm(dim=0) ** 2 / random_vectors.norm(dim=0) ** 2).mean().item()}"
)

subspace_jl_scaled = subspace_jl / subspace_jl.norm(dim=0)
projected_jl_scaled = project_orthonormal(
    random_vectors, subspace_jl_scaled, result_in_orig_space=False
)
print(
    f"Random JL lemma scaled: {(projected_jl_scaled.norm(dim=0) ** 2 / random_vectors.norm(dim=0) ** 2).mean().item()}"
)

projected_jl_orig_space = (
    subspace_jl @ subspace_jl.T @ random_vectors / subspace_jl.norm(dim=0)[0] ** 2
)
print(
    f"Random JL lemma inverted: {(projected_jl_orig_space.norm(dim=0) ** 2 / random_vectors.norm(dim=0) ** 2).mean().item()}"
)
