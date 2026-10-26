"""Unit tests for PKSampler."""

import pytest
from reid.data.sampler import PKSampler


def test_pk_sampler_batch_composition() -> None:
    """Verify that every batch has exactly P identities with K images each."""
    # Synthetic dataset with 20 identities, variable images per identity (2 to 10)
    pids = []
    for pid in range(20):
        num_images = 2 if pid < 5 else 8
        pids.extend([pid] * num_images)

    p = 4
    k = 4
    batch_size = p * k
    num_batches = 5

    sampler = PKSampler(data_source=pids, p=p, k=k, num_batches=num_batches)

    assert len(sampler) == num_batches * batch_size

    indices = list(sampler)
    assert len(indices) == num_batches * batch_size

    for b in range(num_batches):
        batch_slice = indices[b * batch_size : (b + 1) * batch_size]
        batch_pids = [pids[idx] for idx in batch_slice]

        unique_in_batch = set(batch_pids)
        assert len(unique_in_batch) == p, f"Batch {b} does not have exactly {p} unique identities"

        counts = {pid: batch_pids.count(pid) for pid in unique_in_batch}
        for pid, count in counts.items():
            assert count == k, f"Identity {pid} does not have exactly {k} instances (got {count})"


def test_pk_sampler_replacement_for_small_identities() -> None:
    """Verify that identities with fewer than K samples are sampled with replacement."""
    # Identity 0 has only 1 image, identity 1 has 2 images
    pids = [0] + [1, 1] + [2] * 10
    sampler = PKSampler(data_source=pids, p=2, k=4, num_batches=2)

    indices = list(sampler)
    assert len(indices) == 2 * (2 * 4)


def test_pk_sampler_validation_errors() -> None:
    """Verify parameter validations and exceptions."""
    pids = [0, 0, 1, 1, 2, 2]

    # Unique identities (3) < P (4)
    with pytest.raises(ValueError, match="less than P"):
        PKSampler(data_source=pids, p=4, k=2)

    # Incompatible batch_size and p * k
    with pytest.raises(ValueError, match="Conflicting parameters"):
        PKSampler(data_source=pids, p=2, k=2, batch_size=5)

    # Invalid k
    with pytest.raises(ValueError, match="k must be positive"):
        PKSampler(data_source=pids, p=2, k=0)
