"""oro-storage -- Erasure-coded resilient storage with Merkle integrity verification.

This module implements resilient storage providing:
- Reed-Solomon erasure coding for distributed redundancy
- Merkle tree integrity verification
- Configurable redundancy levels
- Recovery from partial data loss
- Pluggable storage backends

Example usage:
    from oro_storage import (
        ErasureCodec,
        RedundancyLevel,
        MerkleTree,
        StorageShard,
        ShardSet,
    )

    # Create codec with federation-level redundancy (5 of 9)
    codec = ErasureCodec(RedundancyLevel.FEDERATION)

    # Encode data into shards
    shards = codec.encode(belief_bytes)

    # Verify integrity
    assert codec.verify_integrity(shards)

    # Recover from partial data (any k shards)
    recovered = codec.decode(partial_shards)
"""

__version__ = "0.1.0"

from .backend import (
    BackendRegistry,
    LocalFileBackend,
    MemoryBackend,
    ShardNotFoundError,
    StorageBackend,
    StorageBackendError,
    StorageQuotaExceededError,
    StorageStats,
)
from .erasure import (
    CorruptedDataError,
    ErasureCodec,
    ErasureCodingError,
    InsufficientShardsError,
)
from .integrity import (
    IntegrityVerifier,
    MerkleProof,
    MerkleTree,
    compute_hash,
    verify_proof,
)
from .models import (
    IntegrityReport,
    RecoveryResult,
    RedundancyLevel,
    ShardMetadata,
    ShardSet,
    StorageShard,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "RedundancyLevel",
    "StorageShard",
    "ShardSet",
    "ShardMetadata",
    "RecoveryResult",
    "IntegrityReport",
    # Erasure coding
    "ErasureCodec",
    "ErasureCodingError",
    "InsufficientShardsError",
    "CorruptedDataError",
    # Integrity
    "MerkleTree",
    "MerkleProof",
    "compute_hash",
    "verify_proof",
    "IntegrityVerifier",
    # Backends
    "StorageBackend",
    "LocalFileBackend",
    "MemoryBackend",
    "BackendRegistry",
    "StorageStats",
    # Backend exceptions
    "StorageBackendError",
    "ShardNotFoundError",
    "StorageQuotaExceededError",
]
