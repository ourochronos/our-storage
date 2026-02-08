"""Public interface for oro-storage.

Re-exports the primary public API for the oro-storage brick.
Consumers should import from here or from the top-level ``oro_storage`` package.

The key abstractions:

- **StorageBackend**: Abstract base class for storage backends.
  Implementations: ``MemoryBackend``, ``LocalFileBackend``.
- **BackendRegistry**: Manages multiple backends and distributes shards.
- **ErasureCodec**: Reed-Solomon erasure coding (encode/decode/repair).
- **IntegrityVerifier**: Merkle-tree-based integrity checking.
- **MerkleTree** / **MerkleProof**: Cryptographic integrity primitives.
- **models**: Data models (``RedundancyLevel``, ``StorageShard``, ``ShardSet``, etc.).
"""

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
    # Backends
    "StorageBackend",
    "MemoryBackend",
    "LocalFileBackend",
    "BackendRegistry",
    "StorageStats",
    # Backend exceptions
    "StorageBackendError",
    "ShardNotFoundError",
    "StorageQuotaExceededError",
    # Erasure coding
    "ErasureCodec",
    "ErasureCodingError",
    "InsufficientShardsError",
    "CorruptedDataError",
    # Integrity
    "IntegrityVerifier",
    "MerkleTree",
    "MerkleProof",
    "compute_hash",
    "verify_proof",
    # Models
    "RedundancyLevel",
    "StorageShard",
    "ShardSet",
    "ShardMetadata",
    "RecoveryResult",
    "IntegrityReport",
]
