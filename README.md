# our-storage

Reed-Solomon erasure coding with Merkle integrity verification and pluggable storage backends for the ourochronos ecosystem.

## Overview

our-storage provides resilient data storage through erasure coding — data is split into shards where any k-of-n shards can reconstruct the original. A Merkle tree provides cryptographic integrity verification without downloading all shards. Shards can be distributed across multiple storage backends (memory, local filesystem, S3-compatible) for redundancy. The entire implementation is pure Python with zero runtime dependencies.

## Install

```bash
pip install our-storage
```

No runtime dependencies.

## Usage

### Encode and Decode

```python
from uuid import uuid4
from our_storage import ErasureCodec, RedundancyLevel

codec = ErasureCodec(level=RedundancyLevel.PERSONAL)  # 3-of-5

# Encode data into shards
shard_set = codec.encode(b"belief data to protect", belief_id=str(uuid4()))

# Decode from any 3 of the 5 shards
available = shard_set.available_shards[:3]
result = codec.decode(available)
assert result.recovered_data == b"belief data to protect"
```

### Redundancy Levels

| Level | Config | Overhead | Survives |
|-------|--------|----------|----------|
| `MINIMAL` | 2-of-3 | 50% | 1 failure |
| `PERSONAL` | 3-of-5 | 67% | 2 failures |
| `FEDERATION` | 5-of-9 | 80% | 4 failures |
| `PARANOID` | 7-of-15 | 114% | 8 failures |

### Integrity Verification

```python
from our_storage import IntegrityVerifier, MerkleTree, verify_proof

# Verify all shards in a set
verifier = IntegrityVerifier()
report = verifier.verify_shard_set(shard_set)
assert report.is_valid

# Generate and verify Merkle proofs for individual shards
tree = MerkleTree.from_shards(shard_set.available_shards)
proof = tree.get_proof(0)
assert verify_proof(proof, tree.root_hash)
```

### Distributed Storage

```python
from our_storage import BackendRegistry, MemoryBackend, LocalFileBackend

registry = BackendRegistry()
registry.register(MemoryBackend("hot-cache"))
registry.register(LocalFileBackend("/data/shards", max_quota_bytes=10_000_000))

# Distribute shards across backends (round-robin)
locations = await registry.distribute_shard_set(shard_set)

# Retrieve and reconstruct
retrieved = await registry.retrieve_shard_set(locations)
result = codec.decode(retrieved.available_shards)
```

### Repair Missing Shards

```python
# Regenerate missing parity shards from available data
repaired = codec.repair(degraded_shard_set)
```

## API

### Erasure Coding

| Symbol | Description |
|--------|-------------|
| `ErasureCodec` | Encode, decode, and repair with Reed-Solomon |
| `RedundancyLevel` | Predefined k-of-n configurations |
| `ShardSet` | Complete set of shards with metadata and Merkle root |
| `StorageShard` | Single shard with data, index, and checksum |
| `RecoveryResult` | Decode outcome with timing and shard usage info |

### Integrity

| Symbol | Description |
|--------|-------------|
| `MerkleTree` | SHA-256 Merkle tree with proof generation |
| `MerkleProof` | Path from leaf to root for individual verification |
| `IntegrityVerifier` | Verify shard sets and generate integrity reports |
| `IntegrityReport` | Valid/corrupted/missing counts, recovery feasibility |
| `verify_proof()` | Standalone proof verification |
| `compute_hash()` | SHA-256, SHA3-256, or BLAKE2b hashing |

### Storage Backends

| Symbol | Description |
|--------|-------------|
| `StorageBackend` | Abstract interface for pluggable backends |
| `MemoryBackend` | In-memory storage (testing) |
| `LocalFileBackend` | Filesystem storage with quota support |
| `BackendRegistry` | Multi-backend coordinator with round-robin distribution |

## Development

```bash
# Install with dev dependencies
make dev

# Run linters
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Auto-format
make format
```

## State Ownership

Owns shard data and metadata in registered storage backends. `MemoryBackend` is ephemeral; `LocalFileBackend` persists to the configured directory.

## Part of Valence

This brick is part of the [Valence](https://github.com/ourochronos/valence) knowledge substrate. See [our-infra](https://github.com/ourochronos/our-infra) for ourochronos conventions.

## License

MIT
