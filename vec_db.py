from operator import index
from typing import Dict, List, Annotated
import numpy as np
import os
import struct
import pickle
import zlib
from collections import defaultdict

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        # IVF (coarse quantizer) config (set locally, not as params)
        self.use_ivf = False
        self.ivf_clusters = 256
        self.ivf_probe = 4
        self.ivf_iters = 10
        self.ivf_batch = 10000
        
        # LSH Parameters - tuned for 20M vectors
        self.L = 5  # Number of hash tables
        self.K = 12  # Bits per hash table
        self.num_probes = 4  # Multi-probe LSH
        self.LSH_SEED = 123  # Separate seed for LSH
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            self.generate_database(db_size)
        else:
            self._build_index()
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        self.vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(self.vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        """
        Build LSH index with optimized binary storage format
        """
        print("Building LSH index...")
        num_records = self._get_num_records()
        # Optionally build IVF coarse quantizer first (keeps centroids in memory)
        if getattr(self, 'use_ivf', False):
            try:
                self._build_ivf()
            except Exception as e:
                print(f"IVF build failed, continuing without IVF: {e}")
        
        # Generate random hyperplanes for LSH (using separate seed)
        # We'll keep them in float32 in memory for accuracy, but store as float16 on disk
        rng = np.random.default_rng(self.LSH_SEED)
        self.hyperplanes = []
        for _ in range(self.L):
            # K random hyperplanes per table, each is a DIMENSION-dimensional vector
            planes = rng.standard_normal((self.K, DIMENSION)).astype(np.float32)
            # Normalize hyperplanes
            planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
            self.hyperplanes.append(planes)
        

        # Initialize hash tables (use integer hash keys - bit-packed)
        # Now store per-hash -> per-cluster lists to avoid filtering at query time
        # For global LSH, just use hash -> list of vector IDs (no cluster split)
        hash_tables = [defaultdict(list) for _ in range(self.L)]


        # Hash all vectors
        print(f"Hashing {num_records} vectors...")
        vectors_mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        batch_size = 10000
        for start_idx in range(0, num_records, batch_size):
            end_idx = min(start_idx + batch_size, num_records)
            batch = np.array(vectors_mmap[start_idx:end_idx])
            for i, vector in enumerate(batch):
                vector_id = start_idx + i
                for table_idx in range(self.L):
                    projections = np.dot(self.hyperplanes[table_idx], vector)
                    bits = (projections >= 0).astype(np.uint8)
                    hash_int = 0
                    for b in bits:
                        hash_int = (hash_int << 1) | int(b)
                    hash_tables[table_idx][hash_int].append(vector_id)
            if (end_idx) % 100000 == 0:
                print(f"  Hashed {end_idx}/{num_records} vectors...")

        # Save index to disk in optimized binary format
        print("Saving index to disk...")
        self._save_index_optimized(hash_tables)
        print("Index built successfully!")

    # --- helpers for varint encoding/decoding ---
    def _encode_varint(self, value: int) -> bytes:
        """Efficient variable-length encoding for integers
        Small numbers (0-127) take 1 byte
        Medium numbers (128-16383) take 2 bytes
        Large numbers (16384-2097151) take 3 bytes
        And so on...
        """
        buf = bytearray()
        while value > 0x7F:
            buf.append(0x80 | (value & 0x7F))
            value >>= 7
        buf.append(value & 0x7F)
        return bytes(buf)

    def _decode_varint(self, f) -> int:
        """Decode a single varint from a file object"""
        shift = 0
        result = 0
        while True:
            i = ord(f.read(1))
            result |= (i & 0x7F) << shift
            shift += 7
            if not (i & 0x80):
                break
        return result

    def _build_ivf(self):
        """Build coarse quantizer (mini-batch kmeans) and store centroids in memory.

        This is a cautious, in-memory-only implementation: it computes centroids
        and keeps them in self.centroids (float32). It does not yet change on-disk
        index format. It uses memmaped vectors and minibatches to avoid large RAM use.
        """
        C = max(1, int(self.ivf_clusters))
        iters = max(1, int(self.ivf_iters))
        batch = max(1, int(self.ivf_batch))

        num_records = self._get_num_records()
        if num_records == 0:
            raise ValueError("Database empty, cannot build IVF")

        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        rng = np.random.default_rng(self.LSH_SEED + 1)

        # Initialize centroids by sampling up to C vectors
        sample_count = min(C, num_records)
        init_idxs = rng.choice(num_records, size=sample_count, replace=False)
        centroids = np.array([mmap_vectors[i] for i in init_idxs], dtype=np.float32)

        # If C > sample_count, duplicate some samples
        if C > sample_count:
            extra = C - sample_count
            dup_idxs = rng.choice(sample_count, size=extra, replace=True)
            centroids = np.vstack([centroids, centroids[dup_idxs]])

        # Mini-batch kmeans iterations
        for it in range(iters):
            # accumulators
            sums = np.zeros((C, DIMENSION), dtype=np.float64)
            counts = np.zeros((C,), dtype=np.int64)

            for start in range(0, num_records, batch):
                end = min(start + batch, num_records)
                batch_vecs = np.array(mmap_vectors[start:end], dtype=np.float32)

                # compute squared L2 distances between batch and centroids
                # shape (batch_size, C)
                # use (a-b)^2 = a^2 + b^2 - 2ab for speed
                a2 = np.sum(batch_vecs * batch_vecs, axis=1, keepdims=True)  # (B,1)
                b2 = np.sum(centroids * centroids, axis=1, keepdims=True).T  # (1,C)
                ab = np.dot(batch_vecs, centroids.T)  # (B,C)
                dists = a2 + b2 - 2 * ab

                assigns = np.argmin(dists, axis=1)

                # accumulate
                for i, c in enumerate(assigns):
                    sums[c] += batch_vecs[i]
                    counts[c] += 1

            # update centroids; reinitialize empty clusters with a random vector
            for c in range(C):
                if counts[c] > 0:
                    centroids[c] = (sums[c] / counts[c]).astype(np.float32)
                else:
                    centroids[c] = np.array(mmap_vectors[rng.integers(num_records)], dtype=np.float32)

        self.centroids = centroids
        print(f"Built IVF with {C} centroids (iters={iters}, batch={batch})")
        # Persist centroids compactly so they can be loaded later
        try:
            cent_path = self.index_path + ".centroids"
            centroids16 = centroids.astype(np.float16)
            centroids16.tofile(cent_path)
        except Exception:
            # Non-fatal: keep centroids in memory
            pass

    def _assign_clusters(self):
        """Assign every vector to nearest centroid and persist assignments as uint32 memmap.

        Writes assignments to a companion file self.index_path + '.clusters' as uint32 array
        of length num_records. Also computes per-cluster counts and stores in self.cluster_counts.
        """
        if not hasattr(self, 'centroids'):
            raise RuntimeError('Centroids not available for assignment')

        num_records = self._get_num_records()
        C = self.centroids.shape[0]

        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        assign_path = self.index_path + '.clusters'
        # create memmap file for assignments
        assignments = np.memmap(assign_path, dtype=np.uint32, mode='w+', shape=(num_records,))

        batch = max(1, int(self.ivf_batch))
        counts = np.zeros((C,), dtype=np.int64)

        for start in range(0, num_records, batch):
            end = min(start + batch, num_records)
            batch_vecs = np.array(mmap_vectors[start:end], dtype=np.float32)

            a2 = np.sum(batch_vecs * batch_vecs, axis=1, keepdims=True)
            b2 = np.sum(self.centroids * self.centroids, axis=1, keepdims=True).T
            ab = np.dot(batch_vecs, self.centroids.T)
            dists = a2 + b2 - 2 * ab
            assigns = np.argmin(dists, axis=1)

            assignments[start:end] = assigns.astype(np.uint32)
            for c in assigns:
                counts[c] += 1

        assignments.flush()
        self.cluster_assignments_path = assign_path
        self.cluster_assignments = assignments  # memmap object
        self.cluster_counts = counts
        print(f"Assigned {num_records} vectors to {C} clusters (min={counts.min()}, max={counts.max()})")

    def _save_index_optimized(self, hash_tables):
        """
        Save global LSH index in optimized binary storage format with delta compression for vector IDs
        """
        with open(self.index_path, 'wb') as f:
            # Write header with reduced int sizes
            f.write(struct.pack('HHH', self.L, self.K, DIMENSION))  # Use uint16

            # Write hyperplanes as int8 for compactness
            for table_idx in range(self.L):
                planes8 = self.hyperplanes[table_idx].astype(np.float16)  # Reduced precision
                scale = np.max(np.abs(planes8))
                norm_planes = (planes8 / scale * 127).astype(np.int8)
                f.write(struct.pack('f', scale))  # Store scale factor
                norm_planes.tofile(f)

            nbits = (self.K + 7) // 8
            # Write hash tables (no clusters)
            for table_idx in range(self.L):
                table = hash_tables[table_idx]
                f.write(struct.pack('H', len(table)))
                for hash_int, vector_ids in table.items():
                    f.write(int(hash_int).to_bytes(nbits, byteorder='big'))
                    vec_count = len(vector_ids)
                    f.write(struct.pack('I', vec_count))
                    if vec_count == 0:
                        continue
                    vector_ids = sorted(vector_ids)
                    prev_id = vector_ids[0]
                    f.write(struct.pack('I', prev_id))
                    for vid in vector_ids[1:]:
                        delta = vid - prev_id
                        f.write(self._encode_varint(delta))
                        prev_id = vid

    def _load_index_optimized(self):
        """
        Load global LSH index from optimized binary format
        """
        with open(self.index_path, 'rb') as f:
            self.L, self.K, dim = struct.unpack('HHH', f.read(6))
            self.hyperplanes = []
            for _ in range(self.L):
                scale = struct.unpack('f', f.read(4))[0]
                norm_planes = np.fromfile(f, dtype=np.int8, count=self.K * DIMENSION)
                planes = (norm_planes.astype(np.float32) / 127 * scale).reshape(self.K, DIMENSION)
                self.hyperplanes.append(planes)
            self.hash_tables = []
            nbytes = (self.K + 7) // 8
            for table_idx in range(self.L):
                table = {}
                num_buckets = struct.unpack('H', f.read(2))[0]
                for _ in range(num_buckets):
                    hash_bytes = f.read(nbytes)
                    hash_int = int.from_bytes(hash_bytes, byteorder='big')
                    vec_count = struct.unpack('I', f.read(4))[0]
                    vector_ids = []
                    if vec_count > 0:
                        first_id = struct.unpack('I', f.read(4))[0]
                        vector_ids.append(first_id)
                        prev_id = first_id
                        for _ in range(vec_count - 1):
                            delta = self._decode_varint(f)
                            next_id = prev_id + delta
                            vector_ids.append(next_id)
                            prev_id = next_id
                    table[hash_int] = vector_ids
                self.hash_tables.append(table)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve top_k most similar vectors using global LSH with multi-probe, using quantized reranking, batch scoring, and a top-k heap.
        """
        import heapq
        if not hasattr(self, 'hash_tables'):
            self._load_index_optimized()
        query = query.flatten()
        query_norm = query / np.linalg.norm(query)
        candidates = set()
        for table_idx in range(self.L):
            projections = np.dot(self.hyperplanes[table_idx], query_norm)
            bits = (projections >= 0).astype(np.uint8)
            hash_int = 0
            for b in bits:
                hash_int = (hash_int << 1) | int(b)
            probe_codes = self._generate_probes(hash_int, self.num_probes)
            for probe_code in probe_codes:
                if probe_code in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][probe_code])
        if len(candidates) < top_k:
            candidates = set(range(min(top_k * 100, self._get_num_records())))
        candidate_list = list(candidates)
        if len(candidate_list) == 0:
            return []
        vectors_mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(self._get_num_records(), DIMENSION))
        heap = []
        batch_vec_size = 1024
        qnorm = np.linalg.norm(query_norm)
        for i in range(0, len(candidate_list), batch_vec_size):
            batch_ids = candidate_list[i:i+batch_vec_size]
            batch_vectors = np.array(vectors_mmap[batch_ids], dtype=np.float32)
            vec_norms = np.linalg.norm(batch_vectors, axis=1)
            vec_norms[vec_norms == 0] = 1e-12
            batch_scores = (batch_vectors @ query_norm) / (vec_norms * qnorm)
            for j, s in enumerate(batch_scores):
                if len(heap) < top_k:
                    heapq.heappush(heap, (float(s), int(batch_ids[j])))
                else:
                    if float(s) > heap[0][0]:
                        heapq.heappushpop(heap, (float(s), int(batch_ids[j])))
        heap.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        return [s[1] for s in heap]

    def _generate_probes(self, hash_code, num_probes):
        """
        Generate neighboring hash codes by flipping bits (multi-probe LSH)
        Returns the original hash code plus up to (num_probes-1) neighbors
        """
        # hash_code is an integer (bits packed, MSB = first hyperplane)
        probes = [hash_code]

        if num_probes <= 1:
            return probes

        # Generate 1-bit flips (flip bit corresponding to each hyperplane)
        # Our packing uses MSB = hyperplane 0, so bit index for hyperplane i is (K-1-i)
        for i in range(self.K):
            if len(probes) >= num_probes:
                break

            mask = 1 << (self.K - 1 - i)
            flipped = hash_code ^ mask
            probes.append(flipped)

        return probes[:num_probes]