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
        
        # LSH Parameters - tuned for 20M vectors
        self.L = 8  # Number of hash tables
        self.K = 16  # Bits per hash table
        self.num_probes = 5  # Multi-probe LSH
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
        hash_tables = [defaultdict(list) for _ in range(self.L)]
        
        # Hash all vectors
        print(f"Hashing {num_records} vectors...")
        
        # Use memory-mapped file to avoid loading all vectors at once
        vectors_mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                 shape=(num_records, DIMENSION))
        
        # Process in batches to manage memory
        batch_size = 10000
        for start_idx in range(0, num_records, batch_size):
            end_idx = min(start_idx + batch_size, num_records)
            batch = np.array(vectors_mmap[start_idx:end_idx])
            
            # Hash each vector in batch
            for i, vector in enumerate(batch):
                vector_id = start_idx + i

                # Compute hash for each table
                for table_idx in range(self.L):
                    # Compute projections onto all K hyperplanes
                    projections = np.dot(self.hyperplanes[table_idx], vector)
                    # Convert to binary bits and pack into integer (MSB = first hyperplane)
                    bits = (projections >= 0).astype(np.uint8)
                    hash_int = 0
                    for b in bits:
                        hash_int = (hash_int << 1) | int(b)
                    # Store vector ID in bucket using integer key
                    hash_tables[table_idx][hash_int].append(vector_id)
            
            if (end_idx) % 100000 == 0:
                print(f"  Hashed {end_idx}/{num_records} vectors...")
        
        # Save index to disk in optimized binary format
        print("Saving index to disk...")
        self._save_index_optimized(hash_tables)
        print("Index built successfully!")

    # --- helpers for varint encoding/decoding ---
    def _encode_varint(self, value: int) -> bytes:
        """LEB128-style unsigned varint encoding"""
        out = bytearray()
        while True:
            to_write = value & 0x7F
            value >>= 7
            if value:
                out.append(0x80 | to_write)
            else:
                out.append(to_write)
                break
        return bytes(out)

    def _decode_varints(self, data: bytes):
        """Decode consecutive unsigned varints from bytes. Returns list of ints."""
        res = []
        i = 0
        n = len(data)
        while i < n:
            shift = 0
            val = 0
            while True:
                b = data[i]
                i += 1
                val |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            res.append(val)
        return res

    def _save_index_optimized(self, hash_tables):
        """
        Save LSH index in optimized binary format
        Structure:
        - Header: L, K, num_hyperplanes
        - Hyperplanes: L × K × DIMENSION float32 values
        - For each table:
            - num_buckets (uint32)
            - For each bucket:
                - hash_code (K bytes)
                - num_vectors (uint32)
                - vector_ids (uint32 array with delta encoding)
        """
        # New format:
        # [header: L,K,DIM]
        # [hyperplanes] each table as float16 raw binary (K*DIM values)
        # For each table:
        #   num_buckets (I)
        #   for each bucket:
        #       hash_code (nbytes = (K+7)//8, big-endian)
        #       num_vectors (I)
        #       if num_vectors == 0: continue
        #       write first_id (I)
        #       if num_vectors > 1:
        #           write compressed_length (I)
        #           write compressed_data (zlib) of varint-encoded deltas
        with open(self.index_path, 'wb') as f:
            # Write header
            f.write(struct.pack('III', self.L, self.K, DIMENSION))

            # Write hyperplanes as float16 for compactness
            for table_idx in range(self.L):
                planes16 = self.hyperplanes[table_idx].astype(np.float16)
                planes16.tofile(f)

            nbytes = (self.K + 7) // 8

            # Write hash tables
            for table_idx in range(self.L):
                table = hash_tables[table_idx]

                # Write number of buckets
                f.write(struct.pack('I', len(table)))

                # Write each bucket
                for hash_int, vector_ids in table.items():
                    # Write packed hash code (minimal bytes)
                    f.write(int(hash_int).to_bytes(nbytes, byteorder='big'))

                    # Write number of vectors
                    f.write(struct.pack('I', len(vector_ids)))

                    # Sort vector IDs for delta encoding
                    vector_ids = sorted(vector_ids)

                    if len(vector_ids) == 0:
                        continue

                    # Write first ID as-is
                    f.write(struct.pack('I', vector_ids[0]))

                    if len(vector_ids) > 1:
                        # Build delta list
                        deltas = []
                        for i in range(1, len(vector_ids)):
                            deltas.append(vector_ids[i] - vector_ids[i - 1])

                        # Varint-encode deltas
                        varints = bytearray()
                        for d in deltas:
                            varints.extend(self._encode_varint(d))

                        # Compress varints with zlib (max compression)
                        comp = zlib.compress(bytes(varints), level=9)

                        # Write compressed length and data
                        f.write(struct.pack('I', len(comp)))
                        f.write(comp)
                    else:
                        # No deltas: for a single-ID bucket we write nothing more (no compressed length)
                        pass

    def _load_index_optimized(self):
        """
        Load LSH index from optimized binary format
        """
        with open(self.index_path, 'rb') as f:
            # Read header
            self.L, self.K, dim = struct.unpack('III', f.read(12))
            
            # Read hyperplanes
            self.hyperplanes = []
            for _ in range(self.L):
                # read as float16 then cast to float32 for computation
                planes16 = np.fromfile(f, dtype=np.float16, count=self.K * DIMENSION)
                planes = planes16.astype(np.float32).reshape(self.K, DIMENSION)
                self.hyperplanes.append(planes)
            
            # Read hash tables
            self.hash_tables = []
            nbytes = (self.K + 7) // 8
            for table_idx in range(self.L):
                table = {}

                # Read number of buckets
                num_buckets = struct.unpack('I', f.read(4))[0]

                # Read each bucket
                for _ in range(num_buckets):
                    # Read packed hash code
                    hash_bytes = f.read(nbytes)
                    hash_int = int.from_bytes(hash_bytes, byteorder='big')

                    # Read number of vectors
                    num_vectors = struct.unpack('I', f.read(4))[0]

                    vector_ids = []
                    if num_vectors > 0:
                        # Read first ID
                        first_id = struct.unpack('I', f.read(4))[0]
                        vector_ids.append(first_id)

                        if num_vectors > 1:
                            # Read compressed length and data
                            comp_len = struct.unpack('I', f.read(4))[0]
                            if comp_len > 0:
                                comp = f.read(comp_len)
                                varint_bytes = zlib.decompress(comp)
                                deltas = self._decode_varints(varint_bytes)
                            else:
                                deltas = []

                            # Reconstruct IDs
                            cur = first_id
                            for d in deltas:
                                cur += d
                                vector_ids.append(cur)

                    table[hash_int] = vector_ids

                self.hash_tables.append(table)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve top_k most similar vectors using LSH with multi-probe
        """
        # Load index if not already loaded
        if not hasattr(self, 'hash_tables'):
            self._load_index_optimized()
        
        # Normalize query for cosine similarity
        query = query.flatten()
        query_norm = query / np.linalg.norm(query)
        
        # Collect candidates from all tables
        candidates = set()
        
        for table_idx in range(self.L):
            # Compute integer hash code for query (MSB = first hyperplane)
            projections = np.dot(self.hyperplanes[table_idx], query_norm)
            bits = (projections >= 0).astype(np.uint8)
            hash_int = 0
            for b in bits:
                hash_int = (hash_int << 1) | int(b)

            # Multi-probe: generate neighboring integer hash codes
            probe_codes = self._generate_probes(hash_int, self.num_probes)
            
            # Retrieve candidates from all probed buckets
            for probe_code in probe_codes:
                if probe_code in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][probe_code])
        
        # If too few candidates, return what we have
        if len(candidates) < top_k:
            # Fall back to checking more candidates
            candidates = set(range(min(top_k * 100, self._get_num_records())))
        
        # Compute exact distances for candidates
        scores = []
        vectors_mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                 shape=(self._get_num_records(), DIMENSION))
        
        for candidate_id in candidates:
            vector = vectors_mmap[candidate_id]
            score = self._cal_score(query_norm, vector)
            scores.append((score, candidate_id))
        
        # Sort by score (descending) and return top_k
        scores.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        return [s[1] for s in scores[:top_k]]

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