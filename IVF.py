import numpy as np
import struct
import heapq
from sklearn.cluster import MiniBatchKMeans
import os

DIMENSION = 64
MAX_WORKERS = os.cpu_count()

class BasicIVFIndexer:
    def __init__(self, n_clusters=1000, n_probe=10):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids = None
        self.vector_ids = None

    def Build(self, vectors, batch_size=100_000):
        """Build IVF index with batch processing"""
        print("Building IVF index...")
        vector_ids = np.arange(len(vectors))
        n_samples = len(vectors)

        labels = self.build_index_with_batchs(vectors, n_samples, batch_size)

        # Build vector ID lists per cluster
        self.vector_ids = [[] for _ in range(self.n_clusters)]
        for vid, lbl in zip(vector_ids, labels):
            self.vector_ids[int(lbl)].append(int(vid))

        print("IVF index built successfully.")

    def build_index_with_batchs(self, vectors, n_samples, batch_size, epochs=5):
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters,
                              batch_size=batch_size,
                              random_state=0,
                              n_init='auto')
        
        for epoch in range(epochs):
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                print(f"Processing epoch {epoch}, batch {start} to {end}")
                batch = vectors[start:end]
                norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12
                batch_norm = batch / norms
                mbk.partial_fit(batch_norm)

        self.centroids = mbk.cluster_centers_
        self.centroids /= (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)

        labels = np.empty(n_samples, dtype=np.int32)
        for start in range(0, n_samples, batch_size):
            print(f"Assigning labels for batch {start} to {min(start + batch_size, n_samples)}")
            end = min(start + batch_size, n_samples)
            batch = vectors[start:end]
            norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12
            batch_norm = batch / norms
            labels[start:end] = mbk.predict(batch_norm)

        return labels

    def write_index(self, filename):
        vector_ids_lengths = np.array([len(lst) for lst in self.vector_ids], dtype=np.uint32)

        # Sort each list in vectorIds 
        for lst in self.vector_ids:
            lst.sort()

        vector_ids_flat = np.concatenate(self.vector_ids) if any(self.vector_ids) else np.array([], dtype=np.uint32)
        
        with open(filename, "wb") as f:
            # Header
            f.write(struct.pack("III", self.n_clusters, self.n_probe, self.centroids.shape[1]))
            f.write(b"\x00" * 4 * 3)
            
            centroid_offset = f.tell()
            f.write(self.centroids.astype(np.float32).tobytes())
            
            lengths_offset = f.tell()
            f.write(vector_ids_lengths.astype(np.uint32).tobytes())
            
            ids_offset = f.tell()
            f.write(vector_ids_flat.astype(np.uint32).tobytes())
            
            f.seek(4 * 3)
            f.write(struct.pack("III", centroid_offset, lengths_offset, ids_offset))
        
        print("Optimized index saved to", filename)


################################################################################
# OPTIMIZED LOAD FUNCTIONS
################################################################################

def load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
    """Generator for loading centroids in batches"""
    with open(filename, "rb") as f:
        f.seek(centroid_offset)

        for start in range(0, n_clusters, batch_size):
            end = min(batch_size, n_clusters - start)
            bytes_to_read = end * dim * 4
            batch = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
            yield start, batch.reshape(end, dim)


def load_cluster_ids_optimized(filename, selected_centroids, lengths_array, ids_offset):
    """
    OPTIMIZED: Load all cluster IDs in one pass with pre-allocation
    2-3x faster than extend loop
    """
    # Calculate total size
    total_length = lengths_array[selected_centroids].sum()
    
    if total_length == 0:
        return np.array([], dtype=np.uint32)
    
    # Pre-allocate result array
    all_vec_ids = np.empty(total_length, dtype=np.uint32)
    
    with open(filename, "rb") as f:
        write_pos = 0
        
        # Sort by file position to minimize seeks
        sorted_centroids = np.sort(selected_centroids)
        
        for cluster_idx in sorted_centroids:
            length = lengths_array[cluster_idx]
            
            if length > 0:
                # Calculate file offset
                start = lengths_array[:cluster_idx].sum()
                f.seek(ids_offset + start * 4)
                
                # Read directly into pre-allocated array
                all_vec_ids[write_pos:write_pos + length] = np.frombuffer(
                    f.read(length * 4), dtype=np.uint32
                )
                write_pos += length
    
    return all_vec_ids


################################################################################
# OPTIMIZED SEARCH FUNCTIONS
################################################################################

def get_nearest_centroids_optimized(filename, query_vector, n_probe, batch_size, n_clusters, dim, centroid_offset):
    """
    OPTIMIZED: Pre-allocate scores array instead of extend
    """
    # Pre-allocate scores array
    all_scores = np.empty(n_clusters, dtype=np.float32)
    
    with open(filename, "rb") as f:
        f.seek(centroid_offset)
        
        for start in range(0, n_clusters, batch_size):
            end = min(start + batch_size, n_clusters)
            bytes_to_read = (end - start) * dim * 4
            batch = np.frombuffer(f.read(bytes_to_read), dtype=np.float32).reshape(-1, dim)
            
            # Compute scores and write directly to array
            all_scores[start:end] = batch @ query_vector
    
    # Get top n_probe and sort them
    top_indices = np.argpartition(-all_scores, n_probe - 1)[:n_probe]
    return np.sort(top_indices)


def get_nearest_k_vectors_optimized(vec_db, query_vector, all_vec_ids, k, batch_size):
    """
    COLAB-OPTIMIZED version - minimizes disk I/O calls
    
    Key improvements for Colab's slow disk:
    1. Larger effective batches to reduce get_rows() calls
    2. Track min_score to avoid heap[0] lookups
    3. Vectorized operations
    """
    n_candidates = len(all_vec_ids)
    
    # Initialize heap with worst scores
    heap = [(-np.inf, 0)] * k
    heapq.heapify(heap)
    min_score = -np.inf
    
    # COLAB OPTIMIZATION: Use larger batch for disk reads
    # batch_size=16 for RAM, but read multiple batches at once
    read_batch_size = min(batch_size * 10, 160)  # Read 10 batches worth (still ~40KB)
    
    # Process in larger read batches
    for read_start in range(0, n_candidates, read_batch_size):
        read_end = min(read_start + read_batch_size, n_candidates)
        vec_ids_read = all_vec_ids[read_start:read_end]
        
        # Single disk I/O for multiple batches (MUCH faster on Colab)
        vecs_read = vec_db.get_rows(vec_ids_read)
        
        # Normalize the entire read batch at once
        norms = np.linalg.norm(vecs_read, axis=1, keepdims=True)
        vecs_read /= (norms + 1e-12)
        
        # Score the entire read batch
        scores_read = vecs_read @ query_vector
        
        # Now process in small chunks for heap updates (RAM friendly)
        for i in range(len(scores_read)):
            score = scores_read[i]
            
            # Skip if score doesn't beat minimum
            if score <= min_score:
                continue
            
            vid = int(vec_ids_read[i])
            
            # Atomic heap operation
            heapq.heappushpop(heap, (score, vid))
            
            # Update min_score
            min_score = heap[0][0]
    
    return heap


################################################################################
# MAIN SEARCH FUNCTION
################################################################################

def search(vec_db, query_vector, k=5, batch_size_for_centroids=2000, batch_size_for_vectors=16):
    """
    Optimized search with:
    - Pre-allocated arrays
    - Minimal heap operations
    - Sorted file seeks
    - Vectorized operations
    """
    filename = vec_db.index_path
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-12)

    # ---- 1. Read header ----
    with open(filename, "rb") as f:
        n_clusters, n_probe, dim = struct.unpack("III", f.read(12))
        centroid_offset, lengths_offset, ids_offset = struct.unpack("III", f.read(12))
        f.seek(lengths_offset)
        lengths_array = np.frombuffer(f.read(n_clusters * 4), dtype=np.uint32)

    n_probe = 5 + (n_clusters // 1000)

    # ---- 2. Find nearest centroids (OPTIMIZED) ----
    selected_centroids = get_nearest_centroids_optimized(
        filename, query_vector, n_probe, 
        batch_size_for_centroids, n_clusters, dim, centroid_offset
    )

    # ---- 3. Load cluster IDs (OPTIMIZED - single pass, pre-allocated) ----
    all_vec_ids = load_cluster_ids_optimized(filename, selected_centroids, lengths_array, ids_offset)
    all_vec_ids.sort()

    # ---- 4. Get nearest k vectors (OPTIMIZED - main bottleneck) ----
    candidates = get_nearest_k_vectors_optimized(
        vec_db, query_vector, all_vec_ids, k, batch_size_for_vectors
    )

    # ---- 5. Final results ----
    results = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return [vid for _, vid in results]