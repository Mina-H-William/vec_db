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
# LOAD FUNCTIONS - KEEP YOUR ORIGINAL MINIMAL RAM APPROACH
################################################################################

def load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
    with open(filename, "rb") as f:
        f.seek(centroid_offset)

        for start in range(0, n_clusters, batch_size):
            end = min(batch_size, n_clusters - start)
            bytes_to_read = end * dim * 4
            batch = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
            yield start, batch.reshape(end, dim)


def load_cluster_ids(filename, cluster_index, lengths_array, ids_offset):
    with open(filename, "rb") as f:
        start = lengths_array[:cluster_index].sum()
        length = lengths_array[cluster_index]

        f.seek(ids_offset + start * 4)
        data = np.frombuffer(f.read(length * 4), dtype=np.uint32)

        return data


################################################################################
# OPTIMIZED FUNCTIONS - TINY IMPROVEMENTS WITHOUT BREAKING RAM
################################################################################

def get_nearest_centroids(filename, query_vector, n_probe, batch_size, n_clusters, dim, centroid_offset):
    # Pre-allocate scores array (saves time vs list appends)
    all_scores = np.empty(n_clusters, dtype=np.float32)
    
    idx = 0
    for _, batch in load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
        # Compute scores for batch
        batch_scores = batch @ query_vector
        batch_size_actual = len(batch_scores)
        all_scores[idx:idx + batch_size_actual] = batch_scores
        idx += batch_size_actual

    # Get top n_probe and sort
    top_indices = np.argpartition(-all_scores, n_probe - 1)[:n_probe]
    return np.sort(top_indices)


def get_nearest_k_vectors(vec_db, query_vector, all_vec_ids, k, batch_size):
    heap = []
    min_score = -np.inf
    min_vid = np.inf  # Track the vid at min_score for tie-breaking
    
    # Process in small batches (batch_size=16 for minimal RAM)
    for start in range(0, len(all_vec_ids), batch_size):
        vec_ids = all_vec_ids[start:start + batch_size]
        
        # Get vectors
        vecs = vec_db.get_rows(vec_ids)
        
        # Vectorized normalization (fast)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        
        # Vectorized scoring (fast)
        scores = vecs @ query_vector
        
        # Update heap
        for i in range(len(scores)):
            score = float(scores[i])
            vid = int(vec_ids[i])
            
            if len(heap) < k:
                heapq.heappush(heap, (score, vid))
                if len(heap) == k:
                    min_score = heap[0][0]
                    min_vid = heap[0][1]
            else:
                # Check if we should replace the minimum
                # Replace if: score > min_score OR (score == min_score AND vid < min_vid)
                if score > min_score or (score == min_score and vid < min_vid):
                    heapq.heappushpop(heap, (score, vid))
                    min_score = heap[0][0]
                    min_vid = heap[0][1]
    
    return heap


################################################################################
# MAIN SEARCH - MINIMAL CHANGES
################################################################################

def search(vec_db, query_vector, k=5, batch_size_for_centroids=2000, batch_size_for_vectors=16):
    filename = vec_db.index_path
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-12)

    # ---- 1. Read header ----
    with open(filename, "rb") as f:
        n_clusters, n_probe, dim = struct.unpack("III", f.read(12))
        centroid_offset, lengths_offset, ids_offset = struct.unpack("III", f.read(12))
        f.seek(lengths_offset)
        lengths_array = np.frombuffer(f.read(n_clusters * 4), dtype=np.uint32)

    n_probe = 5 + (n_clusters // 1000)

    # ---- 2. Find nearest centroids (tiny optimization) ----
    selected_centroids = get_nearest_centroids(
        filename, query_vector, n_probe, 
        batch_size_for_centroids, n_clusters, dim, centroid_offset
    )

    # ---- 3. Load cluster IDs (OPTIMIZED - single allocation) ----
    # Calculate total size needed
    total_length = lengths_array[selected_centroids].sum()
    
    if total_length == 0:
        return []
    
    # Pre-allocate result array
    all_vec_ids = np.empty(total_length, dtype=np.uint32)
    
    # Fill in one pass
    write_pos = 0
    for cid in selected_centroids:
        vec_ids = load_cluster_ids(filename, cid, lengths_array, ids_offset)
        length = len(vec_ids)
        all_vec_ids[write_pos:write_pos + length] = vec_ids
        write_pos += length
    
    all_vec_ids.sort()

    # ---- 4. Get nearest k vectors (minimal changes to your original) ----
    candidates = get_nearest_k_vectors(
        vec_db, query_vector, all_vec_ids, k, batch_size_for_vectors
    )

    # ---- 5. Final results ----
    results = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return [vid for _, vid in results]