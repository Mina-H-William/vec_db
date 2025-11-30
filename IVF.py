import numpy as np
import struct
import heapq
from sklearn.cluster import MiniBatchKMeans

DIMENSION = 64

class BasicIVFIndexer:
    def __init__(self, n_clusters=1000):
        self.n_clusters = n_clusters
        self.centroids = None
        self.vector_ids = None

    def Build(self, vectors, batch_size=100_000):
        print("Building optimized IVF index...")
        vector_ids = np.arange(len(vectors))
        n_samples = len(vectors)

        # Pre-normalize all vectors ONCE (saves repeated normalization)
        print("Pre-normalizing vectors...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors_normalized = vectors / norms
        
        # Build with more epochs for better centroids
        labels = self.build_index_optimized(vectors_normalized, n_samples, batch_size)

        # Build vector ID lists per cluster
        self.vector_ids = [[] for _ in range(self.n_clusters)]
        for vid, lbl in zip(vector_ids, labels):
            self.vector_ids[int(lbl)].append(int(vid))
        
        # Check cluster balance
        cluster_sizes = [len(lst) for lst in self.vector_ids]
        print(f"Cluster stats: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # Rebalance if needed (split large clusters)
        self.rebalance_clusters(vectors_normalized, threshold=3.0)
        
        print("IVF index built successfully.")

    def build_index_optimized(self, vectors_normalized, n_samples, batch_size, epochs=10):
        # Use larger batch for more stable centroids
        effective_batch_size = min(batch_size * 2, n_samples // 10)
        
        mbk = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=effective_batch_size,
            max_iter=300,  # More iterations per batch
            random_state=0,
            n_init='auto',
            reassignment_ratio=0.01,  # Better handling of empty clusters
            max_no_improvement=10
        )
        
        # Train with MORE epochs for better convergence
        for epoch in range(epochs):
            print(f"Training epoch {epoch+1}/{epochs}")
            for start in range(0, n_samples, effective_batch_size):
                end = min(start + effective_batch_size, n_samples)
                batch = vectors_normalized[start:end]
                mbk.partial_fit(batch)

        # Get and normalize centroids
        self.centroids = mbk.cluster_centers_
        self.centroids /= (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)

        # Assign final labels
        print("Assigning final labels...")
        labels = np.empty(n_samples, dtype=np.int32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = vectors_normalized[start:end]
            labels[start:end] = mbk.predict(batch)

        return labels
    
    def rebalance_clusters(self, vectors_normalized, threshold=3.0):
        """
        Split oversized clusters for faster search
        
        If a cluster is >threshold times the average size, split it.
        This reduces search time in large clusters.
        """
        cluster_sizes = np.array([len(lst) for lst in self.vector_ids])
        avg_size = cluster_sizes.mean()
        max_size = avg_size * threshold
        
        large_clusters = np.where(cluster_sizes > max_size)[0]
        
        if len(large_clusters) == 0:
            print("No rebalancing needed.")
            return
        
        print(f"Rebalancing {len(large_clusters)} oversized clusters...")
        
        new_vector_ids = list(self.vector_ids)
        new_centroids = list(self.centroids)
        
        for cluster_idx in large_clusters:
            vec_ids = np.array(self.vector_ids[cluster_idx])
            
            if len(vec_ids) < 2:
                continue
            
            # Get vectors in this cluster
            cluster_vecs = vectors_normalized[vec_ids]
            
            # Split into 2 sub-clusters using k-means
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            sub_labels = km.fit_predict(cluster_vecs)
            
            # Update cluster assignments
            sub0_ids = vec_ids[sub_labels == 0].tolist()
            sub1_ids = vec_ids[sub_labels == 1].tolist()
            
            # Replace original cluster
            new_vector_ids[cluster_idx] = sub0_ids
            new_centroids[cluster_idx] = km.cluster_centers_[0]
            
            # Add new cluster
            new_vector_ids.append(sub1_ids)
            new_centroids.append(km.cluster_centers_[1])
        
        # Update instance variables
        self.vector_ids = new_vector_ids
        self.centroids = np.array(new_centroids)
        self.n_clusters = len(self.centroids)
        
        # Normalize new centroids
        self.centroids /= (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        
        print(f"Rebalanced to {self.n_clusters} clusters (was {len(large_clusters)} oversized)")

    def write_index(self, filename):
        """Write optimized index with metadata for faster search"""
        vector_ids_lengths = np.array([len(lst) for lst in self.vector_ids], dtype=np.uint32)

        # Sort each list AND store cluster statistics
        cluster_stats = []
        for i, lst in enumerate(self.vector_ids):
            lst.sort()
            # Store cluster metadata (can help with search optimization)
            cluster_stats.append({
                'size': len(lst),
                'min_id': lst[0] if lst else 0,
                'max_id': lst[-1] if lst else 0
            })

        vector_ids_flat = np.concatenate(self.vector_ids) if any(self.vector_ids) else np.array([], dtype=np.uint32)
        
        with open(filename, "wb") as f:
            # Header with version info
            f.write(struct.pack("II", 
                self.n_clusters,
                self.centroids.shape[1],
            ))
            
            # Reserve space for offsets
            f.write(b"\x00" * 4 * 3)
            
            centroid_offset = f.tell()
            f.write(self.centroids.astype(np.float32).tobytes())
            
            lengths_offset = f.tell()
            f.write(vector_ids_lengths.astype(np.uint32).tobytes())
            
            ids_offset = f.tell()
            f.write(vector_ids_flat.astype(np.uint32).tobytes())
            
            # Write offsets
            f.seek(4 * 4)  # After header
            f.write(struct.pack("III", centroid_offset, lengths_offset, ids_offset))
        
        print(f"Optimized index saved: {self.n_clusters} clusters, {len(vector_ids_flat)} vectors")
        
        # Print cluster quality metrics
        sizes = [s['size'] for s in cluster_stats]
        print(f"Cluster quality: min={min(sizes)}, max={max(sizes)}, "
              f"avg={np.mean(sizes):.1f}, std={np.std(sizes):.1f}")


################################################################################
# EFFICIENT get_rows() implementation for VecDB
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
        header = struct.unpack("II", f.read(8))
        n_clusters = header[0]
        dim = header[1]
        
        centroid_offset, lengths_offset, ids_offset = struct.unpack("III", f.read(12))
        f.seek(lengths_offset)
        lengths_array = np.frombuffer(f.read(n_clusters * 4), dtype=np.uint32)

    # n_probe scales with n_clusters: 6 for 1K, 8 for 10K, 10 for 20K
    n_probe = int(6 + (n_clusters - 1000) * 4 / 19000)

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