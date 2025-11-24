import numpy as np
import struct
import heapq
from sklearn.cluster import MiniBatchKMeans
import mmap


DIMENSION = 70

class BasicIVFIndexer:
    def __init__(self, n_clusters=1000, n_probe=10):
        self.n_clusters = n_clusters
        self.n_probe = n_probe  # Number of clusters to search
        self.centroids = None
        self.vector_ids = None

    def Build(self, vectors, batch_size=100_000):
        """
        vectors: array-like or memmap supporting slicing: vectors[start:end]
        vector_ids: optional sequence of ids
        If dataset is large (>= use_minibatch_threshold) or vectors is a memmap,
        use MiniBatchKMeans and batch normalization to avoid full in-RAM copies.
        """
        print("Building IVF index...")
        vector_ids = np.arange(len(vectors))

        n_samples = len(vectors)

        labels = self.build_index_with_batchs(vectors, n_samples, batch_size)

        # Build vector ID lists per cluster
        self.vector_ids = [[] for _ in range(self.n_clusters)]
        for vid, lbl in zip(vector_ids, labels):
            self.vector_ids[int(lbl)].append(int(vid))

        print("IVF index built successfully.")

    def build_index_with_batchs(self, vectors, n_samples, batch_size):
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters,
                                  batch_size=batch_size,
                                  random_state=0,
                                  n_init='auto')
        # Partial fit on normalized batches
        for start in range(0, n_samples, batch_size):
            print(f"Processing batch {start} to {min(start + batch_size, n_samples)}")
            end = min(start + batch_size, n_samples)
            batch = vectors[start:end]
            # batch may be a view; normalize without creating a huge extra copy
            norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12
            batch_norm = batch / norms  # small temporary per-batch
            mbk.partial_fit(batch_norm)

        # Save centroids
        self.centroids = mbk.cluster_centers_
        self.centroids /= (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)

        # Assign labels in batches (predict on normalized batches)
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
        vector_ids_lengths = np.array([len(lst) for lst in self.vector_ids], dtype=np.uint32)  # Use uint32

        # sort each list in vectorIds 
        for lst in self.vector_ids:
            lst.sort()

        vector_ids_flat = np.concatenate(self.vector_ids) if any(self.vector_ids) else np.array([], dtype=np.uint32)
        
        with open(filename, "wb") as f:
            # 1. Header (reduced size)
            f.write(struct.pack("III", self.n_clusters, self.n_probe, self.centroids.shape[1]))
            
            # Reserve space for offsets (4 bytes each instead of 8)
            f.write(b"\x00" * 4 * 3)
            
            centroid_offset = f.tell()
            f.write(self.centroids.astype(np.float32).tobytes())  # 4 bytes per element
            
            # 3. Lengths as uint32 instead of int64 (50% reduction)
            lengths_offset = f.tell()
            f.write(vector_ids_lengths.astype(np.uint32).tobytes())  # 4 bytes per length
            
            # 4. Vector IDs - biggest savings here
            ids_offset = f.tell()
            
            # Since IDs are 1-20M, we can use uint32 (4 bytes) instead of int64 (8 bytes)
            f.write(vector_ids_flat.astype(np.uint32).tobytes())  # 4 bytes per ID
            
            # 5. Write offsets as uint32
            f.seek(4 * 3)  # After header
            f.write(struct.pack("III", centroid_offset, lengths_offset, ids_offset))
        
        print("Optimized index saved to", filename)


################################################################################

def cal_score(vec1, vec2):
    # Compute cosine similarity and return a plain Python float.
    # Guard against zero norms to avoid division-by-zero and return 0.0 in that case.
    dot_product = np.dot(vec1, vec2)
    # norm_vec1 = np.linalg.norm(vec1)
    # norm_vec2 = np.linalg.norm(vec2)
    return dot_product


def load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
    with open(filename, "rb") as f:
        f.seek(centroid_offset)

        for start in range(0, n_clusters, batch_size):
            end = min(batch_size, n_clusters - start)
            bytes_to_read = end * dim * 4  # float32 size
            batch = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
            yield start, np.array(batch.reshape(end, dim))

def load_cluster_ids(filename, cluster_index, lengths_array, ids_offset):
    with open(filename, "rb") as f:

        # Get offset of this cluster inside ids
        start = lengths_array[:cluster_index].sum().astype(np.uint32)
        length = lengths_array[cluster_index]

        # Read that slice only
        f.seek(ids_offset + start * 4)
        data = np.frombuffer(f.read(length * 4), dtype=np.uint32)

        return np.array(data)


def search(vec_db, query_vector, k=5, batch_size=500):
    filename = vec_db.index_path
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-12)

    # ---- 1. Read header ----
    with open(filename, "rb") as f:
        # mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n_clusters, n_probe, dim = struct.unpack("III", f.read(12))
        centroid_offset, lengths_offset, ids_offset = struct.unpack("III", f.read(12))
        f.seek(lengths_offset)
        lengths_array = np.frombuffer(f.read(n_clusters * 4), dtype=np.uint32)

    # Min-heap to store top n_probe centroids (score, centroid_index)
    centroid_scores_heap = []

    for start_idx, batch in load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
        # batch shape: (batch_size, dim)
        for i, centroid in enumerate(batch):
            score = cal_score(query_vector, centroid)
            item = (score, start_idx + i)

            if len(centroid_scores_heap) < n_probe:
                heapq.heappush(centroid_scores_heap, item)
            else:
                # pushpop ensures only top n_probe remain
                if item > centroid_scores_heap[0]:
                    heapq.heappushpop(centroid_scores_heap, item)

    # After iterating all batches, extract the top n_probe centroid indices
    selected_centroids = [cid for _, cid in centroid_scores_heap]

    # ---- 4. Search actual vectors in selected clusters ----
    # lengths_array = np.frombuffer(mm, dtype=np.uint32, count=n_clusters, offset=lengths_offset)

    candidates = []
    # all_vec_ids = []

    for cid in selected_centroids:
        vec_ids = load_cluster_ids(filename, cid, lengths_array, ids_offset)
    #     all_vec_ids.extend(vec_ids)
    
    # all_vec_ids = np.sort(np.array(all_vec_ids, dtype=np.uint32))

    # Process in batches to limit memory usage
    # for start in range(0, len(all_vec_ids), 300):
    #     vec_ids = all_vec_ids[start:start+batch_size]

        # for start in range(0, len(vec_ids), batch_size):
        #     vec_ids_batch = vec_ids[start:start+batch_size]

        vecs = vec_db.get_rows(vec_ids)

        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

        for vid, vec in zip(vec_ids.astype(np.int64), vecs):
            score = cal_score(query_vector, vec)

            item = (score, -vid)

            if len(candidates) < k:
                heapq.heappush(candidates, item)
            else:
                if item > candidates[0]:
                    heapq.heappushpop(candidates, item)

    # ---- 5. Final results ----
    results = [(score, -vid) for score, vid in candidates]
    results.sort(key=lambda x: (x[0], x[1]))  # sort by score then ID

    return [vid for _, vid in results]
