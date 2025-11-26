import numpy as np
import struct
import heapq
from sklearn.cluster import MiniBatchKMeans, KMeans


DIMENSION = 64

class BasicIVFIndexer:
    def __init__(self, n_clusters=1000, n_probe=10, n_subclusters=5):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.n_subclusters = n_subclusters

        # Level-1
        self.centroids = None

        # Level-2 (per cluster)
        self.sub_centroids = None          # shape: (n_clusters, n_subclusters, dim)
        self.sub_vector_ids = None         # list-of-list-of-lists:
                                           # sub_vector_ids[c][s] = IDs for cluster c, subcluster s

    # -----------------------------------------------------------
    # BUILD INDEX
    # -----------------------------------------------------------

    def Build(self, vectors, batch_size=100_000):
        print("Building level-1 IVF index...")

        vector_ids = np.arange(len(vectors))
        n_samples = len(vectors)

        labels = self._build_index_lvl1(vectors, n_samples, batch_size)
        print("Level-1 clustering done.")

        # Create per-cluster ID lists
        lvl1_vector_ids = [[] for _ in range(self.n_clusters)]
        for vid, lbl in zip(vector_ids, labels):
            lvl1_vector_ids[int(lbl)].append(int(vid))

        print("Building level-2 clusters inside each level-1 cluster...")

        self._build_index_lvl2(lvl1_vector_ids, vectors)
        
        print("Index fully built.")

    # -----------------------------------------------------------
    # BUILD LVL-1 INDEX
    # -----------------------------------------------------------

    def _build_index_lvl1(self, vectors, n_samples, batch_size):
        mbk = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=batch_size,
            random_state=0,
            n_init='auto'
        )

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            print(f"Level-1 partial fit batch {start}:{end}")

            batch = vectors[start:end]
            norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12
            batch_norm = batch / norms
            mbk.partial_fit(batch_norm)

        self.centroids = mbk.cluster_centers_
        self.centroids /= (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)

        # Assign labels
        labels = np.empty(n_samples, dtype=np.int32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            print(f"Level-1 assigning batch {start}:{end}")
            batch = vectors[start:end]
            norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12
            batch_norm = batch / norms
            labels[start:end] = mbk.predict(batch_norm)

        return labels

    # -----------------------------------------------------------
    # BUILD LVL-2 INDEX
    # -----------------------------------------------------------

    def _build_index_lvl2(self, lvl1_vector_ids, vectors):
        dim = vectors.shape[1]
        self.sub_centroids = np.zeros((self.n_clusters, self.n_subclusters, dim), dtype=np.float32)
        self.sub_vector_ids = [[[] for _ in range(self.n_subclusters)] for _ in range(self.n_clusters)]

        for c in range(self.n_clusters):

            ids = lvl1_vector_ids[c]
            if len(ids) == 0:
                continue

            cluster_vecs = vectors[ids]

            # Normalize vectors
            norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-12
            cluster_vecs_norm = cluster_vecs / norms

            # Only K clusters or fewer if cluster too small
            K = min(self.n_subclusters, len(ids))

            # Standard KMeans (better for small datasets)
            km = KMeans(
                n_clusters=K,
                random_state=0,
                n_init='auto'
            )
            km.fit(cluster_vecs_norm)

            # Save centroids
            self.sub_centroids[c, :K, :] = km.cluster_centers_

            # Assign vector IDs to subclusters
            labels = km.labels_
            for vid, sl in zip(ids, labels):
                self.sub_vector_ids[c][sl].append(vid)

            # Sort IDs
            for s in range(K):
                self.sub_vector_ids[c][s].sort()

        print("Level-2 hierarchical clustering done.")

    # -----------------------------------------------------------
    # WRITE INDEX TO FILE
    # -----------------------------------------------------------

    def write_index(self, filename):

        print("Saving hierarchical IVF index (minimal version)...")

        # Prepare level-2 data
        lvl2_lengths_list = []
        lvl2_ids_list = []

        for c in range(self.n_clusters):
            for s in range(self.n_subclusters):
                lst = self.sub_vector_ids[c][s]
                lvl2_lengths_list.append(len(lst))
                lvl2_ids_list.extend(lst)

        lvl2_lengths = np.array(lvl2_lengths_list, dtype=np.uint32)
        lvl2_ids_flat = np.array(lvl2_ids_list, dtype=np.uint32)

        with open(filename, "wb") as f:

            # -----------------------------------------------------
            # HEADER
            # -----------------------------------------------------
            f.write(struct.pack("IIII",
                                self.n_clusters,
                                self.n_probe,
                                self.centroids.shape[1],
                                self.n_subclusters))

            # Reserve space for 4 offsets
            f.write(b"\x00" * (4 * 4))

            # -----------------------------------------------------
            # Write level-1 centroids
            # -----------------------------------------------------
            lvl1_centroids_offset = f.tell()
            f.write(self.centroids.astype(np.float32).tobytes())

            # -----------------------------------------------------
            # Level-2 centroids
            # -----------------------------------------------------
            lvl2_centroids_offset = f.tell()
            f.write(self.sub_centroids.astype(np.float32).tobytes())

            # -----------------------------------------------------
            # Level-2 cluster lengths
            # -----------------------------------------------------
            lvl2_lengths_offset = f.tell()
            f.write(lvl2_lengths.tobytes())

            # -----------------------------------------------------
            # Level-2 cluster vector IDs
            # -----------------------------------------------------
            lvl2_ids_offset = f.tell()
            f.write(lvl2_ids_flat.tobytes())

            # -----------------------------------------------------
            # Write Offsets
            # -----------------------------------------------------
            f.seek(4 * 4)  # after header
            f.write(struct.pack("IIII",
                                lvl1_centroids_offset,
                                lvl2_centroids_offset,
                                lvl2_lengths_offset,
                                lvl2_ids_offset))

        print("IVF index saved to", filename)

################################################################################

def cal_score(vec1, vec2):
    # Compute cosine similarity and return a plain Python float.
    # Guard against zero norms to avoid division-by-zero and return 0.0 in that case.
    dot_product = np.dot(vec1, vec2)
    # norm_vec1 = np.linalg.norm(vec1)
    # norm_vec2 = np.linalg.norm(vec2)
    return dot_product


####################### functions for load index data from file   ################

def load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
    with open(filename, "rb") as f:
        f.seek(centroid_offset)

        for start in range(0, n_clusters, batch_size):
            end = min(batch_size, n_clusters - start)
            bytes_to_read = end * dim * 4  # float32 size
            batch = np.frombuffer(f.read(bytes_to_read), dtype=np.float32)
            yield start, np.array(batch.reshape(end, dim))

def load_lvl2_centroids(filename, c, n_subclusters, dim, lvl2_centroids_offset):
    with open(filename, "rb") as f:
        # offset = base + c*(n_subclusters*dim*4)
        offset = lvl2_centroids_offset + c * (n_subclusters * dim * 4)
        f.seek(offset)
        data = np.frombuffer(f.read(n_subclusters * dim * 4), dtype=np.float32)
        return data.reshape(n_subclusters, dim)

def load_lvl2_subcluster_ids(filename, c, s, n_subclusters, lengths_array, lvl2_ids_offset):
    """
    lengths_array is length (n_clusters * n_subclusters)
    It is stored in order:
        [c0_s0, c0_s1, ..., c0_s(N-1), c1_s0, ..., c(K-1)_s(N-1)]
    """
    index = c * n_subclusters + s

    start = lengths_array[:index].sum().astype(np.uint32)
    length = lengths_array[index]

    with open(filename, "rb") as f:
        f.seek(lvl2_ids_offset + start * 4)
        data = np.frombuffer(f.read(length * 4), dtype=np.uint32)
        return data

####################### functions for processing search functions   ################

def get_nearest_centroids(filename, query_vector, n_probe, batch_size, n_clusters, dim, centroid_offset):
    centroid_scores_heap = []

    for start_idx, batch in load_centroids_batches(filename, batch_size, n_clusters, dim, centroid_offset):
        # batch shape: (batch_size, dim)
        scores = batch @ query_vector

        for i, score in enumerate(scores):
            item = (score, (start_idx + i))

            if len(centroid_scores_heap) < n_probe:
                heapq.heappush(centroid_scores_heap, item)
            else:
                # pushpop ensures only top n_probe remain
                if item > centroid_scores_heap[0]:
                    heapq.heappushpop(centroid_scores_heap, item)

    # After iterating all batches, extract the top n_probe centroid indices
    selected_centroids = [cid for _, cid in centroid_scores_heap]
    return selected_centroids

def get_nearest_k_vectors(vec_db, query_vector, all_vec_ids, k, batch_size):
    candidates = []

    # Process in batches to limit memory usage
    for start in range(0, len(all_vec_ids), batch_size):
        end = min(start + batch_size, len(all_vec_ids))
        vec_ids = all_vec_ids[start:end]

        vecs = vec_db.get_rows(vec_ids)

        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

        scores = vecs @ query_vector

        for vid, score in zip(vec_ids, scores):
            item = (score, vid)

            if len(candidates) < k:
                heapq.heappush(candidates, item)
            else:
                if item[0] > candidates[0][0] or (item[0] == candidates[0][0] and item[1] < candidates[0][1]):
                    heapq.heappushpop(candidates, item)

    return candidates


######################## Main search function ################

def search(vec_db, query_vector, k=5, 
           batch_size_for_centroids=512, batch_size_for_vectors=16):

    filename = vec_db.index_path
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-12)

    # ---- 1. Read header ----
    with open(filename, "rb") as f:
        n_clusters, n_probe, dim, n_subclusters = struct.unpack("IIII", f.read(16))

        lvl1_centroids_offset, lvl2_centroids_offset, \
        lvl2_lengths_offset, lvl2_ids_offset = struct.unpack("IIII", f.read(16))

        # read lvl2 lengths
        f.seek(lvl2_lengths_offset)
        lvl2_lengths = np.frombuffer(
            f.read(n_clusters * n_subclusters * 4), dtype=np.uint32
        )

    # ---- 2. Find nearest top-level centroids ----
    selected_lvl1 = get_nearest_centroids(
        filename, query_vector, n_probe,
        batch_size_for_centroids, n_clusters, dim,
        lvl1_centroids_offset
    )

    # ---- 3. From each selected L1 cluster pick nearest subcluster ----
    n_probe_sub = 3  # number of subclusters per L1 cluster to scan
    chosen_ids = []

    for c in selected_lvl1:
        # Load lvl2 centroids for this cluster
        sub_centroids = load_lvl2_centroids(
            filename, c, n_subclusters, dim, lvl2_centroids_offset
        )

        # scores for all 5 subclusters
        scores = sub_centroids @ query_vector

         # Pick top n_probe_sub subclusters
        top_s_idx = np.argpartition(-scores, n_probe_sub-1)[:n_probe_sub]

        for s in top_s_idx:
            ids = load_lvl2_subcluster_ids(filename, c, s, n_subclusters, lvl2_lengths, lvl2_ids_offset)
            chosen_ids.extend(ids)

    # ---- 4. sort ----
    chosen_ids = np.array(chosen_ids, dtype=np.uint32)
    chosen_ids.sort()

    # ---- 5. Score vectors ----
    candidates = get_nearest_k_vectors(
        vec_db, query_vector, chosen_ids,
        k, batch_size_for_vectors
    )

    # ---- 6. Sort final results ----
    results = sorted(candidates, key=lambda x: (-x[0], x[1]))

    return [vid for _, vid in results]


