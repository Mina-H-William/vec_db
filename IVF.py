import numpy as np
import heapq
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class BasicIVFIndexer:
    def __init__(self, n_clusters=1000, n_probe=10):
        self.n_clusters = n_clusters
        self.n_probe = n_probe  # Number of clusters to search
        self.centroids = None
        self.vector_ids = None
        
    # Build function 
    def Build(self, vectors, vector_ids=None):
        """K-means clustering with cosine similarity (spherical k-means)"""
        print("Building IVF index...")
        
        if vector_ids is None:
            vector_ids = np.arange(len(vectors))
        
        # Normalize vectors to unit length
        self.normalized_vectors = normalize(vectors, axis=1, norm='l2')
        
        # KMeans on normalized vectors = spherical k-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        cluster_labels = self.kmeans.fit_predict(self.normalized_vectors)
        
        # Centroids are already normalized directions
        self.centroids = self.kmeans.cluster_centers_
        
        # Organize vector IDs by cluster
        self.vector_ids = [[] for _ in range(self.n_clusters)]
        for vector_id, cluster_idx in zip(vector_ids, cluster_labels):
            self.vector_ids[cluster_idx].append(vector_id)
        
        print("IVF index built successfully.")

    def write_index(self, filename: str):
        """
        Save the IVF index to a file using numpy.savez
        More efficient for large arrays
        """
        try:
            # Convert list of lists to a format that numpy can save efficiently
            vector_ids_lengths = [len(lst) for lst in self.vector_ids]
            vector_ids_flat = np.concatenate(self.vector_ids) if any(self.vector_ids) else np.array([], dtype=int)
            
            np.savez(filename,
                    n_clusters=self.n_clusters,
                    n_probe=self.n_probe,
                    centroids=self.centroids,
                    vector_ids_lengths=vector_ids_lengths,
                    vector_ids_flat=vector_ids_flat)
            print(f"Index successfully written to {filename}")
        except Exception as e:
            print(f"Error writing index to file: {e}")

    @classmethod
    def read_index(cls, filename: str):
        """
        Load the IVF index from a numpy .npz file
        """
        try:
            data = np.load(filename, allow_pickle=True)
            
            indexer = cls(n_clusters=int(data['n_clusters']), n_probe=int(data['n_probe']))
            indexer.centroids = data['centroids']
            
            # Reconstruct the list of lists from flat array
            lengths = data['vector_ids_lengths']
            flat = data['vector_ids_flat']
            indexer.vector_ids = []
            start = 0
            for length in lengths:
                indexer.vector_ids.append(flat[start:start+length].tolist())
                start += length
                
            print(f"Index successfully loaded from {filename}")
            return indexer
        except Exception as e:
            print(f"Error loading index from file: {e}")
            return None
    


################################################################################

def cal_score(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity


def search(IVF: BasicIVFIndexer, vec_db, query_vector, k=5):
    """Search for k nearest neighbors"""
    # Find nearest centroids to query
    scores_to_centroids = [cal_score(query_vector, centroid) for centroid in IVF.centroids]
    nearest_centroid_indices = np.argsort(scores_to_centroids)[-IVF.n_probe:][::-1]

    # Search in selected clusters
    candidates = []
    for centroid_idx in nearest_centroid_indices:
        cluster_ids = IVF.vector_ids[centroid_idx]

        for vec_id in cluster_ids:
            vec = vec_db.get_one_row(vec_id)
            score = cal_score(query_vector, vec)

            heap_item = (score, -vec_id)
            
            if len(candidates) < k:
                heapq.heappush(candidates, heap_item)
            else:
                # Push if this score is higher than our current smallest in top-k
                # OR if score equal but lower vec_id
                if heap_item > candidates[0]:
                    heapq.heappushpop(candidates, heap_item)


    results = [(score, -vec_id) for score, vec_id in candidates]
    results.sort(key=lambda x: (x[0], x[1]))  # Sort by score ascending, then ID ascending
    return [idx for _, idx in results]