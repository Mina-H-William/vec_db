from operator import index
from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
import heapq
import time

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        
        # IVF parameters - will be set based on database size
        self.nlist = None
        self.nprobe = None
        
        # PQ parameters
        self.m = None  # Number of subvectors
        self.sub_dim = None  # Dimensions per subvector
        
        # Metadata that will be loaded when needed
        self.index_metadata = None
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self._build_index()
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
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
        # Rebuild the index (handling insertions properly would be more complex)
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function only loads one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this loads all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """Retrieve top-k similar vectors using IVF+PQ with disk-based access"""
        # Get database size to determine parameters
        num_records = self._get_num_records()
        
        # Determine parameters based on database size
        nlist = self._get_nlist_for_size(num_records)
        nprobe = self._get_nprobe_for_size(num_records)
        m = self._get_m_for_size(num_records)
        sub_dim = DIMENSION // m
        
        # Load minimal required data from disk (meets no-caching requirement)
        cluster_centers = self._load_cluster_centers(nlist)
        offsets = self._load_offsets(nlist)
        codebooks = self._load_codebooks(m, sub_dim)
        
        # 1. Find nprobe closest clusters
        # Convert query to float32 and normalize for cosine similarity
        query = query.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        cluster_distances = []
        for i in range(nlist):
            # Normalize cluster center for cosine similarity
            center = cluster_centers[i]
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                center = center / center_norm
                
            # Calculate cosine similarity (higher is better)
            similarity = np.dot(query, center)
            cluster_distances.append((similarity, i))
        
        # Get the nprobe clusters with highest similarity
        closest_clusters = sorted(cluster_distances, reverse=True)[:nprobe]
        
        # 2. Precompute distance tables for PQ
        dist_tables = []
        for j in range(m):
            subvec = query[j*sub_dim:(j+1)*sub_dim]
            # Normalize subvector
            subvec_norm = np.linalg.norm(subvec)
            if subvec_norm > 0:
                subvec = subvec / subvec_norm
                
            # Distance from query subvector to all 256 centroids
            # For cosine similarity: distance = 1 - similarity
            similarities = np.dot(codebooks[j], subvec)
            # Convert to distances (higher similarity = lower distance)
            distances = 1 - similarities
            dist_tables.append(distances)
        
        # 3. Search vectors in selected clusters
        candidates = []  # Will store (similarity, vector_id)
        
        # Process each relevant cluster
        for sim, cluster_id in closest_clusters:
            # Get start and end positions for this cluster
            start_idx = int(offsets[cluster_id])
            end_idx = int(offsets[cluster_id + 1])
            num_vectors = end_idx - start_idx
            
            # Only process up to 1000 vectors per cluster to meet time constraints
            max_vectors = min(num_vectors, 1000)
            
            # Load quantized vectors indices for this cluster
            quantized_indices = self._load_quantized_indices(start_idx, start_idx + max_vectors, m)
            
            # Calculate approximate similarity for each vector in this cluster
            for i in range(max_vectors):
                vec_id = start_idx + i
                approx_sim = 0
                
                # Sum similarities from all subvectors
                for j in range(m):
                    approx_sim += (1 - dist_tables[j][quantized_indices[i, j]])
                
                # Keep top candidates using min-heap for efficiency
                if len(candidates) < top_k:
                    heapq.heappush(candidates, (approx_sim, vec_id))
                else:
                    # If better than the worst in our current top-k
                    if approx_sim > candidates[0][0]:
                        heapq.heapreplace(candidates, (approx_sim, vec_id))
        
        # 4. Return top-k results (sorted by similarity, highest first)
        results = [vec_id for _, vec_id in sorted(candidates, reverse=True)]
        return results[:top_k]
    
    def _cal_score(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)
        
        # Normalize vectors
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        
        if vec1_norm == 0 or vec2_norm == 0:
            return 0.0
            
        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm
        
        # Calculate cosine similarity
        return np.dot(vec1, vec2)
    
    def _get_nlist_for_size(self, size):
        """Determine optimal nlist based on database size"""
        if size <= 1_000_000:
            return 1000
        elif size <= 10_000_000:
            return 5000
        elif size <= 15_000_000:
            return 7500
        else:  # 20M
            return 10000
    
    def _get_nprobe_for_size(self, size):
        """Determine optimal nprobe based on database size and time constraints"""
        if size <= 1_000_000:
            return 50
        elif size <= 10_000_000:
            return 30
        elif size <= 15_000_000:
            return 25
        else:  # 20M
            return 20
    
    def _get_m_for_size(self, size):
        """Determine optimal m (subvectors) based on database size"""
        if size <= 1_000_000:
            return 5  # Higher accuracy
        elif size <= 10_000_000:
            return 8
        elif size <= 15_000_000:
            return 9
        else:  # 20M
            return 10  # Maximize compression
    
    def _build_index(self):
        """Build IVF+PQ index and save to disk"""
        num_records = self._get_num_records()
        
        # Determine parameters based on database size
        nlist = self._get_nlist_for_size(num_records)
        m = self._get_m_for_size(num_records)
        sub_dim = DIMENSION // m  # Should be 7 for m=10
        
        print(f"Building IVF+PQ index for {num_records} vectors...")
        print(f"Using parameters: nlist={nlist}, nprobe={self._get_nprobe_for_size(num_records)}, m={m}")
        
        # 1. Sample vectors for training
        sample_size = min(100_000, num_records)
        sample_indices = np.random.choice(num_records, sample_size, replace=False)
        
        # Load sample vectors
        sample_vectors = np.zeros((sample_size, DIMENSION), dtype=np.float32)
        for i, idx in enumerate(sample_indices):
            sample_vectors[i] = self.get_one_row(idx)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        sample_vectors = sample_vectors / np.where(norms > 0, norms, 1)
        
        # 2. Build IVF index (k-means clustering)
        print(f"Training IVF with {nlist} clusters...")
        kmeans = KMeans(n_clusters=nlist, n_init=1, random_state=42)
        kmeans.fit(sample_vectors)
        
        # Normalize cluster centers for cosine similarity
        cluster_centers = kmeans.cluster_centers_.copy()
        norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        cluster_centers = cluster_centers / np.where(norms > 0, norms, 1)
        
        # Save cluster centers
        cluster_centers_path = os.path.join(os.path.dirname(self.index_path), "cluster_centers.bin")
        with open(cluster_centers_path, 'wb') as f:
            np.array(cluster_centers, dtype=np.float32).tofile(f)
        
        # 3. Create inverted index
        print("Building inverted index...")
        # Process in batches to avoid memory issues
        batch_size = 10_000
        offsets = np.zeros(nlist + 1, dtype=np.int32)  # +1 for convenience
        
        # First pass: count vectors per cluster
        vectors_per_cluster = np.zeros(nlist, dtype=np.int32)
        for start in range(0, num_records, batch_size):
            end = min(start + batch_size, num_records)
            batch_vectors = np.zeros((end - start, DIMENSION), dtype=np.float32)
            
            for i in range(start, end):
                batch_vectors[i - start] = self.get_one_row(i)
            
            # Normalize batch vectors
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.where(norms > 0, norms, 1)
            
            assignments = kmeans.predict(batch_vectors)
            
            for cluster_id in assignments:
                vectors_per_cluster[cluster_id] += 1
        
        # Calculate offsets
        for i in range(1, nlist + 1):
            offsets[i] = offsets[i-1] + vectors_per_cluster[i-1]
        
        # Save offsets
        offsets_path = os.path.join(os.path.dirname(self.index_path), "offsets.bin")
        with open(offsets_path, 'wb') as f:
            np.array(offsets, dtype=np.int32).tofile(f)
        
        # Second pass: assign vectors to clusters
        vector_ids = np.zeros(num_records, dtype=np.int32)
        vectors_per_cluster = np.zeros(nlist, dtype=np.int32)
        
        for start in range(0, num_records, batch_size):
            end = min(start + batch_size, num_records)
            batch_vectors = np.zeros((end - start, DIMENSION), dtype=np.float32)
            
            for i in range(start, end):
                batch_vectors[i - start] = self.get_one_row(i)
            
            # Normalize batch vectors
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.where(norms > 0, norms, 1)
            
            assignments = kmeans.predict(batch_vectors)
            
            for i, cluster_id in enumerate(assignments):
                pos = offsets[cluster_id] + vectors_per_cluster[cluster_id]
                vector_ids[pos] = start + i
                vectors_per_cluster[cluster_id] += 1
        
        # Save vector IDs
        vector_ids_path = os.path.join(os.path.dirname(self.index_path), "vector_ids.bin")
        with open(vector_ids_path, 'wb') as f:
            np.array(vector_ids, dtype=np.int32).tofile(f)
        
        # 4. Build PQ codebooks
        print(f"Training PQ with {m} subvectors...")
        codebooks = []
        for i in range(m):
            # Extract subvectors
            subvectors = sample_vectors[:, i*sub_dim:(i+1)*sub_dim]
            
            # Train k-means with 256 centroids (1 byte index)
            kmeans_pq = KMeans(n_clusters=256, n_init=1, random_state=42)
            kmeans_pq.fit(subvectors)
            
            # Normalize centroids for cosine similarity
            centroids = kmeans_pq.cluster_centers_.copy()
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / np.where(norms > 0, norms, 1)
            
            codebooks.append(centroids)
        
        # Save codebooks
        codebooks_path = os.path.join(os.path.dirname(self.index_path), "codebooks.bin")
        with open(codebooks_path, 'wb') as f:
            for i in range(m):
                np.array(codebooks[i], dtype=np.float32).tofile(f)
        
        # 5. Quantize all vectors
        print("Quantizing vectors...")
        quantized_vectors = np.zeros((num_records, m), dtype=np.uint8)
        
        for start in range(0, num_records, batch_size):
            end = min(start + batch_size, num_records)
            batch_vectors = np.zeros((end - start, DIMENSION), dtype=np.float32)
            
            for i in range(start, end):
                batch_vectors[i - start] = self.get_one_row(i)
            
            # Normalize batch vectors
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.where(norms > 0, norms, 1)
            
            for i, vec in enumerate(batch_vectors):
                vec_id = start + i
                for j in range(m):
                    subvec = vec[j*sub_dim:(j+1)*sub_dim]
                    # Calculate similarities to all centroids
                    similarities = np.dot(codebooks[j], subvec)
                    # Find centroid with highest similarity
                    best_centroid = np.argmax(similarities)
                    quantized_vectors[vec_id, j] = best_centroid
        
        # Save quantized vectors
        quantized_path = os.path.join(os.path.dirname(self.index_path), "quantized.bin")
        with open(quantized_path, 'wb') as f:
            quantized_vectors.tofile(f)
        
        # Save metadata
        metadata = {
            'nlist': nlist,
            'm': m,
            'sub_dim': sub_dim,
            'num_records': num_records
        }
        metadata_path = os.path.join(os.path.dirname(self.index_path), "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"IVF+PQ index built successfully!")
        print(f"Index size: {self._get_index_size():.2f} MB")
    
    def _load_cluster_centers(self, nlist):
        """Load cluster centers from disk"""
        cluster_centers_path = os.path.join(os.path.dirname(self.index_path), "cluster_centers.bin")
        return np.memmap(cluster_centers_path, dtype=np.float32, mode='r', 
                         shape=(nlist, DIMENSION))
    
    def _load_offsets(self, nlist):
        """Load offsets from disk"""
        offsets_path = os.path.join(os.path.dirname(self.index_path), "offsets.bin")
        return np.memmap(offsets_path, dtype=np.int32, mode='r', 
                         shape=(nlist + 1))
    
    def _load_codebooks(self, m, sub_dim):
        """Load codebooks from disk"""
        codebooks_path = os.path.join(os.path.dirname(self.index_path), "codebooks.bin")
        return np.memmap(codebooks_path, dtype=np.float32, mode='r', 
                         shape=(m, 256, sub_dim))
    
    def _load_quantized_indices(self, start_idx, end_idx, m):
        """Load quantized vector indices from disk"""
        quantized_path = os.path.join(os.path.dirname(self.index_path), "quantized.bin")
        return np.memmap(quantized_path, dtype=np.uint8, mode='r', 
                         offset=start_idx * m, 
                         shape=(end_idx - start_idx, m))
    
    def _get_index_size(self):
        """Calculate total index size in MB"""
        total_size = 0
        index_dir = os.path.dirname(self.index_path)
        
        for filename in os.listdir(index_dir):
            if filename.endswith('.bin') or filename.endswith('.pkl'):
                file_path = os.path.join(index_dir, filename)
                total_size += os.path.getsize(file_path)
                
        return total_size / (1024 * 1024)  # Convert to MB