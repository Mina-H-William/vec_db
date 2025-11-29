from operator import index
from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
import heapq
import time
import logging
import struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        
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
        # Rebuild the index
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            logger.error(f"Error loading row {row_num}: {e}")
            return np.zeros(DIMENSION, dtype=np.float32)

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """Retrieve top-k similar vectors using IVF+PQ with disk-based access"""
        # Get database size to determine parameters
        num_records = self._get_num_records()
        
        # Determine parameters based on database size
        nlist = self._get_nlist_for_size(num_records)
        nprobe = self._get_nprobe_for_size(num_records)
        m = self._get_m_for_size(num_records)
        sub_dim = DIMENSION // m
        
        # Check if index file exists
        if not os.path.exists(self.index_path):
            logger.warning("Index file not found, falling back to brute-force search")
            return self._brute_force_retrieve(query, top_k)
        
        # Load minimal required data from disk
        try:
            # Only load metadata first to get index structure
            metadata = self._load_metadata()
            nlist = metadata['nlist']
            m = metadata['m']
            sub_dim = metadata['sub_dim']
            
            # Now load the specific components we need
            cluster_centers = self._load_cluster_centers(nlist)
            offsets = self._load_offsets(nlist)
            codebooks = self._load_codebooks(m, sub_dim)
        except Exception as e:
            logger.error(f"Failed to load index components: {e}")
            return self._brute_force_retrieve(query, top_k)
        
        # 1. Find nprobe closest clusters
        # Convert query to float32 and normalize for cosine similarity
        query = query.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Calculate similarity to all cluster centers
        cluster_distances = []
        for i in range(nlist):
            # Normalize cluster center
            center = cluster_centers[i]
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                center = center / center_norm
            
            # Calculate cosine similarity
            similarity = np.dot(query, center)
            cluster_distances.append((similarity, i))
        
        # Get the nprobe clusters with highest similarity
        closest_clusters = sorted(cluster_distances, reverse=True)[:nprobe]
        
        # 2. Precompute SIMILARITY tables (CRITICAL FIX)
        sim_tables = []
        for j in range(m):
            subvec = query[j*sub_dim:(j+1)*sub_dim]
            # Normalize subvector
            subvec_norm = np.linalg.norm(subvec)
            if subvec_norm > 0:
                subvec = subvec / subvec_norm
            
            # Calculate similarities (dot product = cosine similarity for normalized vectors)
            similarities = np.dot(codebooks[j], subvec)
            sim_tables.append(similarities)
        
        # 3. Search vectors in selected clusters
        candidates = []  # Will store (similarity, vector_id)
        
        # Process each relevant cluster
        for sim, cluster_id in closest_clusters:
            # Get start and end positions for this cluster
            start_idx = int(offsets[cluster_id])
            end_idx = int(offsets[cluster_id + 1])
            num_vectors = end_idx - start_idx
            
            # Only process up to 1000 vectors per cluster
            max_vectors = min(num_vectors, 1000)
            
            # Skip empty clusters
            if max_vectors <= 0:
                continue
            
            # Load quantized vectors indices for this cluster
            try:
                quantized_indices = self._load_quantized_indices(start_idx, start_idx + max_vectors, m)
            except Exception as e:
                logger.error(f"Failed to load quantized indices for cluster {cluster_id}: {e}")
                continue
            
            # Calculate APPROXIMATE SIMILARITY for each vector in this cluster
            for i in range(max_vectors):
                vec_id = start_idx + i
                approx_sim = 0
                
                # Sum similarities from all subvectors (CORRECTED)
                for j in range(m):
                    approx_sim += sim_tables[j][quantized_indices[i, j]]
                
                # Normalize by number of subvectors to get proper range [0, m]
                # This is critical for proper ranking
                approx_sim = approx_sim / m
                
                # Keep top candidates using min-heap for efficiency
                if len(candidates) < top_k * 3:  # Keep more candidates for refinement
                    heapq.heappush(candidates, (approx_sim, vec_id))
                else:
                    # If better than the worst in our current top-k*3
                    if approx_sim > candidates[0][0]:
                        heapq.heapreplace(candidates, (approx_sim, vec_id))
        
        # 4. REFINEMENT STEP: Calculate exact similarity for top candidates
        # This dramatically improves accuracy with minimal time cost
        refined_candidates = []
        for sim, vec_id in candidates:
            # Get the actual vector
            vector = self.get_one_row(vec_id)
            
            # Calculate exact cosine similarity
            exact_sim = self._cal_score(query, vector)
            
            refined_candidates.append((exact_sim, vec_id))
        
        # Sort by exact similarity and return top-k
        refined_candidates.sort(reverse=True)
        return [vec_id for _, vec_id in refined_candidates[:top_k]]
    
    def _brute_force_retrieve(self, query: np.ndarray, top_k: int) -> List[int]:
        """Fallback method if index is not built properly"""
        logger.warning("Falling back to brute-force search")
        num_records = self._get_num_records()
        
        # Normalize query
        query = query.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Use a heap to efficiently track top-k results
        top_k_heap = []
        
        # Process in batches to minimize memory usage
        batch_size = 1000
        for start in range(0, num_records, batch_size):
            end = min(start + batch_size, num_records)
            batch_vectors = np.zeros((end - start, DIMENSION), dtype=np.float32)
            
            for i in range(start, end):
                batch_vectors[i - start] = self.get_one_row(i)
            
            # Normalize batch vectors
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.where(norms > 0, norms, 1)
            
            # Calculate similarities
            similarities = np.dot(batch_vectors, query)
            
            # Update top-k heap
            for i in range(len(similarities)):
                vec_id = start + i
                sim = similarities[i]
                
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (sim, vec_id))
                else:
                    if sim > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (sim, vec_id))
        
        # Return results sorted by similarity (highest first)
        return [vec_id for _, vec_id in sorted(top_k_heap, reverse=True)]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

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
        """Build IVF+PQ index and save to a single file"""
        num_records = self._get_num_records()
        
        # Determine parameters based on database size
        nlist = self._get_nlist_for_size(num_records)
        m = self._get_m_for_size(num_records)
        sub_dim = DIMENSION // m  # Should be 7 for m=10
        
        logger.info(f"Building IVF+PQ index for {num_records} vectors...")
        logger.info(f"Using parameters: nlist={nlist}, nprobe={self._get_nprobe_for_size(num_records)}, m={m}")
        
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
        logger.info(f"Training IVF with {nlist} clusters...")
        kmeans = KMeans(n_clusters=nlist, n_init=1, random_state=42)
        kmeans.fit(sample_vectors)
        
        # Normalize cluster centers for cosine similarity
        cluster_centers = kmeans.cluster_centers_.copy()
        norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        cluster_centers = cluster_centers / np.where(norms > 0, norms, 1)
        
        # 3. Create inverted index
        logger.info("Building inverted index...")
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
        
        # 4. Build PQ codebooks
        logger.info(f"Training PQ with {m} subvectors...")
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
        
        # 5. Quantize all vectors
        logger.info("Quantizing vectors...")
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
        
        # 6. Save everything to a single index file
        logger.info(f"Saving index to {self.index_path}")
        
        # Create a dictionary with all index components
        index_data = {
            'metadata': {
                'nlist': nlist,
                'm': m,
                'sub_dim': sub_dim,
                'num_records': num_records
            },
            'cluster_centers': cluster_centers,
            'offsets': offsets,
            'codebooks': codebooks,
            'quantized_vectors': quantized_vectors
        }
        
        # Save to a single file using pickle
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Log index size information
        index_size = os.path.getsize(self.index_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"IVF+PQ index built successfully!")
        logger.info(f"Index size: {index_size:.2f} MB")
    
    def _load_metadata(self):
        """Load only the metadata from the index file"""
        with open(self.index_path, 'rb') as f:
            # Load only the metadata part
            index_data = pickle.load(f)
            return index_data['metadata']
    
    def _load_cluster_centers(self, nlist):
        """Load cluster centers from the index file"""
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            return index_data['cluster_centers']
    
    def _load_offsets(self, nlist):
        """Load offsets from the index file"""
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            return index_data['offsets']
    
    def _load_codebooks(self, m, sub_dim):
        """Load codebooks from the index file"""
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            return index_data['codebooks']
    
    def _load_quantized_indices(self, start_idx, end_idx, m):
        """Load quantized vector indices from the index file"""
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            return index_data['quantized_vectors'][start_idx:end_idx]