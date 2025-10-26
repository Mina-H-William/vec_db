from operator import index
from typing import Dict, List, Annotated
import numpy as np
import os
import hnswlib

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
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
        # return self.vectors
    
    # def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
    #     scores = []
    #     num_records = self._get_num_records()
    #     # here we assume that the row number is the ID of each vector
    #     for row_num in range(num_records):
    #         vector = self.get_one_row(row_num)
    #         score = self._cal_score(query, vector)
    #         scores.append((score, row_num))
    #     # here we assume that if two rows have the same score, return the lowest ID
    #     scores = sorted(scores, reverse=True)[:top_k]
    #     return [s[1] for s in scores]
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        # Normalize the query vector
        query = query.astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Set ef parameter
        self.index.set_ef(min(200, top_k * 40)) 
        
        # Query the index
        labels, distances = self.index.knn_query(query, k=top_k)
        
        # Return the vector IDs (labels)
        return labels[0].tolist()
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Get vectors
        vectors = self.get_all_rows().astype(np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        # Initialize HNSW index with cosine space
        self.index = hnswlib.Index(space='cosine', dim=DIMENSION)
        
        # Set parameters
        num_records = self._get_num_records()
        
        # Configure index with parameters
        self.index.init_index(
            max_elements=num_records + 10000,  # Allow room for future inserts
            ef_construction=100,
            M=16  # Higher than default (16) for better accuracy
        )
        
        # Add all vectors to the index
        print(f"Building HNSW index for {num_records} vectors...")
        labels = np.arange(num_records)
        self.index.add_items(vectors, labels)
        
        # Set default ef (query time parameter)
        self.index.set_ef(100)
        
        # Save index to disk
        self.index.save_index(self.index_path)
        print(f"HNSW index built and saved to {self.index_path}")
            
