from operator import index
from typing import Dict, List, Annotated
import numpy as np
import os
from IVF import BasicIVFIndexer, search

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.db_size = db_size
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                self._build_index()
            else:
                self.generate_database(self.db_size)
    
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
        
    # def get_rows(self, row_nums) -> np.ndarray:
    #     start_offset = np.int64(row_nums[0]) * DIMENSION * ELEMENT_SIZE
    #     # Create memmap for the whole file (does NOT load all data)
    #     mmap_vectors = np.memmap(
    #         self.db_path,
    #         dtype=np.float32,
    #         mode='r',
    #         offset=start_offset,
    #         shape=(row_nums[-1] - row_nums[0] + 1, DIMENSION)
    #     )

    #     # Vectorized retrieval (loads only required rows)
    #     return np.array(mmap_vectors[row_nums - row_nums[0]])
    #     # return np.array(mmap_vectors[row_nums])

    def get_rows_efficient(self, row_nums):
        # Calculate density: ratio of requested rows to span
        span = row_nums[-1] - row_nums[0] + 1
        density = len(row_nums) / span
        
        # If density > 0.5 (more than half the rows needed), load contiguous block
        if density > 0.5:
            # DENSE: Load contiguous block (your current approach - FAST!)
            start_offset = np.int64(row_nums[0]) * DIMENSION * ELEMENT_SIZE
            mmap_vectors = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                offset=start_offset,
                shape=(span, DIMENSION)
            )
            return np.array(mmap_vectors[row_nums - row_nums[0]])
        
        else:
            # SPARSE: Load individually (safer for RAM)
            result = np.empty((len(row_nums), DIMENSION), dtype=np.float32)
            
            with open(self.db_path, 'rb') as f:
                for i, row_num in enumerate(row_nums):
                    offset = np.int64(row_num) * DIMENSION * ELEMENT_SIZE
                    f.seek(offset)
                    result[i] = np.frombuffer(
                        f.read(DIMENSION * ELEMENT_SIZE), 
                        dtype=np.float32
                    )
            
            return result

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        
        return search(self, query.ravel(), top_k)

    def _build_index(self):
        if os.path.exists(self.index_path):
                os.remove(self.index_path)
        
        n_clusters = self.db_size // 1000
        n_probe = 10 + (self.db_size // 1000000) * 2

        self.ivf = BasicIVFIndexer(n_clusters=n_clusters, n_probe=n_probe)
        vectors = self.get_all_rows()
        self.ivf.Build(vectors)
        self.ivf.write_index(self.index_path)
