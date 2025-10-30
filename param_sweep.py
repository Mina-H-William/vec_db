import numpy as np
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
import os
import itertools
from tqdm import tqdm
import json

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        
        tic = time.time()
        db_ids = db.retrieve(query, top_k)
        toc = time.time()
        run_time = toc - tic
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval(results: List[Result]):
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append(-1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_index_size(index_path):
    """Get total size of index files in MB"""
    total = 0
    for ext in ['', '.centroids', '.clusters']:
        path = index_path + ext
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total / (1024 * 1024)  # Convert to MB

def evaluate_params(size_m, params):
    """Evaluate a single parameter combination"""
    try:
        db = VecDB(
            db_size=int(size_m * 10**6),
            database_file_path=f'saved_db_{size_m}m.dat',
            index_file_path=f'index_{size_m}m.dat',
            new_db=True,
            use_ivf=True,
            ivf_clusters=params['ivf_clusters'],
            ivf_probe=params['ivf_probe'],
            ivf_iters=params['ivf_iters'],
            ivf_batch=params['ivf_batch']
        )
        # Override LSH params
        db.L = params['L']
        db.K = params['K']
        db.num_probes = params['num_probes']
        
        # Run evaluation
        all_db = db.get_all_rows()
        results = run_queries(db, all_db, 5, 10)
        score, query_time = eval(results)
        
        # Get index size before cleanup
        index_size = get_index_size(f'index_{size_m}m.dat')
        
        # Clean up memory maps
        if hasattr(db, 'cluster_assignments'):
            db.cluster_assignments._mmap.close()
            db.cluster_assignments = None
            
        # Delete files
        for ext in ['', '.centroids', '.clusters']:
            path = f'index_{size_m}m.dat{ext}'
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        if os.path.exists(f'saved_db_{size_m}m.dat'):
            try:
                os.remove(f'saved_db_{size_m}m.dat')
            except:
                pass
        
        return dict(
            params=params,
            score=float(score),
            query_time=float(query_time),
            index_size=float(index_size)
        )
    except Exception as e:
        print(f"Error with params {params}: {e}")
        # Cleanup on error
        for ext in ['', '.centroids', '.clusters']:
            path = f'index_{size_m}m.dat{ext}'
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        if os.path.exists(f'saved_db_{size_m}m.dat'):
            try:
                os.remove(f'saved_db_{size_m}m.dat')
            except:
                pass
        raise
# Parameter grid optimized for smaller index size
param_grid = {
    'ivf_clusters': [64, 128, 256],  # Fewer clusters
    'ivf_probe': [2, 4, 8],
    'ivf_iters': [10],  # Fixed since impact on query time is minimal
    'ivf_batch': [10000],  # Fixed
    'L': [4, 5, 6],  # Fewer hash tables
    'K': [10, 12, 14],  # Fewer bits per table
    'num_probes': [2, 4, 6]  # Multi-probe LSH
}

def sweep_params(size_m=1):
    """Run parameter sweep and save results"""
    results = []
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    for combo in tqdm(combinations):
        params = dict(zip(keys, combo))
        try:
            result = evaluate_params(size_m, params)
            results.append(result)
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
            
        # Save intermediate results
        with open(f'sweep_results_{size_m}m.json', 'w') as f:
            json.dump(results, f)
    
    return results

def analyze_results(results):
    """Analyze and print parameter sweep results"""
    if not results:
        print("No results to analyze")
        return
        
    # Sort by different metrics
    by_score = sorted(results, key=lambda x: x['score'], reverse=True)
    by_time = sorted(results, key=lambda x: x['query_time'])
    by_size = sorted(results, key=lambda x: x['index_size'])
    
    print("\nTop 5 by score (higher is better):")
    for r in by_score[:5]:
        print(f"Score: {r['score']:.1f}, Time: {r['query_time']:.3f}s, Size: {r['index_size']:.1f}MB")
        print(f"Params: {r['params']}\n")
        
    print("\nTop 5 by query time (lower is better):")
    for r in by_time[:5]:
        print(f"Time: {r['query_time']:.3f}s, Score: {r['score']:.1f}, Size: {r['index_size']:.1f}MB")
        print(f"Params: {r['params']}\n")
        
    print("\nTop 5 by index size (lower is better):")
    for r in by_size[:5]:
        print(f"Size: {r['index_size']:.1f}MB, Score: {r['score']:.1f}, Time: {r['query_time']:.3f}s")
        print(f"Params: {r['params']}\n")
    
    # Find balanced configurations
    print("\nMost balanced configurations:")
    # Normalize metrics to 0-1 range
    score_range = by_score[0]['score'] - by_score[-1]['score']
    time_range = by_time[-1]['query_time'] - by_time[0]['query_time']
    size_range = by_size[-1]['index_size'] - by_size[0]['index_size']
    
    balanced = []
    for r in results:
        # Convert score to 0-1 (1 is best)
        norm_score = (r['score'] - by_score[-1]['score']) / score_range if score_range > 0 else 0.5
        # Convert time to 0-1 (1 is best = fastest)
        norm_time = (by_time[-1]['query_time'] - r['query_time']) / time_range if time_range > 0 else 0.5
        # Convert size to 0-1 (1 is best = smallest)
        norm_size = (by_size[-1]['index_size'] - r['index_size']) / size_range if size_range > 0 else 0.5
        
        # Compute balanced score (geometric mean to penalize poor performance in any dimension)
        balanced_score = (norm_score * norm_time * norm_size) ** (1/3)
        balanced.append((balanced_score, r))
    
    balanced.sort(reverse=True)
    print("\nTop 3 balanced configurations:")
    for score, r in balanced[:3]:
        print(f"Balanced score: {score:.3f}")
        print(f"Score: {r['score']:.1f}, Time: {r['query_time']:.3f}s, Size: {r['index_size']:.1f}MB")
        print(f"Params: {r['params']}\n")
        
if __name__ == "__main__":
    # Run sweep for 1M vectors
    try:
        results = sweep_params(1)
        analyze_results(results)
    except KeyboardInterrupt:
        print("\nSweep interrupted")
        # Try to load partial results
        try:
            with open('sweep_results_1m.json', 'r') as f:
                results = json.load(f)
            print("\nAnalyzing partial results:")
            analyze_results(results)
        except:
            print("Could not load partial results")