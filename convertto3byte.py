import struct
import numpy as np

def convert_ids_to_3byte(old_file, new_file):
    with open(old_file, "rb") as f:
        # --- 1. Read header ---
        n_clusters, n_probe, dim = struct.unpack("III", f.read(12))

        # 3 offsets (uint32)
        centroid_offset, lengths_offset, ids_offset = struct.unpack("III", f.read(12))

        # --- 2. Read centroids ---
        f.seek(centroid_offset)
        # number of floats = n_clusters * dim
        num_floats = n_clusters * dim
        centroids = np.frombuffer(f.read(num_floats * 4), dtype=np.float32)

        # --- 3. Read vector_ids lengths ---
        f.seek(lengths_offset)
        lengths = np.frombuffer(
            f.read(n_clusters * 4), dtype=np.uint32
        )  # uint32 length per cluster

        # --- 4. Read the flat vector_ids (currently uint32) ---
        f.seek(ids_offset)
        total_ids = lengths.sum()
        vector_ids = np.frombuffer(
            f.read(total_ids * 4), dtype=np.uint32
        )

    # ==========================================================
    #  Convert IDs → 3-byte binary
    # ==========================================================

    # Each ID = 3 bytes little-endian
    def to_uint24_bytes(arr):
        out = bytearray()
        for x in arr:
            out.extend(struct.pack("<I", x)[:3])   # take only first 3 bytes
        return out

    vector_ids_3byte = to_uint24_bytes(vector_ids)

    # ==========================================================
    #  Re-write a NEW FILE with 3-byte IDs
    # ==========================================================

    with open(new_file, "wb") as f:
        # Header
        f.write(struct.pack("III", n_clusters, n_probe, dim))

        # Reserve offsets
        f.write(b"\x00" * 12)

        # 1. Centroids
        centroid_offset = f.tell()
        f.write(centroids.astype(np.float32).tobytes())

        # 2. Lengths
        lengths_offset = f.tell()
        f.write(lengths.astype(np.uint32).tobytes())

        # 3. New 3-byte IDs
        ids_offset = f.tell()
        f.write(vector_ids_3byte)

        # Write offsets
        f.seek(12)
        f.write(struct.pack("III", centroid_offset, lengths_offset, ids_offset))

    print(f"Success! New index with 3-byte IDs written to {new_file}")




if __name__ == "__main__":
    old_index_file = "index_20m.ivf"
    new_index_file = "index_20m_3byte.ivf"
    convert_ids_to_3byte(old_index_file, new_index_file)