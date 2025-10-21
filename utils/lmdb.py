import lmdb
import pickle
import os

def write_lmdb(lmdb_path):
    # Ensure parent directory exists when using subdir=False
    parent_dir = os.path.dirname(lmdb_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    env = lmdb.open(
        lmdb_path, 
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=60 * 1024 * 1024 * 1024
    )
    return env


def read_lmdb(lmdb_path, max_readers=32):
    env = lmdb.open(
        lmdb_path, 
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )
    return env

def get_data(txn, key):
    if isinstance(key, str):
        key = f"{key}".encode('ascii')
    value = txn.get(key)
    data = pickle.loads(value)
    return data

def get_all_keys(txn):
    keys = [bytes(key) for key, _ in txn.cursor()]
    return keys