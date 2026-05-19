import pickle
import numpy as np

with open('multi_peak_ft.pkl', 'rb') as f:
    d = pickle.load(f)

print("=== multi_peak_ft keys ===")
for key, val in d.items():
    print(f"\n--- {key} ---")
    if hasattr(val, 'shape'):
        print(f"  type:  {type(val).__name__}")
        print(f"  shape: {val.shape}")
        if hasattr(val, 'columns'):
            print(f"  columns: {list(val.columns)}")
        if hasattr(val, 'index'):
            print(f"  index[:5]: {list(val.index[:5])}")
        print(f"  head:\n{val[:5] if not hasattr(val, 'head') else val.head()}")
    elif isinstance(val, (int, float, str, bool)):
        print(f"  {val}")
    elif isinstance(val, (list, tuple)):
        print(f"  type:  {type(val).__name__}, len: {len(val)}")
        print(f"  first 5: {val[:5]}")
    else:
        print(f"  type: {type(val)}")
        print(f"  {val}")
