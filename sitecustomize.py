import os
import sys

# Set OpenMP duplicate library allowance before any C extensions load
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Pre-register torch DLL directory on Windows and initialize torch before pandas/pyarrow
if sys.platform == 'win32':
    _py_dir = os.path.dirname(sys.executable)
    _tlib = os.path.join(_py_dir, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(_tlib) and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_tlib)
        except Exception:
            pass
    try:
        import torch
    except Exception:
        pass
