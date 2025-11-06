docker compose run --rm trainer
[+] Running 1/1
 ✔ Container steam-recommender-pipeline-1  Started                                                                                        0.3s 

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.3.4 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/src/train/run_training.py", line 9, in <module>
    from .build_faiss import run as build_faiss_run
  File "/app/src/train/build_faiss.py", line 4, in <module>
    import faiss
  File "/usr/local/lib/python3.11/site-packages/faiss/__init__.py", line 16, in <module>
    from .loader import *
  File "/usr/local/lib/python3.11/site-packages/faiss/loader.py", line 87, in <module>
    from .swigfaiss_avx2 import *
  File "/usr/local/lib/python3.11/site-packages/faiss/swigfaiss_avx2.py", line 10, in <module>
    from . import _swigfaiss_avx2
AttributeError: _ARRAY_API not found

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.3.4 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/src/train/run_training.py", line 9, in <module>
    from .build_faiss import run as build_faiss_run
  File "/app/src/train/build_faiss.py", line 4, in <module>
    import faiss
  File "/usr/local/lib/python3.11/site-packages/faiss/__init__.py", line 16, in <module>
    from .loader import *
  File "/usr/local/lib/python3.11/site-packages/faiss/loader.py", line 98, in <module>
    from .swigfaiss import *
  File "/usr/local/lib/python3.11/site-packages/faiss/swigfaiss.py", line 10, in <module>
    from . import _swigfaiss
AttributeError: _ARRAY_API not found
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/src/train/run_training.py", line 9, in <module>
    from .build_faiss import run as build_faiss_run
  File "/app/src/train/build_faiss.py", line 4, in <module>
    import faiss
  File "/usr/local/lib/python3.11/site-packages/faiss/__init__.py", line 16, in <module>
    from .loader import *
  File "/usr/local/lib/python3.11/site-packages/faiss/loader.py", line 98, in <module>
    from .swigfaiss import *
  File "/usr/local/lib/python3.11/site-packages/faiss/swigfaiss.py", line 10, in <module>
    from . import _swigfaiss
ImportError: numpy.core.multiarray failed to import