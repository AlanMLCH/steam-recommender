Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'game_title'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/src/pipeline.py", line 26, in <module>
    main()
  File "/app/src/pipeline.py", line 23, in main
    run_step(args.step)
  File "/app/src/pipeline.py", line 17, in run_step
    extract.run()
  File "/app/src/extract.py", line 107, in run
    raw = normalize_titles(raw)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/extract.py", line 34, in normalize_titles
    out["game_title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    ~~~^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'game_title'