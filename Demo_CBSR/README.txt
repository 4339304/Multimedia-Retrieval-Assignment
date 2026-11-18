DEMO: CONTENT-BASED 3D SHAPE RETRIEVAL
======================================

What the demo does
------------------
Two shapes from outside the database run through the whole pipeline, one step at a time:

  Query 1: Jet m1161  (a good result: 8 of the 10 closest shapes are jets)
  Query 2: Car m1512  (a poor result: 1 car, otherwise motorcycles, guns, a piano)

  Step 1    read the OBJ file and show the shape
  Step 2    clean, reorient and resample (Jet: 1,486 -> 7,576 vertices; Car: already in range)
  Step 3.1  normalize: translation, PCA alignment, flipping, scaling (shown on the x, y, z axes)
  Step 3.2  global descriptors (printed) and the five histograms (chart)
  Step 4    the 10 closest of the 2,473 database shapes, as 3D pictures with their distances
  Step 5    the same query through the HNSW index (FAISS): same 10 shapes, search time

Numbers are printed in the terminal; every 3D view and chart opens as a page in the web browser
(the 3D views can be rotated and zoomed with the mouse). After each step the program waits until
Enter is pressed in the terminal. The whole run takes a few seconds of computing per query.


Requirements
------------
Python 3.12 (64-bit) from https://www.python.org . During installation on Windows, tick
"Add python.exe to PATH". Internet is needed once, for installing the packages.
Nothing else is needed: the two query files, the database features and the 20 result shapes
are included in this folder.


Windows
-------
1. Unzip the folder (not inside a OneDrive folder that is still syncing, if possible).
2. Double-click  setup_windows.bat   (once; installs the packages into a local .venv folder,
   takes a few minutes).
3. Double-click  run_demo_windows.bat   to start the demo.

Mac / Linux (in a terminal, inside this folder)
------------------------------------------------
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python demo.py


Options
-------
   python demo.py --only Jet        only the Jet query (or --only Car)
   python demo.py --no-pause        do not wait for Enter between the steps
   python demo.py --no-show         do not open browser pages (they are still saved in demo_output/)

On Windows, the options can be added after the file name, e.g. in a terminal:
   run_demo_windows.bat --only Car


Contents of the folder
----------------------
  demo.py                     the demo
  mr_io.py, mr_resample.py, mr_normalize.py, run_s3_2.py, run_step4.py, step5_common.py
                              the pipeline code of the project (unchanged copies)
  Data/                       the two original query files
  processed/step3_normalized/ the 20 database shapes that appear as results
  report/data/step4/          the normalized features of the 2,473 database shapes
  requirements.txt            the Python packages
