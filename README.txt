Multimedia Retrieval assignment - code for Step 1 and Step 2
==============================================================

1. Put the shape database in the folder Data/ next to src/
   (one sub-folder per class, e.g. Data/Bicycle/D00016.obj).
   Download: https://github.com/MaxRee94/ShapeDatabase_INFOMR

2. Install Python 3.12 and the libraries:
       pip install -r requirements.txt

3. Step 1 - view a shape:
       python src/view_mesh.py Data/Bird/D00089.obj --wireframe
   (left-drag: rotate, scroll: zoom, ctrl+left-drag: pan, w: wireframe)
   The Plotly viewer and the database-wide load test are in
   report/step1_read_and_view.ipynb (outputs were cleared to keep the file small;
   run the notebook to recreate them).

4. Step 2 - analysis and resampling of the whole database:
       python src/run_step2.py --force               (about 7-8 minutes on 8 cores)
       python src/run_step2.py --force --skip-full   (only the reduced database, about 2 minutes)
   Results:
       report/data/step2/      csv files with the statistics
       processed/step2_full/   resampled database (input for Step 3)
   Figures and statistics for the report: report/step2_preprocessing.ipynb

Keep this folder structure: the code finds Data/ relative to src/.
