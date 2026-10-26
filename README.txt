Multimedia Retrieval assignment - code for Steps 1 to 4
========================================================

1. Put the shape database in the folder Data/ next to the .py files
   (one sub-folder per class, e.g. Data/Bicycle/D00016.obj).
   Download: https://github.com/MaxRee94/ShapeDatabase_INFOMR

2. Install Python 3.12 and the libraries:
       pip install -r requirements.txt

3. Step 1 - view a shape:
       python view_mesh.py Data/Bird/D00089.obj --wireframe
   (left-drag: rotate, scroll: zoom, ctrl+left-drag: pan, w: wireframe)
   The Plotly viewer and the database-wide load test are in step1_read_and_view.ipynb.

4. Step 2 - analysis, resampling, checking and normalization (run in this order):
       python run_step2.py --force        2.1-2.3: statistics and resampling (about 7 minutes)
       python check_resampling.py         2.4: check of the resampled database (about 3 minutes)
       python normalize_shapes.py         2.5: centering and scaling (about 8 minutes)
   Figures and statistics for the report: step2_preprocessing.ipynb

5. Step 3.1 - full normalization (alignment and flipping), needs the output of 2.5:
       python run_step3_1.py --force      (about 4 minutes)
       python figures_step3_1.py          figures for the report

6. Step 3.2 - shape descriptors, needs the output of 3.1 (8 GB memory is enough):
       python validate_step3_2.py         check on a cube, sphere, box and open box
       python run_s3_2.py                 all shapes (about 16 minutes on 8 cores)
       python run_s3_2.py --shape processed/step3_normalized/Mug/D00685.obj
                                          print the features of one shape
       python figures_step3_2.py          figures for the report
       python figures_step3_2.py --ranges --bins
                                          histogram ranges and choice of the number of bins (about 8 minutes)

7. Step 4 - querying (normalization and Euclidean distance), needs the feature table of 3.2:
       python run_step4.py --build        normalize the features; 10 query shapes are taken out of the database
       python run_step4.py --show         print the normalized feature table
       python run_step4.py --query Data/Mug/D00172.obj -k 5
                                          query with any OBJ file (processed like the database)
       python run_step4.py --query Mug/D00685.obj -k 5
                                          query with a shape of the database (class/file)
       python run_step4.py --compare      compare the normalization and weighting options (about 1 minute)
       python figures_step4.py            figures for the report

8. Step 5 - approximate nearest-neighbour search and t-SNE visualization:

       python run_step5.py --build
              build the HNSW approximate nearest-neighbour index

       python run_step5.py --query Mug/D00685.obj --compare -k 10
              query the ANN index and compare the results with Step 4

       python run_step5.py --query HumanHead/D00591.obj --compare -k 10
       python run_step5.py --query Computer/m1778.obj --compare -k 10
       python run_step5.py --query Quadruped/D00795.obj --compare -k 10
       python run_step5.py --query Hand/D01022.obj --compare -k 10
              test five query shapes and record the ANN search time,
              exact search time and recall@10

       python run_step5.py --query Mug/D00685.obj --radius 2.7 --max-candidates 2473
              perform a radius-based query for the Mug shape

       python run_step5_tsne.py --perplexity 30 --iterations 1000 --seed 42
              generate the two-dimensional t-SNE visualization

       Results and index files are saved in:
       report/data/step4/step5/

       t-SNE screenshot:
       report/data/step4/3d-shape-feature-database-t-sne-2d-embed.png

       Validation logs and screenshot:
       report/data/step4/step5/validation/

Results:
   processed/step2_full/                        resampled shapes (Step 2.3)
   processed/step2_normalized/                  centered and scaled shapes (Step 2.5)
   processed/step3_normalized/                  fully normalized shapes (Step 3.1, input for Step 3.2)
   report/data/step3/features_step3_2.csv       feature table (Step 3.2, one row per shape)
   report/data/step4/features_normalized.csv    normalized feature table of the database (Step 4)
   report/data/step4/query_shapes.csv           shapes taken out of the database to be used as queries (Step 4)
   report/data/                                 other statistics (csv)
   report/figures/                              figures (copied to Images/ for the report)
