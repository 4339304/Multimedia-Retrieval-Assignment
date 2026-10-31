Multimedia Retrieval assignment - code for Steps 1 to 6
======================================================

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
       python run_step2.py --force        2.1-2.3: statistics and resampling (about 6 minutes)
       python check_resampling.py         2.4: check of the resampled database (about 3 minutes)
       python normalize_shapes.py         2.5: centering and scaling (about 8 minutes)
       python figures_step2_2.py          2.2: histograms of the vertex and face counts, class sizes, smallest/median/largest shapes
       python figures_step2_3.py          large figures of the decimation cases and orientation conflicts

5. Step 3.1 - full normalization (translation, alignment, flipping, scaling), starts from the output of 2.3:
       python run_step3_1.py --force      (about 6 minutes with 4 worker processes)
       python figures_step3_1.py          figures for the report

6. Step 3.2 - shape descriptors, needs the output of 3.1 (8 GB memory is enough):
       python validate_step3_2.py         check on a cube, sphere, box and open box
       python run_s3_2.py                 all shapes (about 17 minutes with 6 worker processes)
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
       python run_step4_metrics.py --compare
                                          Euclidean, cosine and EMD for the histograms, every database shape as a query
       python figures_step4.py            figures for the report

8. Step 5 - approximate nearest-neighbor search and t-SNE map, needs the normalized features of Step 4:
       python run_step5.py --build        build the HNSW index of all feature vectors (optional: a query
                                          builds it by itself if it is missing or older than the features)
       python run_step5.py --query Mug/D00685.obj -k 10
                                          the K nearest shapes (also a mesh file from outside the database)
       python run_step5.py --query Mug/D00685.obj --radius 2.0 --max-candidates 2473
                                          all shapes within a radius of the query
       python run_step5.py --query Mug/D00685.obj --compare -k 10
                                          search time and recall of the index against an exact search
       python run_step5.py --evaluate-all -k 10
                                          the same for every database shape as a query
       python evaluate_radius.py          radius queries of the index against an exact search
       python run_step5_tsne.py --perplexity 30
       python run_step5_tsne.py --perplexity 40
                                          t-SNE maps (interactive HTML, hover for the shape)
       python figures_step5.py            figures for the report (t-SNE maps, query results of the index)
   Results: report/data/step5/ (index, timing and recall, radius tests, t-SNE positions and plots)

9. Step 6 - evaluation, every database shape once as a query (uses the index of Step 5):
       python run_step6.py                precision of the 10 closest shapes, per class and in total (about 20 s)
       python figures_step6.py            figures for the report (per-class precision, precision for K, examples)
   Results: report/data/step6/ (per query, per class, summary, precision and recall for K = 1 .. 50)

Results stored in this repository:
   report/data/step3/features_step3_2.csv       feature table (Step 3.2, one row per shape)
   report/data/step4/features_normalized.csv    normalized feature table of the database (Step 4)
   report/data/step4/query_shapes.csv           shapes taken out of the database to be used as queries (Step 4)
   report/data/                                 other statistics and results of Steps 2 to 6 (csv; t-SNE maps as html)
   Images/                                      figures of the report

Results written by the scripts, not stored in this repository (too large):
   processed/step2_full/                        resampled shapes (Step 2.3)
   processed/step2_normalized/                  centered and scaled shapes (Step 2.5)
   processed/step3_normalized/                  fully normalized shapes (Step 3.1, input for Step 3.2)
   report/figures/                              figures as written by the figure scripts (copied to Images/)
   report/data/step5/ann_index.bin, ann_vectors.npy
                                                index of Step 5 (rebuilt automatically when missing)
