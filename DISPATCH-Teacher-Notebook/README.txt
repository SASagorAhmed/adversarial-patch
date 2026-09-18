DISPATCH Adversarial Patch Defense Experiment
=============================================

MAIN FILE
---------
dispatch_defense_experiment.ipynb

This is the executed Jupyter notebook for teacher submission.


HOW TO OPEN
-----------
Double-click:

    open_notebook.bat

This starts Jupyter in this folder. Then open dispatch_defense_experiment.ipynb.


HOW TO RE-RUN ALL CELLS
-----------------------
Double-click:

    run_notebook.bat

This executes the notebook from top to bottom and saves outputs in place.
Diffusion inference is slow (tens of minutes). Do not close the window.


DEPENDENCIES (read-only, configured in config.json)
---------------------------------------------------
- DISPATCH-Defense  (CompVis LDM checkpoint + working Python env)
- DISPATCH-Paper-Faithful  (audited official Automatic implementation)
- Hyper-YOLO  (hyper-yolon.pt)

Do not edit those projects. All notebook outputs stay in this folder.


IMPORTANT
---------
CLEAN images are visual/evaluation only.
LDM restoration always uses the ATTACKED/PATCHED image.
Automatic DISPATCH does not receive true patch coordinates until after
the predicted mask is frozen (evaluation only).
Known Location is an oracle diagnostic, not the Automatic method.


SELECTED IMAGES
---------------
000000127263.jpg  car     attack_success  patch 20
000000393093.jpg  car     attack_success  patch 14
000000026926.jpg  car     attack_success  patch 22
000000407083.jpg  car     attack_success  patch 144
000000331604.jpg  person  control         patch 21
