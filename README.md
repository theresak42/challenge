Audio and Music Processing Challenge Team C
===========================================

We used the provided scripts as a basis but wrote our own scripts.

`main.py`
-------------
This script is used for training and evaluating the onset detection function as well as predicting onsets and tempo.

By running the script, our current best model is evaluated.

By removing the # in line 45, you can train a new model. By removing the #s in lines 61-65, you can make predictions on the given test set.



`main_beats.py`
-------------

This script is used for training and evaluating the beat detector as well as predicting beats. Running the script evaluates the current best model. By enabling (remove #) line 92, you can train a new model. By enabling line 100-105 you can evaluate onsets, tempo and beats.

Lines 32-62 use the multiple-agent approach for beat detection. You can enable this section to test this approach.



`experimnents.txt`
-------------

This file includes a list of experiments we conducted for the different tasks.


Predictions
-----------
All of our predictions for the test set can be found in the folder "pred".
