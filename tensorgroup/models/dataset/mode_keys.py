"""Standard names for input dataloader modes.

The following standard keys are defined:

* `TRAIN`: training mode.
* `EVAL`: evaluation mode.
* `PREDICT`: prediction mode.
* `PREDICT_WITH_GT`: prediction mode with groundtruths in returned variables.
"""

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'
PREDICT_WITH_GT = 'predict_with_gt'