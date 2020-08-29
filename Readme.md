# Quick-start

## Train a new model with default parameters

```python
import sort_game_net
sort_game_net.utilities.train(
  model_path="where/to/save/model.pth",
  result_path="where/to/save/result.csv"
  )
```

## Run model

```python
import sort_game_net
sort_game_net.utilities.run_batch(
  model_path="./pre_trained/example_model.pth",
  batches=200
  )
```

## Parameters for train
* *model_path* file path to where the model (.pth) is to be saved during training.
* *result_path* file path to where the result (.csv) is to be saved during training.
* *model* optional path to a pre-trained model (.pth) to start from
* *state_fun* which function to use in order to encode the state "encoded" or "int"
* *empty_pos_indicator* integer that denotes an empty position
* *hidden_size* the size of the hidden layers for actor and critic nets. Can be defined as an integer, as [int, int] or [[int, ...], [int, ...]] to set different sizes for different layers.
* *batch_size* the number of batches in each epoch
* *batches* number of batches to train for
* *learning_rate*
* *gamma*
* *entropy_loss_coeff*