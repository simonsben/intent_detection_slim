![](banner.png)

This is a slimmed-down implementation of abusive intent detection.
It is based on the research done for my Master's.

Abusive intent is defined as a statement of future action or desire to negatively impact a target individual or group.
Simply put, the model is designed to find text where someone states that they are/want to do something in the future.
This statement should also contain abusive language, since we are most interested in malicious actions.

## Usage

Before making predictions prepare the environment by executing [`setup.py`](setup.sh).
This will create the appropriate data directories, download dependencies, and download the fastText model.

Once the environment is prepared place your dataset (saved in a csv) to [`data/source/`](data/source).
Then change the dataset name in [`config.py`](config.py) to correspond to that filename.
Set your shell to use the virtual environment by running `source ".env/bin/activate"` from the root directory.
Change the `context_index` value in [`prepare_data.py`](execution/prepare_data.py) to the column index of the text.
Prepare the data by running [`prepare_data.py`](execution/prepare_data.py).
While still in the virtual environment, you can now execute [`make_predictions.py`](execution/make_predictions.py).
The top 25 documents with abusive intent will be printed to the console.
All of the predictions will also be saved to [`data/predictions/`](data/predictions/).
