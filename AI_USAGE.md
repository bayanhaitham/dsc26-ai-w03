# AI usage

## Tools I used, and for what

I used ChatGPT to understand the assignment requirements, explain the implementation step by step, help design the feature engineering and scikit-learn pipeline, and debug errors during development. I used the terminal and pytest to run the provided tests and verify the implementation.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I reviewed the suggested implementations of `add_history_features` and `time_split` against their requirements. I did not need to make corrections to either function after testing them; both passed their corresponding public tests.

During pipeline development, I did fix an issue in the suggested `clean_text` helper: the text column was passed as a one-dimensional Series, but the initial implementation treated it as a two-dimensional DataFrame. I changed it to handle the Series correctly.

## How I checked that my model has no leak

I excluded `failed`, `courier_notes`, and `delivered_at` from `FEATURE_COLUMNS`, used a chronological train/test split, and ran the public leakage test, which passed.
