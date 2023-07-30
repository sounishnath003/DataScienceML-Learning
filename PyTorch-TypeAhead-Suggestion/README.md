# TypeAhead Word Pharse Suggestion

A simple base sequence-2-sequence model, that learn from a domain of text passages, understands the context and tries to suggest next word pharses for your typings.

Inspired by intellense into our daily mobile keyboard. just an conceptual implementation.

There are lots of ways to improve the functionality, also statistical method of predicting suggestives words can be taken similar to open ai concepts.

## Setups

1. Add any `data.txt` file as textual file that has let's say a wikipedia article or any type of blog.

2. Run the script to train model
```bash
pip install --force-reinstall poetry
poetry install
poetry shell
chmod +x run.sh
sh +x run.sh
```

3. Try the inference
```bash

sh +x torchf.inference "Pitbull is a worldwide"
```
