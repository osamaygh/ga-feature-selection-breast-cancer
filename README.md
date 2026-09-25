 GA Feature Selection for Breast Cancer Classification

My first project for Computational Intelligence and Machine Learning (week 3, Genetic Algorithms). I wrote a genetic algorithm from scratch and used it to choose which features to keep when classifying tumors in the Breast Cancer Wisconsin dataset as malignant or benign.
 The idea

Each tumor in the dataset is described by 30 measurements, like mean radius, texture and smoothness. Not all of them are useful, and some carry the same information. With 30 features there are 2^30 possible subsets, which is over a billion, so trying all of them isn't realistic. A GA suits this kind of problem because it improves a population of candidate subsets over many generations instead of checking every combination.

How it works

Every candidate solution is a list of 30 bits, one per feature. A 1 means the feature is used and a 0 means it's dropped.

To score a candidate, I train a Random Forest on only the selected features and take its 3-fold cross-validation accuracy on the training set. Then I subtract 0.0015 for every feature used, so when two subsets are about equally accurate, the smaller one wins.

The rest is a standard GA: tournament selection with 3 competitors, uniform crossover at a rate of 0.8, bit-flip mutation at 0.02 per gene, and elitism so the best solution always survives into the next generation. The population size is 20 and it runs for 30 generations.

The GA itself is written by hand in `genetic_algorithm.py`. I only used scikit-learn for the dataset and the Random Forest.

I went with Random Forest because it doesn't need scaled features and it's fast, which matters because it gets trained again for almost every candidate in every generation. A friend who had worked with this dataset before also told me RF does well on it.

The test set (30% of the data) is never used during the search. It's only used once at the end, to compare the GA's feature subset against the baseline that uses all 30 features. My first version got this wrong: it used a single train/test split inside the fitness function, so the GA could end up fitting that one split. Switching to cross-validation on the training data fixed it.

 Results

One run (seed 42):

| | Features used | Test accuracy |
|---|---|---|
| Baseline (all features) | 30 | 0.9357 |
| GA-selected features | 8 | 0.9474 |

To check that this wasn't luck, I ran the GA again with 5 different seeds on the same train/test split:

| | |
|---|---|
| Baseline test accuracy | 0.9357 |
| GA test accuracy (mean ± std) | 0.9474 ± 0.0052 |
| GA test accuracy (min - max) | 0.9415 - 0.9532 |
| Features used (mean ± std) | 9.4 ± 2.0 |
| Runs that beat the baseline | 5 / 5 |

The GA beat the baseline in all 5 runs while using about a third of the features. `convergence.png` shows how the fitness improves during one run, and `robustness_convergence.png` shows all five runs together.

## Files

- `genetic_algorithm.py`: the GA
- `utils.py`: loading the data and evaluating the baseline
- `main.py`: one full run, including the plot
- `robustness.py`: the 5-seed experiment
- `baseline.py`: a quick check with all 30 features and no GA
- the `.png` and `.txt` files are the saved results

## How to run

```bash
pip install -r requirements.txt
python main.py
python robustness.py
```

`main.py` takes about a minute. `robustness.py` takes a few minutes.

Limitations

This dataset is fairly easy (93.6% accuracy with all features already), so the gain is around one percentage point. A harder dataset would probably show a bigger difference. I also only tried Random Forest, so I don't know yet whether the selected features would work as well with an SVM or logistic regression. Training a new model for every candidate is the slowest part. Caching repeated candidates helps, but a bigger dataset would need something like parallel evaluation.

Author

Osama Yehya Ghazal, Magister Ilmu Komputer, Universitas Gadjah Mada
