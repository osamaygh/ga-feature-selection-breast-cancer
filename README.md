# GA-Based Feature Selection for Breast Cancer Classification

This is my Task 1 project for Computational Intelligence and Machine Learning
(Week 3 topic: Genetic Algorithms). I built a GA from scratch and applied it
to feature selection - picking the best subset of features for classifying
tumors in the Breast Cancer Wisconsin dataset as malignant or benign.

## Why feature selection

The dataset has 30 numerical features per sample (things like mean radius,
mean texture, mean smoothness, etc.), and not all of them are equally useful
for classification - some are probably redundant. With 30 binary features
there are 2^30 possible subsets, way too many to check by hand or brute
force, which is exactly the kind of problem a GA is good at: it searches
that space efficiently by evolving better and better subsets over
generations instead of checking every combination.

## How I set it up

- **Chromosome:** binary vector, length 30, one bit per feature (1 =
  keep it, 0 = drop it).
- **Fitness:** 3-fold cross-validated accuracy of a Random Forest trained
  only on the selected features, minus a small penalty per feature used
  (0.0015 per feature) so the GA is pushed toward smaller subsets rather
  than that happening by chance. Cross-validation runs only on the
  training set - the test set never gets touched during the search.
- **Selection:** tournament selection, k=3.
- **Crossover:** uniform crossover, rate 0.8.
- **Mutation:** bit-flip, rate 0.02 per gene.
- **Elitism:** keep the best individual around every generation so it's
  never accidentally lost.
- Population 20, 30 generations.

I wrote the GA loop itself by hand (`genetic_algorithm.py`) - no GA
library. scikit-learn is only used for the dataset and for the Random
Forest classifier inside the fitness function.

I picked Random Forest as the classifier mainly because it doesn't need
feature scaling and trains fast, which matters since it gets retrained for
basically every chromosome, every generation. I also asked around and a
friend who's worked with this exact dataset before said RF tends to work
well on it, which lined up with what I was already planning.

One thing I got wrong at first: I originally computed fitness using a
single train/test split, which meant the GA could end up quietly
overfitting to whatever happened to land in that one split. I switched to
cross-validation on the training set instead, and kept the test set
completely separate, touched only once at the very end for the final
number. That final comparison (baseline vs. GA, both on the untouched test
set) is what's reported below.

## Results

### One run

| | Features used | Test accuracy |
|---|---|---|
| Baseline (all features) | 30 | 0.9357 |
| GA-selected features | 8 | 0.9474 |

### Across 5 different seeds

I re-ran the GA with 5 different random seeds (same train/test split each
time) just to make sure the result above wasn't a lucky one-off.

| | |
|---|---|
| Baseline test accuracy | 0.9357 |
| GA test accuracy (mean ± std) | 0.9474 ± 0.0052 |
| GA test accuracy (min-max) | 0.9415 - 0.9532 |
| Features used (mean ± std) | 9.4 ± 2.0 |
| Seeds where GA ≥ baseline | 5 / 5 |

Every single seed beat the baseline, using roughly a third of the original
features. The plots are `convergence.png` (one run) and
`robustness_convergence.png` (all 5 seeds overlaid).

## Files

```
genetic_algorithm.py         the GA itself
utils.py                     data loading + baseline evaluation (shared)
main.py                      one full run: baseline + GA + plot
robustness.py                runs it across 5 seeds
baseline.py                  quick standalone baseline check
convergence.png              convergence plot, one run
robustness_convergence.png   convergence plot, all 5 seeds
summary.txt                  results of one run
robustness_summary.txt       results across 5 seeds
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python main.py          # baseline + GA, single run
python robustness.py    # 5-seed check, takes a few minutes
```

## Things I'd still want to try

- The dataset is honestly pretty "easy" - baseline is already at 93.6% -
  so the accuracy gain from feature selection is real but not huge (~1
  point). A messier dataset would probably show a bigger effect.
- I only tried Random Forest as the classifier. Testing SVM or logistic
  regression in the same setup would tell me whether the selected
  features are actually meaningful or just convenient for RF specifically.
- Fitness evaluation is the bottleneck (retraining a classifier per
  chromosome per generation) - a cache helps, but a bigger dataset would
  need something smarter, maybe evaluating the population in parallel.

## Author

osama - Master's student, Computer Science, Universitas Gadjah Mada (UGM)
