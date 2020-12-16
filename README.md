# NIH COMmunity - COVID-19 Ongoing Monitoring - Phase 1

NIH-supported effort to rapidly integrate data from multiple sources, including surveys and wearable sensors, to identify individuals who may have undiagnosed COVID-19.

This repository provides ML models for detecting the onset of COVID-like symptoms from wearable data and a survey data-based model with the potential to differentiate flu-like symptoms from COVID-19 symptoms.

### Content

1. _Installation_ - see package [requirements](./requirements.txt) (install via `pip` or `conda`)

2. See [example notebook](./0__example_notebook.ipynb) which includes a walk-through of training and evaluating a COVID onset detection model using synthetic data
  * Invokes functions from [utils.py](./utils.py)

3. Trained [wearable model](./trained_models/wearable_model.json) for reapplication to new datasets -- this can be subsituted in the example notebook above and reapplied to actual data

### Example

* **Synthetic data** is used to illustrate the model training and evaluation pipeline. Using the [default generation function and parameters](https://github.com/evidation-opensource/nih-community/blob/219f2d50b174edb31896f212a1d15287519e3b89/utils.py#L40) the following shifts in normalized heart-rate derived features around Influenza-like Illness (**ILI** e.g. `Flu`, or `COVID-19`) event onset can be seen:

![Signal around event onset](./images/signal_around_onset.png)

* Missing data is also handled by the model (_informative missingness_) and performance decay around days with all-missing inputs is shown below:

![Missingness performance](./images/missingness_performance.png)

* Validation AUROC is used to select a model after [hyperparameter tuning](https://github.com/evidation-opensource/nih-community/blob/219f2d50b174edb31896f212a1d15287519e3b89/utils.py#L170) and a threshold (for prediction of COVID-19 events) is selected based on a `95% specificify` 

```
95% Specifivity cutoff = 0.8778
              precision    recall  f1-score   support

           0       0.85      0.95      0.90       385
           1       0.83      0.57      0.68       154

    accuracy                           0.84       539
   macro avg       0.84      0.76      0.79       539
weighted avg       0.84      0.84      0.83       539
```

![Validation ROC](./images/validation_ROC.png)

* **Cumulative recall** around onset is used to evaluate performance of the COVID-19 detection model. However, a model trained without explicitly accounting for other ILI events is suseptible to confusing non-COVID-19 ILI events (red and blue lines in plot below) as COVID-19 (orange line)

![Cumulative recall around onset](./images/cumulative_recall_around_onset.png)
