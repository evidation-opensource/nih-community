# NIH COMmunity - COVID-19 Ongoing Monitoring - Phase 1

NIH-supported effort to rapidly integrate data from multiple sources, including surveys and wearable sensors, to identify individuals who may have undiagnosed COVID-19.

This repository provides ML models for detecting the onset of COVID-like symptoms from wearable data and a survey data-based model with the potential to differentiate flu-like symptoms from COVID-19 symptoms.

* See [example notebook](./0__example_notebook.ipynb) which includes a walk-through of training and evaluating a COVID onset detection model using synthetic data
  * Invokes functions from [utils.py](./utils.py)

* Trained [wearable model](./trained_models/wearable_model.json) for reapplication to new datasets -- this can be subsituted in the example notebook above and reapplied to actual data
