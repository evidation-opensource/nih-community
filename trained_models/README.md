## Trained models for NIH COMmunity project

* [Wearable model](./wearable_model.json) - takes in 4 heart-rate related daily features with a look-back of 5 days to make a predicton about whether the current day is a COVID (post-onset) day
    * The wrapper code for applying this model to new datasets follows the process outlined in the [example notebook](../0__example_notebook.ipynb)

* [Survey model](./survey_model.json) - takes a set of demographic and daily symptom features to predict whether the current day is COVID (post-onset) day
    * 
