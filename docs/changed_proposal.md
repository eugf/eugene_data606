# INSTRUCTIONS:

(EMAIL) "issue: lack of details. They are too brief and have very little substance."

1) Please provide details on

   - the datasets,
   - \# of obs,
   - \# columns,
   - what the columns are and their types.
   - If it is geographical,
     - what areas it covers,
   - If it is time series,
     - what time frame it covers.
   - What is the label you want to predict,
   - What are the features you use to predict the label.
2) What ML models you plan to use, how will you compare them and pick the best?
3) If you deploy the ML model to a webapp, what functions will it provide?
4) Make sure you perform the initial EDA and have a jupyter notebook file in src with good documentations.

# [1.1] DATASETS

I originally found the dataset on [Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers), which itself is sourced from the [Open Access Series of Imaging Studies (OASIS)](https://www.oasis-brains.org/#about). This is a public dataset of patient data centered around neuroimaging and includes indicators of mental health, including brain scans and related health metrics.

The dataset I'm using is from OASIS-1. In my case I will be using the text-based parts of the dataset. This includes 416 subjects. The dataset includes both healthy subjects, as well as subjects already diagnosed with mild to moderate Alzheimer's disease (AD). OASIS-1 has Cross-sectional MRI data -- meaning a single observation of the patient at that point in their life.

# [1.2] \# OF OBJECTS

The total # of objects in this dataset is ????

# [1.3] \# OF ROWS X COLUMNS

There are 20,021 rows

- 2 header rows?

There are 39 columns

- Many are null, or large portions are null (depending on the type of data)

# [1.4] COLUMN NAMES + TYPES


| COLUMN NAME | DATATYPE |
| :------------ | :--------- |
| ???         |          |

In summary, the datatypes in this dataset can be broken down into 2 main types:

1) ???

# [1.5] GEOGRAPHICAL? -N

### Areas covered: ???


# [1.6] TIME SERIES? -N

### Time frame covered: 1951 - 2025

### Time units:???

But there may be some issues with the automatically assigned datatypes here...

For example, there are 3 columns for "YEAR" in addition to 2 more columns for "STARTYEAR" and "ENDYEAR," making for 5 different columns representing "YEAR." The "YEAR (URL)" is 100% empty, while the others have Issues like having decimal points for a single year and the year ranges being considered as strings due to the dash symbol (-), but it's probably intended to refer to the "STARTYEAR" and "ENDYEAR" columns for those and this was a convenient way for them to merge those datasets in which probably didn't collect yearly stats.

# [1.7] LABEL TO PREDICT

Demented or not demented

# [1.8] FEATURES USED FOR PREDICTION

Under the `GHO (CODE)` and `GHO (DISPLAY)` columns they have various Global Health Indicators, mortality is included there but I think I ???

# [2] What ML models you plan to use, how will you compare them and pick the best?

I'm not sure which would be the best to use yet, but I think multiclass classification would be the most relevant to identifiy multiple types of categorical data and their causes. But regression could work for a time-based prediction on mortality rates. I'll probably want to try both and see if they work. For comparing I would want to randomly split the data and use a Confusion Matrix to rate how well each method worked. This will allow me to calculate the:

- Precision
- Recall
- F1 score

# [3] If you deploy the ML model to a webapp, what functions will it provide?

???
