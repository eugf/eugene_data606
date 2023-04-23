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

The total # of objects in this dataset is: 5,232

# [1.3] \# OF ROWS X COLUMNS

There are 436 rows

- 1 header row

There are 12 columns

- Most of the `Delay` column is null
- A few of the other columns are around half null
- Most of the rest are non-null

# [1.4] COLUMN NAMES + TYPES


| COLUMN NAME | DATATYPE |
| :------------ | :--------- |
| ID          | object   |
| M/F         | object   |
| Hand        | object   |
| Age         | int64    |
| Educ        | float64  |
| SES         | float64  |
| MMSE        | float64  |
| CDR         | float64  |
| eTIV        | int64    |
| nWBV        | float64  |
| ASF         | float64  |
| Delay       | float64  |

In summary, the column datatypes in this dataset can be broken down into 3 main types:

- float64(7)  --> 7 numerical
- int64(2)    --> 2 integer
- object(3)   --> 3 string (categorical)

# [1.5] GEOGRAPHICAL? -N

### Areas covered: (N/A)

# [1.6] TIME SERIES? -N

### Time frame covered: (N/A)

# [1.7] LABEL TO PREDICT

Demented or not demented

# [1.8] FEATURES USED FOR PREDICTION

The rows designate patients. Healthy controls, as well as mild-to-moderate Alzheimer's Disease (AD) patients, are included for comparison. This dataset is already processed from the larger OASIS-1 data, such as joining multiple datasets and removing the MRI image data, so I will attempt to use most of the columns present here. Handedness may be able to be removed, as all patients are right-handed.

# [2] What ML models you plan to use, how will you compare them and pick the best?

I think binary classification is the most relevant ML method to use. 

I'll probably want to try multiple methods of classification to see if different methods can achieve higher accuracy. For comparing I would want to randomly split the data using stratified splitting to preserve the ratio of demented to non-demented patients. Then utilize a Confusion Matrix to rate how well each method worked. This will allow me to calculate the:

- Precision
- Recall
- F1 score

# [3] If you deploy the ML model to a webapp, what functions will it provide?

???
