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

# [1.4] COLUMN NAMES + TYPES + METADATA/DATA DICTIONARY


| COLUMN NAME | DATATYPE | CATEGORY                | DESCRIPTION                                                                                                                   |
| :------------ | :--------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------------------ |
| ID          | object   | (N/A)                   | Patient identifier                                                                                                            |
| M/F         | object   | Demographics            | Gender: male/female                                                                                                           |
| Hand        | object   | Demographics            | Handedness: all right-handed (remove???)                                                                                      |
| Age         | int64    | Demographics            | Age: 18 - 96                                                                                                                  |
| Educ        | float64  | Demographics            | Education codes: 1 = less than high school grad, 2 = high school grad, 3 = some college, 4 = college grad, 5 = beyond college |
| SES         | float64  | Demographics            | Socioeconomic status (SES)                                                                                                    |
| MMSE        | float64  | Clinical                | [Mini-Mental State Examination (MMSE)](https://pubmed.ncbi.nlm.nih.gov/34313331/)                                             |
| CDR         | float64  | Clinical                | Clinical Dementia Rating (CDR): 0 = nondemented, 0.5 = very mild dementia, 1 = mild dementia, 2 = moderate dementia           |
| eTIV        | int64    | Derived anatomic values | Estimated total intracranial volume (eTIV) (mm^3)                                                                             |
| nWBV        | float64  | Derived anatomic values | Normalized whole brain volume (nWBV)                                                                                          |
| ASF         | float64  | Derived anatomic values | Atlas scaling factor (ASF)                                                                                                    |
| Delay       | float64  | ???                     | (This may be for the longitudinal follow-up dataset, wherein patients returned <90 days later for follow-up scans)            |

In summary, the column datatypes can be broken down into 3 main types:

- float64(7)  --> 7 numerical
- int64(2)    --> 2 integer
- object(3)   --> 3 string (categorical)

# [1.5] GEOGRAPHICAL? -N

### Areas covered: (N/A)

# [1.6] TIME SERIES? -N

The cross-sectional data is a single slice of time, one observation only.

### Time frame covered: (N/A)

# [1.7] LABEL TO PREDICT

Demented or not demented as denoted by the `CDR` column which is the Clinical Dementia Rating. This could be grouped into just 2 groups for binary classification or potentially used to try to predict the severity of dementia, although this may be much harder and not have enough samples to get good accuracy.

# [1.8] FEATURES USED FOR PREDICTION

The rows designate patients. Healthy controls, as well as mild-to-moderate Alzheimer's Disease (AD) patients, are included for comparison. This dataset is already processed from the larger OASIS-1 data, such as joining multiple datasets and removing the MRI image data, so I will attempt to use most of the columns present here. `Hand` may be able to be removed, as all patients are right-handed. The `Delay` column might be able to be removed as well, it's more relevant for the longitudinal study as it indicates days after the initial visit that they returned for a follow-up.

# [2] What ML models you plan to use, how will you compare them and pick the best?

I think binary classification is the most relevant ML method to use.

I'll probably want to try multiple methods of classification to see if different methods can achieve higher accuracy. For comparing I would want to randomly split the data using stratified splitting to preserve the ratio of demented to non-demented patients. Then utilize a Confusion Matrix to rate how well each method worked. This will allow me to calculate the:

- Precision
- Recall
- F1 score

# [3] If you deploy the ML model to a webapp, what functions will it provide?

I want to try using Streamlit to provide GUI based data visualizations of the dataset along with the ability to manually enter variable parameters to have it make a prediction with user supplied data
