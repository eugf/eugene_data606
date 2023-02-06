# INSTRUCTIONS:

(EMAIL)"issue: lack of details. They are too brief and have very little substance."

1) Please provide details on

   - the datasets,
   - \# of obs,
   - \# columns,
   - what the columns are and their types.
   - If it is geographical,
     - what areas it covers,
   - if it is time series,
     - what time frame it covers.
   - What is the label you want to predict,
   - what are the features you use to predict the label.
2) What ML models you plan to use, how will you compare them and pick the best?
3) if you deploy the ML model to a webapp, what the functions it will provide?
4) Make sure you perform the initial EDA and have a jupyter notebook file in src with good documentations.

# [1.1] DATASETS

I originally found the dataset on [Hum Data's dataset collection](https://data.humdata.org/dataset/who-data-for-united-states-of-america), which itself is sourced from the [World Health Organization's (WHO) dataset](https://www.who.int/data/gho) on [Global Health Indicators](https://www.who.int/data/gho/data/indicators). These Global Health Indicators come from the [Global Health Observatory (GHO)](https://en.wikipedia.org/wiki/Global_Health_Observatory) which is the division of the WHO which shares global data specific to health measures by country.

The dataset I've chosen to look are the Health Measures for the US, which is updated monthly and is currently up to JAN 2023.

# [1.2] \# OF OBJECTS

The total # of objects in this dataset is 780858

# [1.3] \# OF COLUMNS

There are 39 columns

# [1.4] COLUMN NAMES + TYPES

GHO (CODE)                         object
GHO (DISPLAY)                      object
GHO (URL)                          object
PUBLISHSTATE (CODE)                object
PUBLISHSTATE (DISPLAY)             object
PUBLISHSTATE (URL)                float64
YEAR (CODE)                        object
YEAR (DISPLAY)                     object
YEAR (URL)                        float64
REGION (CODE)                      object
REGION (DISPLAY)                   object
REGION (URL)                      float64
WORLDBANKINCOMEGROUP (CODE)        object
STARTYEAR                          object
ENDYEAR                            object
WORLDBANKINCOMEGROUP (DISPLAY)     object
WORLDBANKINCOMEGROUP (URL)        float64
COUNTRY (CODE)                     object
COUNTRY (DISPLAY)                  object
COUNTRY (URL)                     float64
AGEGROUP (CODE)                    object
AGEGROUP (DISPLAY)                 object
AGEGROUP (URL)                    float64
SEX (CODE)                         object
SEX (DISPLAY)                      object
SEX (URL)                         float64
GHECAUSES (CODE)                   object
GHECAUSES (DISPLAY)                object
GHECAUSES (URL)                   float64
CHILDCAUSE (CODE)                  object
CHILDCAUSE (DISPLAY)               object
CHILDCAUSE (URL)                  float64
Display Value                      object
Numeric                            object
Low                               float64
High                              float64
StdErr                            float64
StdDev                            float64
Comments                           object

In summary, the datatypes in this dataset can be broken down into 2 main types:
1) float64(13) --> which are the numerical datatype columns
2) object(26) --> which are strings (Python's categorical) datatype columns

# [1.5] GEOGRAPHICAL? -Y

### Areas covered: USA

The dataset is filtered by country code for only the USA subset. There should be a larger dataset somewhere for global reach, but I'm not sure if there's a more specific geographical dataset that breaks down regions by state or ZIP code available from the WHO. 

# [1.6] TIME SERIES? -Y

### Time frame covered:

# [1.7] LABEL TO PREDICT

# [1.8] FEATURES USED FOR PREDICTION
