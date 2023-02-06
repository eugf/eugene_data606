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


| COLUMN NAME                    | DATATYPE |
| :------------------------------- | :--------- |
| GHO (CODE)                     | object   |
| GHO (DISPLAY)                  | object   |
| GHO (URL)                      | object   |
| PUBLISHSTATE (CODE)            | object   |
| PUBLISHSTATE (DISPLAY)         | object   |
| PUBLISHSTATE (URL)             | float64  |
| YEAR (CODE)                    | object   |
| YEAR (DISPLAY)                 | object   |
| YEAR (URL)                     | float64  |
| REGION (CODE)                  | object   |
| REGION (DISPLAY)               | object   |
| REGION (URL)                   | float64  |
| WORLDBANKINCOMEGROUP (CODE)    | object   |
| STARTYEAR                      | object   |
| ENDYEAR                        | object   |
| WORLDBANKINCOMEGROUP (DISPLAY) | object   |
| WORLDBANKINCOMEGROUP (URL)     | float64  |
| COUNTRY (CODE)                 | object   |
| COUNTRY (DISPLAY)              | object   |
| COUNTRY (URL)                  | float64  |
| AGEGROUP (CODE)                | object   |
| AGEGROUP (DISPLAY)             | object   |
| AGEGROUP (URL)                 | float64  |
| SEX (CODE)                     | object   |
| SEX (DISPLAY)                  | object   |
| SEX (URL)                      | float64  |
| GHECAUSES (CODE)               | object   |
| GHECAUSES (DISPLAY)            | object   |
| GHECAUSES (URL)                | float64  |
| CHILDCAUSE (CODE)              | object   |
| CHILDCAUSE (DISPLAY)           | object   |
| CHILDCAUSE (URL)               | float64  |
| Display Value                  | object   |
| Numeric                        | object   |
| Low                            | float64  |
| High                           | float64  |
| StdErr                         | float64  |
| StdDev                         | float64  |
| Comments                       | object   |

In summary, the datatypes in this dataset can be broken down into 2 main types:

1) float64(13)
   - These are the numerical datatype columns
   - There are 13 of them which were automatically assigned as a number
2) object(26)
   - These are string (Python's version of categorical) datatype columns

# [1.5] GEOGRAPHICAL? -Y

### Areas covered: USA

The dataset is filtered by country code for only the USA subset. There should be a larger dataset somewhere for global reach, but I'm not sure if there's a more specific geographical dataset that breaks down regions by state or ZIP code available from the WHO.

# [1.6] TIME SERIES? -Y

### Time frame covered: 1951 - 2025

### Time units:

- 1 yr
- AND
- multiple different year ranges:
  - 2 yrs at the shortest
  - 12 yrs at the longest

But there may be some issues with the automatically assigned datatypes here... For example, there are 3 columns for "YEAR" in addition to 2 more columns for "STARTYEAR" and "ENDYEAR," making for 5 different columns representing "YEAR." The "YEAR (URL)" is 100% empty, while the others have Issues like having decimal points for a single year and the year ranges being considered as strings due to the dash symbol (-), but it's probably intended to refer to the "STARTYEAR" and "ENDYEAR" columns for those.

# [1.7] LABEL TO PREDICT

This dataset probably isn't going to be enough for my intended predictions... I might need to augment this with some additional data like mortality data by year and maybe region if I add in more countries or get finer-grained geographical divisions, like individual states. Or just add in other countries since it is the WORLD Health Organization, the other countries should be available on their website for download.

I think it would be interesting to explore 2 different dimensions here:

1) Causes of mortality and predicting what kind of indicators have the biggest impacts
2) Expanding time horizon and predicting how mortality would be different if a particular category was missing or had existed earlier and plot how much of a difference that would be, this may be more of a Data Visualization thing though. But I think the prediction here should be thought of as a hypothetical "What if?" scenario. I think it would be interesting for traditional future predictions, but might also work for hypothetical past scenarios?

# [1.8] FEATURES USED FOR PREDICTION

Under the `GHO (CODE)` and `GHO (DISPLAY)` columns they have various Global Health Indicators, mortality is included there but I think I would have to group them into major categories and reorganize the table since these are in the rows right now. The various `YEAR` columns will definitely be necessary to identify changing trends and when certain interventions, like vaccines, began. `AGEGROUP` would certainly be important. `GHECAUSES` using the major disease groups, such as, `CARDIOVASCULAR DISEASES` and `RESPIRATORY DISEASES`, which could be linked to `Ambient and household air pollution attributable death rate (per 100 000 population)` under the `GHO (DISPLAY)` column. I think I'd have to go through all 20,000 rows of this `GHO (DISPLAY)` column to figure out which ones are useful, some of them seem to be summary statistics which probably shouldn't be used as a feature for prediction.

# [2] What ML models you plan to use, how will you compare them and pick the best?

I'm not sure which would be the best to use yet, but I think multiclass classification would be the most relevant for identifiy multiple types of categorical data and their causes. But regression could work for a time-based prediction on mortality rates. I'll probably want to try both and see if they work. For comparing I would want to split some of the data and use a Confusion Matrix to rate how well it worked.

# [3] if you deploy the ML model to a webapp, what the functions it will provide?

For the time-based one I would want to create predictions for going forward from our current future and see how mortality rates would change. But I also want to try to see if I can force a specific feature to 0 and predict how mortality would have changed back then if a certain feature didn't exist, like air pollution. I'd like it to be an interactive checklist to add/remove features and see the change in results.

For the mortality prediction, I would want a similar checklist but instead of moving across time, I want want it to change specific features and see how it changes disease outcomes instead. Like if air pollution was 0, how would the respiratory disease category change?
