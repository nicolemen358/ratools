# Emily_streamlit

This repository contains the Streamlit application for the Compas Recidivism use case. The application contains a streamlit dashboard with 3 sections - the left section contains the overall model results, the middle column contains the results and statistics of the African-American race and the right section contains the results and statistics of the Caucasian race. There is also a sidebar that contains options for the user to select the column names for various column groups.

To run the app, open your terminal, go to the folder containing the code file "Compas Streamlit.py" :
```
streamlit run Compas\ Streamlit.py
```

In order to control the output dynamically, there is a config.json file with multiple variables.

A summary with descriptions of each key-value pair in the config file is given below:

## config.json

  The key data_file expects the name of the data file in the value.<br>
  ```
  "data_file":"pretrial-priors-outcomes.csv"
  ```
  <br>
  The baseline_model_configs key contains 3 keys:<br>
  - column_groups_baseline : This key contains various column groups and their selected column for the baseline model. The rest of the columns must be mentioned in the others section. <br>
  - output_column : The output column expects the output column for the models.<br>
  - categorical_columns: The categorical_column contains the list of categorical columns so that they can be converted into numerical values. <br>
  
  ```
  "baseline_model_configs":{
      "column_groups_baseline":{
          "convict":["convict_present"],
          "charges":["charges_prior"],
          "others":["race",
          "gender",
          "representation",
          "fta",
          "AgeFirstArrest",
          "pre21",
          "convict_present_pre21",
          "convict_present_post21",
          "convict_present_pre21_flag",
          "convict_present_post21_flag",
          "convict_pre21_prior",
          "convict_pre21_prior_flag",
          "convict_post21_prior",
          "convict_post21_prior_flag",
          "incarceration",
          "dui_chgs_present",
          "motor_vehicle_chgs_present",
          "person_chgs_present",
          "property_chgs_present",
          "public_order_chgs_present",
          "criminal_chgs_present",
          "drugs_chgs_present",
          "weapons_chgs_present",
          "Sexual_chgs_present",
          "misdemeanor_chgs_present",
          "felony_chgs_present",
          "summary_chgs_present",
          "homicide_chgs_present",
          "dui_convict_present",
          "motor_vehicle_convict_present",
          "person_convict_present",
          "property_convict_present",
          "public_order_convict_present",
          "criminal_convict_present",
          "drugs_convict_present",
          "weapons_convict_present",
          "Sexual_convict_present",
          "misdemeanor_convict_present",
          "felony_convict_present",
          "summary_convict_present",
          "homicide_convict_present",
          "dui_chgs_present_flag",
          "motor_vehicle_chgs_present_flag",
          "person_chgs_present_flag",
          "property_chgs_present_flag",
          "public_order_chgs_present_flag",
          "criminal_chgs_present_flag",
          "drugs_chgs_present_flag",
          "weapons_chgs_present_flag",
          "Sexual_chgs_present_flag",
          "misdemeanor_chgs_present_flag",
          "felony_chgs_present_flag",
          "summary_chgs_present_flag",
          "homicide_chgs_present_flag",
          "dui_convict_present_flag",
          "motor_vehicle_convict_present_flag",
          "person_convict_present_flag",
          "property_convict_present_flag",
          "public_order_convict_present_flag",
          "criminal_convict_present_flag",
          "drugs_convict_present_flag",
          "weapons_convict_present_flag",
          "Sexual_convict_present_flag",
          "misdemeanor_convict_present_flag",
          "felony_convict_present_flag",
          "summary_convict_present_flag",
          "homicide_convict_present_flag",
          "dui_chgs_present_prior",
          "motor_vehicle_chgs_present_prior",
          "person_chgs_present_prior",
          "property_chgs_present_prior",
          "public_order_chgs_present_prior",
          "criminal_chgs_present_prior",
          "drugs_chgs_present_prior",
          "weapons_chgs_present_prior",
          "Sexual_chgs_present_prior",
          "misdemeanor_chgs_present_prior",
          "felony_chgs_present_prior",
          "summary_chgs_present_prior",
          "homicide_chgs_present_prior",
          "dui_convict_present_prior",
          "motor_vehicle_convict_present_prior",
          "person_convict_present_prior",
          "property_convict_present_prior",
          "public_order_convict_present_prior",
          "criminal_convict_present_prior",
          "drugs_convict_present_prior",
          "weapons_convict_present_prior",
          "Sexual_convict_present_prior",
          "misdemeanor_convict_present_prior",
          "felony_convict_present_prior",
          "summary_convict_present_prior",
          "homicide_convict_present_prior",
          "dui_chgs_present_prior_flag",
          "motor_vehicle_chgs_present_prior_flag",
          "person_chgs_present_prior_flag",
          "property_chgs_present_prior_flag",
          "public_order_chgs_present_prior_flag",
          "criminal_chgs_present_prior_flag",
          "drugs_chgs_present_prior_flag",
          "weapons_chgs_present_prior_flag",
          "Sexual_chgs_present_prior_flag",
          "misdemeanor_chgs_present_prior_flag",
          "felony_chgs_present_prior_flag",
          "summary_chgs_present_prior_flag",
          "homicide_chgs_present_prior_flag",
          "dui_convict_present_prior_flag",
          "motor_vehicle_convict_present_prior_flag",
          "person_convict_present_prior_flag",
          "property_convict_present_prior_flag",
          "public_order_convict_present_prior_flag",
          "criminal_convict_present_prior_flag",
          "drugs_convict_present_prior_flag",
          "weapons_convict_present_prior_flag",
          "Sexual_convict_present_prior_flag",
          "misdemeanor_convict_present_prior_flag",
          "felony_convict_present_prior_flag",
          "summary_convict_present_prior_flag",
          "homicide_convict_present_prior_flag",
          "incarceration_prior",
          "incarceration_prior_flag",
          "fta_present_flag",
          "chgs_present",
          "fta_prior",
          "fta_prior_flag",
          "convict_prior"]},
      "output_column":["nca_flag"],
      "categorical_columns":["race","gender","representation","fta"]
  },
  ```
  The column groups contains the various column groups and their corresponding columns to be featured in the dropdowns for all the column groups.
  ```
  "column_groups":
  {
      "convict":["convict_present","convict_present_flag","convict_present_pre21",
          "convict_present_post21",
          "convict_present_pre21_flag",
          "convict_present_post21_flag"],
      "charges":["charges_prior",
          "charges_prior_flag"]
  }
}
```
