# Sensor-Based Modeling of Fatigue Using Transformer Model

# Data flowchart
```mermaid
graph TD;
    Data --> id1([data_loader]); 
    id1([data_loader]) --> Output;
    Output --> id2([preproc_data]);
    Output --> id3([data_analyzation]);
    id2([preproc_data]) --> Models;
    Models --> id4([majority_voting])
    Models --> id5([CNN]);   
```

# Notes
- normalize subject data (Z-score) to combat intra-subject variability? (~ calibration)
- Feature vector: incl. whether data artificial or not?
- Better imputation strategy