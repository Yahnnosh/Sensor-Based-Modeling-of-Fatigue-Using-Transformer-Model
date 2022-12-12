# Sensor-Based Modeling of Fatigue Using Transformer Model

# Data flowchart
```mermaid
flowchart TD;
    Data --> id1([data_loader])
    id1([data_loader]) --> Output
    
    Output --> id3([data_analyzation])
    Output --> id2([preproc_data])
    Output --> id6([preproc_data_stat])
    Output ---> id4([majority_voting])
    Output ---> id9([random_guess])
    
    id6([preproc_data_stat]) --> id8([random_forest])
    id6([preproc_data_stat]) --> id10([XGBoost])
    id2([preproc_data]) <.-> id12([imputation_comparison])
    id2([preproc_data]) --> id5([CNN])
    id5([CNN]) <.-> temp
    
    id4([majority_voting]) --> id11([evaluator])
    id5([CNN]) --> id11([evaluator])
    id8([random_forest]) --> id11([evaluator])
    id10([XGBoost]) --> id11([evaluator])
    id9([random_guess]) --> id11([evaluator])
```

# Currently:
- latexify table
- model performance significant
- CNN improvement: overlap incr., augm. incr., daily majority vote

### MobileNet
- MobileNet spectrogram? (-> not trained specifically on spectrograms?)
- MobileNetV2 doesn't expect 30 input channels (-> multiple input nets)
- MobileNetV2 input spectrograms usually much clearer (-> image processing?)
- Other pretrained models?

### Custom CNN
- Smaller models (-> 400k params for 4k data is a lot)
- Combat overfitting
- Weighted loss function
- Multi-head classification (phF + MF)
- image processing?

### Upsampling
- SMOTE (-> meaningful for image data? we lose local correlation)
- RUS (-> downsampling leads to smaller dataset)
- Just upsampling (-> overfitting)
- Upsampling + data augmentation (-> augmentations different enough to avoid overfitting?)
- Weighted loss function

### Additional
- LOSO for general models (-> group (stratified) k-fold)
- Artificial data mask as input
- Data augmentation scaling

# Notes
- normalize subject data (Z-score) to combat intra-subject variability? (~ calibration)
- Feature vector: incl. whether data artificial or not?
- Better imputation strategy