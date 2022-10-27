import pandas as pd
import math
import datetime
from tqdm import tqdm

paths = {'windows': 'C:/Users/jjung/iCloudDrive/ETH/MSc 3rd semester/Semester project/Data',
         'macOS': '/Users/janoschjungo/Library/Mobile Documents/com~apple~CloudDocs/ETH/MSc 3rd semester/Semester project/Data'}
path = paths['windows']

# import physiological data
physio = pd.DataFrame()
for subjectID in range(1, 28):
    # load physiological data for subject
    try:
        file = path + f'/subjectID_{subjectID}.csv'
        physio_subject = pd.read_csv(file)
    except FileNotFoundError:
        path = paths['macOS']
        file = path + f'/subjectID_{subjectID}.csv'
        physio_subject = pd.read_csv(file)
    n_entries = physio_subject.shape[0]
    physio_subject['SubjectID'] = [subjectID for i in range(n_entries)]

    # clean column names (dataset contains different column names per subject)
    if 'SkinTemperature.Value' in physio_subject.columns:
        physio_subject = physio_subject.rename(columns={'SkinTemperature.Value': 'SkinTemperature'}, errors='raise')

    # combine all subject data
    physio = pd.concat([physio, physio_subject])

# %%
# import fatigue (PROs) data
fatigue = pd.read_csv(path + '/fatiguePROs.csv')

# %%
# convert questionnaires into variables
data = {}
for _, row in fatigue.iterrows():
    # extract data
    subjectID, timestamp, timezone, question, VAS, answer = row
    fatigue_label = {'Physically, today how often did you feel exhausted?': 'phF',
                     'Mentally, today how often did you feel exhausted?': 'MF',
                     'Describe fatigue on a scale of 1 to 10, where 1 means you don’t feel tired at all and 10 means the worst tiredness you can imagine': 'VAS',
                     'Are you feeling better, worse or the same as yesterday?': 'ReIP',
                     'Did you do sport today?': 'Sport'}[question]
    fatigue_score = VAS if not math.isnan(VAS) else answer

    # combine same day data
    day = timestamp.split()[0]
    if (subjectID, day) in data.keys():
        # already other fatigue data for this day
        data[(subjectID, day)][fatigue_label] = fatigue_score  # every day has 1 fatigue score per label
    else:
        # no data for this day yet
        data[(subjectID, day)] = {
            'timezone': timezone,
            fatigue_label: fatigue_score,
            'subjectID': subjectID,
            'day': day
        }

# %%
# combine all data
for _, row in tqdm(physio.iterrows(), total=physio.shape[0]):
    # extract data
    row = row.to_dict()
    subjectID, timestamp = row.pop('SubjectID'), row.pop('Timestamp')

    # combine same day data (concatenate all physiological data for 1 day)
    [day, hour] = timestamp.split()
    if (subjectID, day) in data.keys():
        # append physiological data to day
        for physio_name, physio_value in row.items():
            datapoint = (hour, physio_value)    # TODO: sort dataframes before s.t. here not necessary
            try:
                data[(subjectID, day)][physio_name] += [datapoint]
            except KeyError:
                # this variable hasn't been measured yet
                data[(subjectID, day)][physio_name] = [datapoint]
    else:
        # no fatigue labels for this day
        data[(subjectID, day)] = {physio_name: [(hour, physio_value)] for physio_name, physio_value in row.items()}

# %%
# create single dataframe from all data
dataframe = pd.DataFrame()
for _, row in tqdm(data.items()):
    entry = {key: value if not isinstance(value, list) else \
        [element[1] for element in sorted(value, key=lambda x: int(x[0].replace(':', '')))] \
             for key, value in row.items()}

    entry = pd.DataFrame.from_dict([entry])
    dataframe = pd.concat([dataframe, entry], ignore_index=True)
