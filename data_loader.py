import pandas as pd
import math
import datetime

path = 'C:/Users/jjung/iCloudDrive/ETH/MSc 3rd semester/Semester project/Data'

# import physiological data
physio = pd.DataFrame()
for subjectID in range(1, 28):
    # load physiological data for subject
    file = path + f'/subjectID_{subjectID}.csv'
    physio_subject = pd.read_csv(file)
    n_entries = physio_subject.shape[0]
    physio_subject['SubjectID'] = [subjectID for i in range(n_entries)]

    # clean column names (dataset contains different column names per subject)
    if 'SkinTemperature.Value' in physio_subject.columns:
        physio_subject = physio_subject.rename(columns={'SkinTemperature.Value': 'SkinTemperature'}, errors='raise')

    # combine all subject data
    physio = pd.concat([physio, physio_subject])

#%%
# import fatigue (PROs) data
fatigue = pd.read_csv(path + '/fatiguePROs.csv')

#%%
# combine data
data = {}
for _, row in fatigue.iterrows():
    subjectID, timestamp, timezone, question, VAS, answer = row
    fatigue_label = {'Physically, today how often did you feel exhausted?': 'phF',
     'Mentally, today how often did you feel exhausted?': 'MF',
     'Describe fatigue on a scale of 1 to 10, where 1 means you don’t feel tired at all and 10 means the worst tiredness you can imagine': 'VAS',
     'Are you feeling better, worse or the same as yesterday?': 'ReIP',
     'Did you do sport today?': 'Sport'}[question]
    fatigue_score = VAS if not math.isnan(VAS) else answer

    if (subjectID, timestamp) in data.keys():
        data[(subjectID, timestamp)][fatigue_label] = fatigue_score
    else:
        data[(subjectID, timestamp)] = {
            'timestamp': timestamp,
            'timezone': timezone,
            fatigue_label: fatigue_score
        }

#%%
for _, row in physio.iterrows():
    row = row.to_dict()
    subjectID, timestamp = row.pop('SubjectID'), row.pop('Timestamp')
    if (subjectID, timestamp) in data.keys():
        data[(subjectID, timestamp)] = {**data[(subjectID, timestamp)], **row}
    else:
        data[(subjectID, timestamp)] = {**row}

#%%
temp = list(data.keys())[0][1]
print(temp)
e = datetime.datetime.strptime(temp, "%d.%m.%y %H:%M")
t = datetime.datetime.timestamp(e)
print(t)
print(type(t))
datetime.datetime.fromtimestamp(t)
tt = pd.Timestamp(temp)
tt.month