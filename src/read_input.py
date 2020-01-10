import pandas as pd
import time


keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count',
             'event_code','title' ,'game_time', 'type', 'world','timestamp']
train=pd.read_csv('../input/train.csv')
train_labels=pd.read_csv('../input/train_labels.csv',
                         usecols=['installation_id','game_session','accuracy_group'])
test=pd.read_csv('../input/test.csv')


print(train.head())

def get_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df
    
train = get_time(train)

## Feature engineering
print(f'No of rows in train_labels: {train_labels.shape[0]}')
print(f'No of unique game sessions in train_labels: {train_labels.game_session.nunique()}')

train = train.drop(['date','month', 'hour', 'dayofweek'], axis=1)

