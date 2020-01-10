from fastai.core import *

Path.read_csv = lambda o: pd.read_csv(o)
input_path = Path("../input/")
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200
print(input_path.ls())
## Read data
sample_subdf = (input_path/'sample_submission.csv').read_csv()
specs_df = (input_path/"specs.csv").read_csv()
train_df = (input_path/"train.csv").read_csv()
train_labels_df = (input_path/"train_labels.csv").read_csv()
test_df = (input_path/"test.csv").read_csv()

train_with_features_part1 = pd.read_feather("../input/dsbowl2019-feng-part1/train_with_features_part1.fth")
train_with_features_part1.shape, test_df.shape, train_labels_df.shape
train_with_features_part1.head()
test_df.head()

# there shouldn't be any common installation ids between test and train 
assert set(train_df.installation_id).intersection(set(test_df.installation_id)) == set()

from fastai.tabular import *
import types

stats = ["median","mean","sum","min","max"]
UNIQUE_COL_VALS = pickle.load(open("../input/dsbowl2019-feng-part1/UNIQUE_COL_VALS.pkl", "rb"))
list(UNIQUE_COL_VALS.__dict__.keys())

# add accuracy_group unique vals
UNIQUE_COL_VALS.__dict__['accuracy_groups'] = np.unique(train_with_features_part1.accuracy_group)
UNIQUE_COL_VALS.accuracy_groups
pickle.dump(UNIQUE_COL_VALS, open( "UNIQUE_COL_VALS.pkl", "wb" ))

def target_encoding_stats_dict(df, by, targetcol):
    "get target encoding stats dict, by:[stats]"
    _stats_df = df.groupby(by)[targetcol].agg(stats)   
    _d = dict(zip(_stats_df.reset_index()[by].values, _stats_df.values))
    return _d

def _value_counts(o, freq=False): return dict(pd.value_counts(o, normalize=freq))

def countfreqhist_dict(df, by, targetcol, types, freq=False):
    "count or freq histogram dict for categorical targets"
    types = UNIQUE_COL_VALS.__dict__[types]
    _hist_df = df.groupby(by)[targetcol].agg(partial(_value_counts, freq=freq))
    _d = dict(zip(_hist_df.index, _hist_df.values))
    for k in _d: _d[k] = array([_d[k][t] for t in types]) 
    return _d

print(countfreqhist_dict(train_with_features_part1, "title", "accuracy_group", "accuracy_groups"))

f1 = partial(target_encoding_stats_dict, by="title", targetcol="num_incorrect")
f2 = partial(target_encoding_stats_dict, by="title", targetcol="num_correct")
f3 = partial(target_encoding_stats_dict, by="title", targetcol="accuracy")
f4 = partial(target_encoding_stats_dict, by="world", targetcol="num_incorrect")
f5 = partial(target_encoding_stats_dict, by="world", targetcol="num_correct")
f6 = partial(target_encoding_stats_dict, by="world", targetcol="accuracy")

f7 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=False)
f8 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=True)
f9 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=False)
f10 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=True)

##K_FOLD
from sklearn.model_selection import KFold
# create cross-validated indexes
unique_ins_ids = np.unique(train_with_features_part1.installation_id)
train_val_idxs = KFold(5, random_state=42).split(unique_ins_ids)

feature_dfs = [] # collect computed _val_feats_dfs here
for train_idxs, val_idxs  in train_val_idxs:
    # get train and val dfs
    train_ins_ids, val_ins_ids = unique_ins_ids[train_idxs], unique_ins_ids[val_idxs]
    _train_df = train_with_features_part1[train_with_features_part1.installation_id.isin(train_ins_ids)]
    _val_df = train_with_features_part1[train_with_features_part1.installation_id.isin(val_ins_ids)]
    assert (_train_df.shape[0] + _val_df.shape[0]) == train_with_features_part1.shape[0]
    # compute features for val df
    _idxs = _val_df['title'].map(f1(_train_df)).index
    feat1 = np.stack(_val_df['title'].map(f1(_train_df)).values)
    feat2 = np.stack(_val_df['title'].map(f2(_train_df)).values)
    feat3 = np.stack(_val_df['title'].map(f3(_train_df)).values)
    feat4 = np.stack(_val_df['world'].map(f4(_train_df)).values)
    feat5 = np.stack(_val_df['world'].map(f5(_train_df)).values)
    feat6 = np.stack(_val_df['world'].map(f6(_train_df)).values)
    feat7 = np.stack(_val_df['title'].map(f7(_train_df)).values)
    feat8 = np.stack(_val_df['title'].map(f8(_train_df)).values)
    feat9 = np.stack(_val_df['world'].map(f9(_train_df)).values)
    feat10 = np.stack(_val_df['world'].map(f10(_train_df)).values)
    # create dataframe with same index for later merge
    _val_feats = np.hstack([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10])
    _val_feats_df = pd.DataFrame(_val_feats, index=_idxs)
    _val_feats_df.columns = [f"targenc_feat{i}"for i in range(_val_feats_df.shape[1])]
    feature_dfs.append(_val_feats_df)

train_feature_df = pd.concat(feature_dfs, 0)

train_with_features_part2 = pd.concat([train_with_features_part1, train_feature_df],1)

train_with_features_part2.to_feather("train_with_features_part2.fth")

##PArt 1 notebook with 6160 features

train_labels_df[['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].corr()


train_labels_df.pivot_table(values= "installation_id",index="accuracy_group", columns="accuracy", aggfunc=np.count_nonzero)

# Get last assessment start for each installation_id - it should have 'event_code' == 2000, we have exactly 1000 test samples that we need predictions of
test_assessments_df = test_df.sort_values("timestamp").query("type == 'Assessment' & event_code == 2000").groupby("installation_id").tail(1).reset_index(drop=True)

# event_data, installation_id, event_count, event_code, game_time is constant for any assessment start
# for extarcting similar rows we can look at event_code==2000 and type==Assessment combination for each installation_id
test_assessments_df = test_assessments_df.drop(['event_data', 'installation_id', 'event_count', 'event_code', 'game_time'],1); test_assessments_df

# there is unique event_id for each assesment
test_assessments_df.pivot_table(values=None, index="event_id", columns="title", aggfunc=np.count_nonzero)['game_session']

# there are common worlds among different assessments
test_assessments_df.pivot_table(values=None, index="world", columns="title", aggfunc=np.count_nonzero)['game_session']

test_assessments_df.describe(include='all')

def get_assessment_start_idxs(df): return listify(df.query("type == 'Assessment' & event_code == 2000").index)

# drop installation ids without at least 1 completed assessment
_train_df = train_df[train_df.installation_id.isin((train_labels_df.installation_id).unique())].reset_index(drop=True)

# join training labels to game starts by game sessions  
_trn_str_idxs = get_assessment_start_idxs(_train_df)
_label_df = _train_df.iloc[_trn_str_idxs]
_label_df = _label_df.merge(train_labels_df[['game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']], "left", on="game_session")
_label_df = _label_df[["event_id", "installation_id", 'game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']]
_label_df.head()

_label_df['accuracy_group'].value_counts(dropna=False).sort_index()
# join labels to train by event_id, game_session, installation_id
train_with_labels_df = _train_df.merge(_label_df, "left", on=["event_id", "game_session", "installation_id"])
train_with_labels_df['accuracy_group'].value_counts(dropna=False).sort_index()

train_with_labels_df.shape
# success statistics per game
(train_with_labels_df.query("type == 'Assessment'")
                     .groupby(["title", "world"])['accuracy']
                     .agg({np.mean, np.median, np.max, np.min})
                     .sort_values("mean"))
def count_nonnan(l): return np.sum([0 if np.isnan(o) else 1 for o in l])

# verify that all training installation ids have at least one assesment with non NaN label
assert not any(train_with_labels_df.groupby("installation_id")['accuracy'].agg(count_nonnan) == 0) 
# save dataframe train with labels
train_with_labels_df.to_csv("train_with_labels.csv", index=False)

# save MEM space
del _label_df
del _train_df
gc.collect()

###HISTORY of user

from fastai.tabular.transform import add_datepart

# set filtered and labels added df
train_df = train_with_labels_df

def get_assessment_start_idxs_with_labels(df):
    "return indexes that will be used for supervised learning"
    df = df[~df.accuracy.isna()]
    return listify(df.query("type == 'Assessment' & event_code == 2000").index)

def get_sorted_user_df(df, ins_id):
    "extract sorted data for a given installation id and add datetime features"
    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)
    add_datepart(_df, "timestamp", time=True)
    return _df

# pick installation_id and get data until an assessment_start
rand_id = np.random.choice(train_df.installation_id)
user_df = get_sorted_user_df(train_df, rand_id)
start_idxs = get_assessment_start_idxs_with_labels(user_df)
print(f"Assessment start idxs in user df: {start_idxs}")

# we would like to get and create features for each assessment start for supervised learning
str_idx = start_idxs[1]
user_assessment_df = user_df[:str_idx+1]; user_assessment_df


## FEATURES to try out
