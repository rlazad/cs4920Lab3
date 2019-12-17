import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# aka_peer_participants
# notes_public_peer_participants
# notes_private_peer_participants
# irr_as_set_peer_particpiants
# info_scope_peer_particpiants
# info_type_peer_particpiants (labels)
# info_prefixes_peer_particpiants


def conv_traffic(df):
    df = df[~df['info_traffic_peer_participants'].isin(['Not Disclosed', 0])]
    
    x = df['info_traffic_peer_participants'].to_numpy()
    #drop the non disclosed and blank rows first
    col = []
    cnt = 0
    for i in x:
        cnt += 1
        num = []
        for d in i:
            if d.isalpha(): 
                if d == 'T':
                    col.append(int(i[0])*1e+12)
                    break
            if d.isalpha() == False:
                if d == '+':
                    k = ''.join(num).split('+')
                    col.append(int(k[0])*1e+11)
                    break
                num.append(d)
                k = ''.join(num).split('-')  
            if d.isalpha(): 
                act = (int(k[0])+int(k[1]))/2
                if d == 'G':
                    col.append(act*1e+9)
                if d == 'M':
                    col.append(act*1e+6)

    df['info_traffic_peer_participants'] = col
    return df

def join_csvs():
    one = pd.read_csv('data/peerParticipants.csv')
    two = pd.read_csv('data/mgmtFacilities.csv')
    three = pd.read_csv('data/mgmtPublics.csv')
    four = pd.read_csv('data/mgmtPublicsFacilities.csv')
    five = pd.read_csv('data/mgmtPublicsIPs.csv')
    six = pd.read_csv('data/peerParticipantsContacts.csv')
    seven = pd.read_csv('data/peerParticipantsPrivates.csv')
    eight = pd.read_csv('data/peerParticipantsPublics.csv')

    df = one.merge(two, how='outer', on='id')
    df = df.merge(three, how='outer', on='id')
    df = df.merge(four, how='outer', on='id')
    df = df.merge(five, how='outer', on='id')
    df = df.merge(six, how='outer', on='id')
    df = df.merge(seven, how='outer', on='id')
    df = df.merge(eight, how='outer', on='id')

    df.to_csv('master.csv', index=False)

def encode_features(lst):
    feature_dict = {}
    val_num = 0
    for ele in set(lst):
        if ele not in feature_dict:
            feature_dict[ele] = val_num
            val_num += 1
    
    new_col = []
    for feature in lst:
        new_col.append(feature_dict[feature])
    
    return new_col

def p1_process(df):
    # get only the columns we need for ml
    data = df[['asn_peer_participants', 'info_traffic_peer_participants', 'info_ratio_peer_participants', 'info_scope_peer_participants', 'info_prefixes_peer_participants', 'policy_general_peer_participants', 'policy_locations_peer_participants', 'policy_ratio_peer_participants', 'policy_contracts_peer_participants', 'info_type_peer_participants']]

    # delete all rows that do not have asn
    data = data[pd.notnull(df['asn_peer_participants'])]

    # delete all rows with nans
    data = data.dropna(axis=0)

    # convert traffic to floats
    data = conv_traffic(data)

    # encode ratio column
    encoded_col = encode_features(data['info_ratio_peer_participants'].tolist())
    data['info_ratio_peer_participants'] = encoded_col

    # encode scope column
    encoded_col = encode_features(data['info_scope_peer_participants'].tolist())
    data['info_scope_peer_participants'] = encoded_col

    # encode general
    encoded_col = encode_features(data['policy_general_peer_participants'].tolist())
    data['policy_general_peer_participants'] = encoded_col

    # encode locations
    encoded_col = encode_features(data['policy_locations_peer_participants'].tolist())
    data['policy_locations_peer_participants'] = encoded_col

    # encode policy ration
    encoded_col = encode_features(data['policy_ratio_peer_participants'].tolist())
    data['policy_ratio_peer_participants'] = encoded_col

    encoded_col = encode_features(data['policy_contracts_peer_participants'].tolist())
    data['policy_contracts_peer_participants'] = encoded_col

    encoded_col = encode_features(data['info_type_peer_participants'].tolist())
    data['info_type_peer_participants'] = encoded_col

    return data

def tree_fs(df):
    labels = df['info_type_peer_participants'].tolist()
    data = df.drop(columns=['info_type_peer_participants'])
    old_cols = set(data.columns)
    # print(data.shape)

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(data, labels)
    # print(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    new_data = data[data.columns[model.get_support(indices=True)]]
    # print(new_data.shape)
    new_cols = set(new_data.columns)

    removed_col = old_cols.difference(new_cols)
    print('Tree-based feature selection removed the following set of features: ')
    print(removed_col)

    tup = new_data.shape
    col = tup[1]
    new_data.insert(col, 'type', labels)

    return new_data

def variance_fs(df):
    labels = df['info_type_peer_participants'].tolist()
    data = df.drop(columns=['info_type_peer_participants'])
    old_cols = set(data.columns)

    sel = VarianceThreshold(threshold=(0.9 * (1 - 0.9)))

    sel.fit(data)
    new_data = data[data.columns[sel.get_support(indices=True)]]
    new_cols = set(new_data.columns)

    removed_col = old_cols.difference(new_cols)
    print('Variance feature selection removed the following set of features: ')
    print(removed_col)

    tup = new_data.shape
    col = tup[1]
    new_data.insert(col, 'type', labels)

    return new_data

def p2_process(df):
    # shuffle rows of df
    df = shuffle(df)
    
    labels = df['type'].tolist()
    df = df.drop(columns=['type'])
    column_names = list(df.columns) 
    df = pd.DataFrame(df, columns=column_names)

    # normalize values
    scaler = preprocessing.MinMaxScaler()       # init normalizer
    df = scaler.fit_transform(df)               # normalize data
    df = pd.DataFrame(df, columns=column_names) # convert data back into a DF

    tup = df.shape
    col = tup[1]
    df.insert(col, 'type', labels)
    return df

def naive_bayes(df):
    labels = df['type'].tolist()
    data = df.drop(columns=['type'])
    features = list(df.columns)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=0)  

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    model = gnb.fit(train_data, train_labels)
    pred = model.predict(test_data)

    gnb_accuracy = accuracy_score(test_labels, pred)
    gnb_precision = precision_score(test_labels, pred, average='macro')
    gnb_recall = recall_score(test_labels, pred, average='macro')
    gnb_fscore = f1_score(test_labels, pred, average='macro')
    print('GNB accuracy = %f' %gnb_accuracy)
    print('GNB precision = %f' %gnb_precision)
    print('GNB recall = %f' %gnb_recall)
    print('GNB F-score = %f' %gnb_fscore)

if __name__ == "__main__":
    path = 'master.csv'
    df = pd.read_csv(path)
    df = p1_process(df)
    
    df = variance_fs(df)
    df = p2_process(df)
    naive_bayes(df)









