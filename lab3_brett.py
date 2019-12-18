import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree.export import export_text
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
    data = df[['asn_peer_participants', 'proto_unicast_mgmt_publics', \
                'public_id_mgmt_publics_ips', 'proto_multicast_mgmt_publics', \
                'info_traffic_peer_participants', 'info_ratio_peer_participants', \
                'info_scope_peer_participants', 'info_prefixes_peer_participants', \
                'policy_general_peer_participants', 'policy_locations_peer_participants', \
                'policy_ratio_peer_participants', 'policy_contracts_peer_participants', \
                'proto_ipv6_mgmt_publics', 'info_type_peer_participants']]              

    data = data.dropna(axis=0)  # delete all rows with nans
    data = data.replace(['Not Disclosed'], np.nan).dropna(axis=0)
    data = conv_traffic(data)   # convert traffic to floats

    to_encode_list = ['info_ratio_peer_participants', 'info_scope_peer_participants', \
                    'policy_general_peer_participants', 'policy_locations_peer_participants', \
                    'policy_ratio_peer_participants', 'policy_contracts_peer_participants', \
                    'info_type_peer_participants']
    
    for col in to_encode_list:
        encoded_col = encode_features(data[col].tolist())
        data[col] = encoded_col

    print(data.shape)

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

    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))

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

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=0)  

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    model = gnb.fit(train_data, train_labels)
    pred = model.predict(test_data)

    gnb_accuracy = accuracy_score(test_labels, pred)
    gnb_precision = precision_score(test_labels, pred, average='weighted')
    gnb_recall = recall_score(test_labels, pred, average='weighted')
    gnb_fscore = f1_score(test_labels, pred, average='weighted')
    print('GNB accuracy = %f' %gnb_accuracy)
    print('GNB precision = %f' %gnb_precision)
    print('GNB recall = %f' %gnb_recall)
    print('GNB F-score = %f' %gnb_fscore)

def decision_tree(df):
    labels = df['type'].tolist()
    data = df.drop(columns=['type'])
    features = list(data.columns)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=0)  

    ### Decision Tree ###
    d_tree = tree.DecisionTreeClassifier()
    model = d_tree.fit(train_data, train_labels)
    pred = model.predict(test_data, test_labels)

    taccuracy = accuracy_score(test_labels, pred)
    tprecision = precision_score(test_labels, pred, average='weighted')
    trecall = recall_score(test_labels, pred, average='weighted')
    tfscore = f1_score(test_labels, pred, average='weighted')
    print('DTree accuracy = %f' %taccuracy)
    print('DTree precision = %f' %tprecision)
    print('DTree recall = %f' %trecall)
    print('DTree F-score = %f' %tfscore)

    # generate visual representation of decision tree
    r = export_text(model, feature_names=features)
    txt = open('decision_tree.txt', 'w')
    txt.write(r)
    txt.close()

def random_forest(df):
    labels = df['type'].tolist()
    data = df.drop(columns=['type'])

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=0)  

    ### Random Forest ###
    rf = RandomForestClassifier(random_state=0)
    model = rf.fit(train_data, train_labels)
    pred = model.predict(test_data)

    rfaccuracy = accuracy_score(test_labels, pred)
    rfprecision = precision_score(test_labels, pred, average='weighted')
    rfrecall = recall_score(test_labels, pred, average='weighted')
    rffscore = f1_score(test_labels, pred, average='weighted')
    print('RF accuracy = %f' %rfaccuracy)
    print('RF precision = %f' %rfprecision)
    print('RF recall = %f' %rfrecall)
    print('RF F-score = %f' %rffscore)

def ann(df):
    
    labels = df['type'].tolist()
    data = df.drop(columns=['type'])
    num_labels = len(set(labels))  # get number of unique labels
    num_features = len(list(data.columns))  # get number of unique features

    data = data.values.tolist()

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=0) 
    
    ### Artificial Neural Net ###
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(num_features, activation='relu'), tf.keras.layers.Dense(100, activation='relu'), tf.keras.layers.Dense(num_labels, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5, verbose=2)
    predictions = model.predict(test_data)
    scores = model.evaluate(test_data, test_labels, verbose=2)

def test(df):
    label_dict = {}
    nan_count = 0
    for i in range(len(df)):
        if df.loc[i, 'info_type_peer_participants'] != df.loc[i, 'info_type_peer_participants']:
            nan_count += 1
        else:
            if df.loc[i, 'info_type_peer_participants'] not in label_dict:
                label_dict[df.loc[i,'info_type_peer_participants']] = 1
            else:
                label_dict[df.loc[i,'info_type_peer_participants']] += 1
    print(label_dict)
    print(nan_count)


if __name__ == "__main__":
    path = 'master.csv'
    df = pd.read_csv(path)

    df = p1_process(df)
    
    # df = variance_fs(df)
    df = tree_fs(df)

    df = p2_process(df)
    
    random_forest(df)
    # naive_bayes(df)
    # decision_tree(df)
    # ann(df)

    # need to know what three labels we're focusing on.
    # need to know how removing NaN's









