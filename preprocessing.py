# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse

dat_f = open('./raw_data/ratings.csv', encoding='utf-8')
sentimentList = []
for line in dat_f:
    s = line.strip().split(',')
    sentimentList.append(s)
dat_f.close()
data = pd.DataFrame(sentimentList).iloc[:, :3]
data.columns = ['user_id', 'item_id', 'ratings']
print('data loaded...')


def get_count(tp, key):
    count_groupBy_key = tp[[key, 'ratings']].groupby(key, as_index=True)
    count = count_groupBy_key.size()
    return count


MIN_USER_COUNT = 1
MIN_ITEM_COUNT = 1


def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_ic=MIN_ITEM_COUNT):
    """Only keep the triplets for items which were rated by at least min_ic users."""
    _item_count = get_count(tp, 'item_id')
    tp = tp[tp['item_id'].isin(_item_count.index[_item_count >= min_ic])]
    '''Only keep the triplets for users who listened to at least min_uc songs
    After doing this, some of the songs will have less than min_uc users, 
    but should only be a small proportion'''
    _user_count = get_count(tp, 'user_id')
    tp = tp[tp['user_id'].isin(_user_count.index[_user_count >= min_uc])]
    _item_count = get_count(tp, 'item_id')
    _user_count = get_count(tp, 'user_id')

    return tp, _item_count, _user_count


data, item_count, user_count = filter_triplets(data)

unique_uid = user_count.index
unique_iid = item_count.index

user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))

user_num = len(user2id)  # total number of users
item_num = len(item2id)  # total number of items
interaction_num = data.shape[0]  # total number of interactions

print('user_num: %d, item_num: %d, interaction_num: %d' % (user_num, item_num, interaction_num))


def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    iid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(iid)
    return tp


csv_data = numerize(data)

df_gp = csv_data.groupby(['user_id'])
uid_name = df_gp.size().index


def split_train_test(tp_rating):
    n_ratings = tp_rating.shape[0]
    test = np.random.choice(n_ratings, size=int(0.30 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    _tp_test = tp_rating[test_idx]
    _tp_train = tp_rating[~test_idx]
    return _tp_train, _tp_test


edge_index = np.empty(shape=[0, 2], dtype=int)
train_ui = np.empty(shape=[0, 2], dtype=int)
test_ui = np.empty(shape=[0, 2], dtype=int)

print('edge_index train_ui test_ui started...')
for name in uid_name:
    tp_train, tp_test = split_train_test(df_gp.get_group(name))
    tr_ui = np.array(tp_train[['user_id', 'item_id']])
    te_ui = np.array(tp_test[['user_id', 'item_id']])
    edge = tr_ui + np.array([0, user_num])
    re_edge = edge[:, [1, 0]]
    edge_index = np.append(edge_index, edge, axis=0)
    edge_index = np.append(edge_index, re_edge, axis=0)
    train_ui = np.append(train_ui, tr_ui, axis=0)
    test_ui = np.append(test_ui, te_ui, axis=0)
print('edge_index train_ui test_ui finished...')

print('train_matrix test_matrix started...')
row = train_ui[:, 0]
col = train_ui[:, 1]
data = np.ones_like(row)
train_matrix = sparse.coo_matrix((data, (row, col)), shape=(user_num, item_num), dtype=np.int8)
row = test_ui[:, 0]
col = test_ui[:, 1]
data = np.ones_like(row)
test_matrix = sparse.coo_matrix((data, (row, col)), shape=(user_num, item_num), dtype=np.int8)
print('train_matrix test_matrix constructed...')

para = {'user_num': user_num, 'item_num': item_num, 'edge_index': edge_index, 'train_matrix': train_matrix,
        'test_matrix': test_matrix, 'train_ui': train_ui}
pickle.dump(para, open('./para/movie_load.para', 'wb'))
print('data_load finished...')
