'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    
    prev_user = 0
    cur = []
    metrix = []

    for i in range(len(_testRatings)):
        user, item = _testRatings[i][0], _testRatings[i][1]
        if user != prev_user:
            metrix.append(eval_one_rating1(user, cur))
            cur = []
            prev_user = user
        cur.append(item)
    metrix.append(eval_one_rating1(user, cur))

    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(_testRatings[idx][0])
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs, metrix)

def eval_one_rating1(idx, cur):

    u = idx
    # print("u idx", u, idx)
    # assert(u == idx)
    items = _testNegatives[idx]

    gtItems = cur
    items.extend(gtItems)

    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], batch_size=114, verbose=0)

    map_item_score = {}

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    metx = getMetxRatio(ranklist, gtItems)
    return metx

def getMetxRatio(ranklist, gtItems):
    return len(list(set(ranklist) & set(gtItems))) / len(gtItems)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    
    # print("rating", rating)
    # print("items", items)

    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # print("gt item", gtItem)
    # print("u", u)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    # print('users', users)

    predictions = _model.predict([users, np.array(items)], batch_size=100, verbose=0)

    # predictions = predictions[0]
    # print('predictions[0]', predictions[0])
    # print('predictions[1]', predictions[1])
    # print("len predictions[0]", len(predictions[0]))
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
