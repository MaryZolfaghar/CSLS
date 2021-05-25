import torch 
import numpy as np

from itertools import combinations
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def analyze_episodic(model, test_data, args):
    # Collect attention weights for each sample in test set
    model.eval()
    m, x_ = test_data[0] # only 1 episode in test data
    m = m.to(args.device) # m: [1, n_train, sample_dim]
    x = x_[:,:,:-1].to(args.device) # x: [1, n_test, sample_dim]
    y = x_[:,:,-1].type(torch.long).to(args.device)
    y = y.squeeze() # y: [1, n_test]
    with torch.no_grad():
        y_hat, attention = model(x, m) 
        attention = attention[0] # first (only) memory layer
        attention = np.squeeze(attention)
        # attention: [n_train, n_test]
    
    # Check the retrieval weights of relevant vs. irrelevant training samples
    grid = test_data.grid
    train = grid.train # train *samples* in test *episode*
    test = grid.test   # test *samples* in test *episode*
    n_train = len(train)
    n_test = len(test)
    rel_ids = grid.hub_sample_ids # relevant memory ids (train samples)
    attn_ranks = np.zeros_like(attention)
    for i in range(n_test):
        argsorted_attn = np.argsort(attention[i])
        ranks = np.zeros([n_train])
        ranks[argsorted_attn] = np.arange(n_train)
        attn_ranks[i] = ranks
    relevant = []
    irrelevant = []
    for i in range(n_test):
        for j in range(n_train):
            if j in rel_ids[i]:
                relevant.append(attn_ranks[i,j])
            else:
                irrelevant.append(attn_ranks[i,j])
    rank_data = {"relevant": relevant, "irrelevant": irrelevant}

    # Check how often a legitimate "path" was retrieved in the top 5%
    k = 8 # top k memories with highest weights (k = 8 means 5 percent)
    used_hub = []
    for i in range(n_test):
        highest_attn = np.argsort(attention[i])[-k:]
        test_f1, test_f2, test_ax, test_y = test[i]

        # Get relevant hubs for current test sample
        hubs = []
        for rel_id in rel_ids[i]:
            train_sample = train[rel_id]
            train_f1, train_f2 = train_sample[0], train_sample[1]
            if train_f1 in [test_f1, test_f2]: 
                hubs.append(train_f2)
            if train_f2 in [test_f1, test_f2]:
                hubs.append(train_f1)
        hubs = list(set(hubs))
        hubs_dict = {h:[] for h in hubs}
        assert len(hubs) == 2, "shouldn't be more than 2 hubs?"

        # Check if one of the hubs appears with f1 and f2
        attended_train = [train[idx] for idx in highest_attn]
        for sample in attended_train:
            train_f1, train_f2, train_ax, train_y = sample
            if train_ax != test_ax:
                continue # must be samples testing the same axis to be relevant
            if hubs[0] == train_f1:
                hubs_dict[hubs[0]].append(sample[1])
            if hubs[1] == sample[0]:
                hubs_dict[hubs[1]].append(sample[1])
            if hubs[0] == sample[1]:
                hubs_dict[hubs[0]].append(sample[0])
            if hubs[1] == sample[1]:
                hubs_dict[hubs[1]].append(sample[0])
        if test_f1 in hubs_dict[hubs[0]] and test_f2 in hubs_dict[hubs[0]]:
            used_hub.append(True)
        elif test_f1 in hubs_dict[hubs[1]] and test_f2 in hubs_dict[hubs[1]]:
            used_hub.append(True)
        else:
            used_hub.append(False)
    p_used_hub = np.mean(used_hub)
    print("Proportion that episodic system retrieved a hub path:", p_used_hub)

    results = {"rank_data":rank_data, "p_used_hub": p_used_hub}
    return results

def analyze_cortical(model, test_data, args):
    # Useful dictionaries from test dataset
    n_states = test_data.n_states 
    loc2idx = test_data.loc2idx 
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    idxs = [idx for idx in range(n_states)]
    locs = [idx2loc[idx] for idx in idxs]
    idx2tensor = test_data.idx2tensor 

    # Get embeddings from model for each face
    model.eval()
    face_embedding = model.face_embedding
    face_embedding.to(args.device)
    embeddings = []
    with torch.no_grad():
        for idx in range(n_states):
            face_tensor = idx2tensor[idx].unsqueeze(0).to(args.device)
            embedding = face_embedding(face_tensor) # [1, state_dim]
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0) # [n_states, state_dim]

    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(embeddings)
    pca_results = {'grid_locations': locs, 'pca_2d': pca_2d}

    # Correlation
    grid_distances = []
    embed_distances = []
    for idx1, idx2 in combinations(idxs, 2):
        # Ground-truth distance in grid
        (x1, y1), (x2, y2) = idx2loc[idx1], idx2loc[idx2]
        grid_distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        grid_distances.append(grid_distance)
        # Euclidean distance between embeddings
        emb1, emb2 = embeddings[idx1], embeddings[idx2]
        embed_distance = np.linalg.norm(emb1 - emb2)
        embed_distances.append(embed_distance)
    grid_distances = np.array(grid_distances)
    embed_distances = np.array(embed_distances)
    r, p_val = pearsonr(grid_distances, embed_distances)
    correlation_results = {'r': r, 'p_val': p_val}

    results = {'pca': pca_results, 'correlation': correlation_results}

    return results
