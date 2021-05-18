import torch
import random
from itertools import permutations 
from torch.utils.data import Dataset, DataLoader 
from torchvision.datasets import ImageFolder 
from torchvision.transforms import Compose, Grayscale, ToTensor

class Grid:
    def __init__(self):
        self.size = 4 # 4x4 grid (fixed)
        
        # Generate locations (tuples) for each state in grid
        locs = [(i,j) for i in range(self.size) for j in range(self.size)]
        
        # Define groups
        group1 = [(1,0),(2,0),(0,1),(3,1),(1,2),(2,2),(0,3),(3,3)]
        group2 = [loc for loc in locs if loc not in group1]
        
        # Generate all within-group pairs
        all_within1 = [pair for pair in permutations(group1, 2)]
        all_within2 = [pair for pair in permutations(group2, 2)]
        
        # Get rank distances for both axes in both groups
        d_within1_ax1 = [pair[0][0] - pair[1][0] for pair in all_within1]
        d_within1_ax2 = [pair[0][1] - pair[1][1] for pair in all_within1]
        d_within2_ax1 = [pair[0][0] - pair[1][0] for pair in all_within2]
        d_within2_ax2 = [pair[0][1] - pair[1][1] for pair in all_within2]
        
        # Create within-group samples, excluding pairs with distance > 1
        within = []
        # group 1
        for pair, d1, d2 in zip(all_within1, d_within1_ax1, d_within1_ax2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((f1, f2, 0, y)) # (F1, F2, axis, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((f1, f2, 1, y)) # (F1, F2, axis, y)
        # group 2
        for pair, d1, d2 in zip(all_within2, d_within2_ax1, d_within2_ax2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((f1, f2, 0, y)) # (F1, F2, axis, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((f1, f2, 1, y)) # (F1, F2, axis, y)
        
        # Between-group "hub" pairs
        hubs_ax1 = [(1,0), (2,2), (1,1), (2,3)]
        hubs_ax2 = [(1,2), (3,1), (0,2), (2,1)]
        hub_pairs_ax1 = [((2,2),(1,1)),((2,2),(1,3)),
                         ((2,2),(3,0)),((2,2),(3,2)),
                         ((1,0),(0,0)),((1,0),(0,2)),
                         ((1,0),(2,1)),((1,0),(2,3)),
                         ((2,3),(1,0)),((2,3),(1,2)),
                         ((2,3),(3,1)),((2,3),(3,3)),
                         ((1,1),(0,1)),((1,1),(0,3)),
                         ((1,1),(2,0)),((1,1),(2,2))]
        hub_pairs_ax2 = [((1,2),(1,1)),((1,2),(1,3)),
                         ((1,2),(2,1)),((1,2),(2,3)),
                         ((3,1),(0,0)),((3,1),(0,2)),
                         ((3,1),(3,0)),((3,1),(3,2)),
                         ((0,2),(0,1)),((0,2),(0,3)),
                         ((0,2),(3,1)),((0,2),(3,3)),
                         ((2,1),(1,0)),((2,1),(1,2)),
                         ((2,1),(2,0)),((2,1),(2,2))]
        
        # Add in reversals of each pair
        between_ax1 = []
        for pair in hub_pairs_ax1:
            between_ax1.append((pair[0],pair[1]))
            between_ax1.append((pair[1],pair[0]))
        between_ax2 = []
        for pair in hub_pairs_ax2:
            between_ax2.append((pair[0],pair[1]))
            between_ax2.append((pair[1],pair[0]))
        
        # Get rank distances for both axes in both groups
        d_between_ax1 = [pair[0][0] - pair[1][0] for pair in between_ax1]
        d_between_ax2 = [pair[0][1] - pair[1][1] for pair in between_ax2]
        not_one_over_ax1 = [d for d in d_between_ax1 if not abs(d) == 1]
        not_one_over_ax2 = [d for d in d_between_ax2 if not abs(d) == 1]
        msg1 = "{} are not one-over!".format(not_one_over_ax1)
        assert len(not_one_over_ax1) == 0, msg1
        msg2 = "{} are not one-over!".format(not_one_over_ax2)
        assert len(not_one_over_ax2) == 0, msg2
        
        # Create samples from between-group "hub" pairs
        between = []
        for pair, d1 in zip(between_ax1, d_between_ax1):
            f1 = pair[0]
            f2 = pair[1]
            y = int(d1 > 0)
            between.append((f1, f2, 0, y))
        for pair, d2 in zip(between_ax2, d_between_ax2):
            f1 = pair[0]
            f2 = pair[1]
            y = int(d2 > 0)
            between.append((f1, f2, 1, y))
        
        # Compile training set
        train = within + between
        random.shuffle(train)
        
        # Get all test pairs separated by a hub
        test_pairs_ax1 = []
        for pair in between_ax1:
            locb1 = pair[0]
            locb2 = pair[1]
            for sample in within:
                if sample[2] != 0:
                    continue
                locw1 = sample[0]
                locw2 = sample[1]
                if locb1 == locw1:
                    test_pairs_ax1.append((locb2, locw2))
                    test_pairs_ax1.append((locw2, locb2))
                if locb1 == locw2:
                    test_pairs_ax1.append((locb2, locw1))
                    test_pairs_ax1.append((locw1, locb2))
                if locb2 == locw1:
                    test_pairs_ax1.append((locb1, locw2))
                    test_pairs_ax1.append((locw2, locb1))
                if locb2 == locw2:
                    test_pairs_ax1.append((locb1, locw1))
                    test_pairs_ax1.append((locw1, locb1))            
        test_pairs_ax1 = list(set(test_pairs_ax1))
        
        test_pairs_ax2 = []
        for pair in between_ax2:
            locb1 = pair[0]
            locb2 = pair[1]
            for sample in within:
                if sample[2] != 1:
                    continue
                locw1 = sample[0]
                locw2 = sample[1]
                if locb1 == locw1:
                    test_pairs_ax2.append((locb2, locw2))
                    test_pairs_ax2.append((locw2, locb2))
                if locb1 == locw2:
                    test_pairs_ax2.append((locb2, locw1))
                    test_pairs_ax2.append((locw1, locb2))
                if locb2 == locw1:
                    test_pairs_ax2.append((locb1, locw2))
                    test_pairs_ax2.append((locw2, locb1))
                if locb2 == locw2:
                    test_pairs_ax2.append((locb1, locw1))
                    test_pairs_ax2.append((locw1, locb1))            
        test_pairs_ax2 = list(set(test_pairs_ax2))
        
        # Remove pairs that include an axis 1 hub
        test_pairs_nohub_ax1 = []
        for loc1, loc2 in test_pairs_ax1:
            if loc1 not in hubs_ax1 and loc2 not in hubs_ax1:
                test_pairs_nohub_ax1.append((loc1, loc2))
        # Remove pairs that include an axis 2 hub
        test_pairs_nohub_ax2 = []
        for loc1, loc2 in test_pairs_ax2:
            if loc1 not in hubs_ax2 and loc2 not in hubs_ax2:
                test_pairs_nohub_ax2.append((loc1, loc2))
                
        # Create test set, removing 0s
        test = []
        for pair in test_pairs_nohub_ax1:
            loc1 = pair[0]
            loc2 = pair[1]
            d = loc1[0] - loc2[0]
            if d != 0:
                f1 = loc1
                f2 = loc2
                y = int(d > 0)
                test.append((f1, f2, 0, y))
        for pair in test_pairs_nohub_ax2:
            loc1 = pair[0]
            loc2 = pair[1]
            d = loc1[1] - loc2[1]
            if d != 0:
                f1 = loc1
                f2 = loc2
                y = int(d > 0)
                test.append((f1, f2, 1, y))
                
        # Get relevant hub pairs for each sample in test set
        hub_sample_ids = [] # ids of relevant samples
        for sample in test:
            loc1 = sample[0]
            loc2 = sample[1]
            axis = sample[2]
            possible_hubs_f1 = []
            possible_hubs_f2 = []
            for i, s in enumerate(train):
                if s[2] == axis:
                    if loc1 == s[0]:
                        possible_hubs_f1.append((s[1], i))
                    elif loc1 == s[1]:
                        possible_hubs_f1.append((s[0], i))
                    if loc2 == s[0]:
                        possible_hubs_f2.append((s[1], i))
                    elif loc2 == s[1]:
                        possible_hubs_f2.append((s[0], i))
            sample_ids = []
            for loc, i in possible_hubs_f1:
                if loc in [p[0] for p in possible_hubs_f2]:
                    sample_ids.append(i)
            for loc, i in possible_hubs_f2:
                if loc in [p[0] for p in possible_hubs_f1]:
                    sample_ids.append(i)
            hub_sample_ids.append(list(set(sample_ids)))
        
        # Save variables
        self.locs = locs
        self.group1 = group1
        self.group2 = group2
        self.within = within
        self.between = between
        self.train = train
        self.test = test
        self.hub_sample_ids = hub_sample_ids
        
        self.n_train = len(self.train)
        self.n_test = len(self.test)

class GridDataset(Dataset):
    """
    Dataset used for cortical system. Each sample is a tuple (f1, f2, axis, y):
        f1   : face 1, either int to be embedded or an image (1, 64, 64)
        f2   : face 2, either int to be embedded or an image (1, 64, 64)
        axis : axis variable, always an int to be embedded 
        y    : correct answer, always either 0 or 1
    """
    def __init__(self, testing, use_images, image_dir = None):
        self.testing = testing       # use test set
        self.use_images = use_images # use images rather than one-hot vectors
        self.image_dir = image_dir   # directory with images
        self.grid = Grid()

        # Create 1 fixed mapping from locs to idxs
        locs = self.grid.locs 
        idxs = [idx for idx in range(len(locs))]
        self.loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}
        self.n_states = len(idxs)

        # Prepare tensors for each idx
        idx2tensor = {}
        if self.use_images:
            # Tensors are images
            transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
            face_images = ImageFolder(self.image_dir, transform)
            for idx in idxs:
                idx2tensor[idx] = face_images[idx][0] # [1, 64, 64]
        else:
            # Tensors are one-hot vectors
            for idx in idxs:
                idx2tensor[idx] = torch.tensor(idx).type(torch.long) # [16]
        self.idx2tensor = idx2tensor
    
    def __len__(self):
        if self.testing:
            return len(self.grid.test)
        else:
            return len(self.grid.train)
    
    def __getitem__(self, i):
        if self.testing:
            s = self.grid.test[i]
        else:
            s = self.grid.train[i]
        loc1 = s[0]
        loc2 = s[1]
        idx1 = self.loc2idx[loc1]
        idx2 = self.loc2idx[loc2]
        f1 = self.idx2tensor[idx1].unsqueeze(0) # [1, 1, 64, 64]
        f2 = self.idx2tensor[idx2].unsqueeze(0) # [1, 1, 64, 64]
        axis = torch.tensor(s[2]).type(torch.long).unsqueeze(0) # [1]
        y = torch.tensor(s[3]).unsqueeze(0).unsqueeze(0) # [1]
        y = y.type(torch.long)
        return f1, f2, axis, y


class GridMetaDataset(Dataset):
    """
    Dataset used for episodic system.
    Returns a tuple of (training_episode, testing episode):
        training_episode : [n_train, sample_dim]
        testing_episode  : [n_test, sample_dim]
    Each episode has an entire training set and testing set for one transitive
    inference problem. The model is trained on many such problems.
    """
    def __init__(self, testing, n_episodes=10000):
        self.testing = testing        # use test set
        self.n_episodes = n_episodes  # number of episodes (meta-learning)
        
        self.grid = Grid()
        locs = self.grid.locs
        idxs = [idx for idx in range(len(locs))]
        self.n_states = len(idxs)

        # Save 1 mapping from idxs to locs for testing
        self.test_loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}

        # Create n_episodes random mappings from locs to idxs for meta-training
        train_loc2idx = []
        while len(train_loc2idx) < n_episodes:
            random.shuffle(idxs)
            l2i= {loc:idx for loc, idx in zip(locs, idxs)}
            if l2i != self.test_loc2idx: # ensure different from test
                train_loc2idx.append(l2i)
        self.train_loc2idx = train_loc2idx 

        # Tensors are one-hot vectors
        n_states = len(idxs)
        idx2tensor = {}
        for idx in idxs:
            idx2tensor[idx] = torch.eye(n_states)[idx].type(torch.long) # [16]
        self.idx2tensor = idx2tensor

    def __len__(self):
        if self.testing:
            return 1 # one sample for testing
        else:
            return self.n_episodes
    
    def __getitem__(self, i):
        axis_dim = 2
        y_dim = 1
        sample_dim = 2*self.n_states + axis_dim + y_dim

        if self.testing:
            loc2idx = self.test_loc2idx # only 1 random mapping
        else:
            loc2idx = self.train_loc2idx[i] # sample i = ith random mapping
        
        # Training episode
        train_episode = torch.zeros(self.grid.n_train, sample_dim)
        for s_i, s in enumerate(self.grid.train):
            loc1 = s[0]
            loc2 = s[1]
            idx1 = loc2idx[loc1]
            idx2 = loc2idx[loc2]
            f1 = self.idx2tensor[idx1] # [16]
            f2 = self.idx2tensor[idx2] # [16]
            axis = torch.eye(2)[s[2]].type(torch.long)   # [2]
            y = torch.tensor(s[3]).unsqueeze(0).type(torch.long) # [1]
            sample = torch.cat([f1, f2, axis, y], dim=0) # [sample_dim]
            train_episode[s_i] = sample
        train_episode = train_episode.unsqueeze(0) # [1, n_train, sample_dim]
        
        # Testing episode
        test_episode = torch.zeros(self.grid.n_test, sample_dim)
        for s_i, s in enumerate(self.grid.test):
            loc1 = s[0]
            loc2 = s[1]
            idx1 = loc2idx[loc1]
            idx2 = loc2idx[loc2]
            f1 = self.idx2tensor[idx1] # [16]
            f2 = self.idx2tensor[idx2] # [16]
            axis = torch.eye(2)[s[2]].type(torch.long)  # [2]
            y = torch.tensor(s[3]).unsqueeze(0).type(torch.long) # [1]
            sample = torch.cat([f1, f2, axis, y], dim=0) # [sample_dim]
            test_episode[s_i] = sample
        test_episode = test_episode.unsqueeze(0) # [1, n_test, sample_dim]
            
        return train_episode, test_episode

# Collate functions
def meta_collate(samples):
    train_batch = torch.cat([s[0] for s in samples], dim=0)
    test_batch = torch.cat([s[1] for s in samples], dim=0)
    return train_batch, test_batch

def grid_collate(samples):
    f1_batch = torch.cat([s[0] for s in samples], dim=0)
    f2_batch = torch.cat([s[1] for s in samples], dim=0)
    ax_batch = torch.cat([s[2] for s in samples], dim=0)
    y_batch = torch.cat([s[3] for s in samples], dim=0)
    return f1_batch, f2_batch, ax_batch, y_batch

def get_loaders(batch_size, meta, use_images, image_dir, n_episodes):
    if meta:
        # Train
        train_data = GridMetaDataset(testing=False, n_episodes=n_episodes)
        train_loader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True, collate_fn=meta_collate) 
        # Test
        test_data = GridMetaDataset(testing=True, n_episodes=n_episodes)
        test_loader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, collate_fn=meta_collate)
    else:
        # Train
        train_data = GridDataset(testing=False, use_images=use_images, 
                                 image_dir=image_dir)
        train_loader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True, collate_fn=grid_collate)
        # Test
        test_data = GridDataset(testing=True, use_images=use_images, 
                                image_dir=image_dir)
        test_loader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, collate_fn=grid_collate)
    return train_data, train_loader, test_data, test_loader

