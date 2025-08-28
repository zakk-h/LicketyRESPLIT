from Tree import Node, Leaf
from itertools import product
from collections import Counter
import bisect
import numpy as np

class SplitNode:
    __slots__ = ('feature', 'left', 'right', 'num_valid_trees')
    def __init__(self, feature, left, right):
        self.feature = feature
        self.left = left    # TreeTrieNode
        self.right = right  # TreeTrieNode
        self.num_valid_trees = None  # number of trees in this split <= parent budget

class TreeTrieNode:
    __slots__ = ('budget', 'children', 'min_objective', 'objectives')
    def __init__(self, budget):
        self.budget = budget
        self.children = []
        self.min_objective = float('inf') # need min objective to reduce the budget of the other side in LicketyRESPLIT
        self.objectives = Counter()

    def add_leaf(self, leaf):
        self.children.append(leaf)
        if leaf.loss < self.min_objective:
            self.min_objective = leaf.loss
        self.objectives[leaf.loss] += 1

    def add_split(self, feature, left_node, right_node): # left and right subtrie are given
        split = SplitNode(feature, left_node, right_node)
        self.children.append(split)
        min_sum = left_node.min_objective + right_node.min_objective # independence
        if min_sum < self.min_objective:
            self.min_objective = min_sum

        total_valid = 0
        for l_loss, l_count in left_node.objectives.items():
            for r_loss, r_count in right_node.objectives.items():
                tot = l_loss + r_loss
                if tot <= self.budget:
                    count_here = l_count * r_count
                    self.objectives[tot] += count_here
                    total_valid += count_here
        split.num_valid_trees = total_valid
      
    def count_trees(self):
        return sum(self.objectives.values())

    def is_empty(self):
        return not self.objectives

    def __bool__(self):
        return not self.is_empty()
    
    def __len__(self):
        return self.count_trees()
    
    def __getitem__(self, i): # be warned not the same order as list
        return self.get_ith_tree(i) # 0-index
    
    def __iter__(self):
        return self.iter_trees()
    
    def iter_trees(self): # depending on your need, consider list_trees
        total = self.count_trees()
        for i in range(total):
            yield self.get_ith_tree(i)
        
    def pick_child_for_ith_tree(self, i):
        cum = 0
        # there's not that many children so i think this linear scan is fine
        for child in self.children:
            if isinstance(child, Leaf):
                sz = 1
            elif isinstance(child, SplitNode):
                sz = child.num_valid_trees
            else:
                raise Exception("Unknown child type")
            if i < cum + sz:
                local_i = i - cum
                return child, local_i
            cum += sz
        raise IndexError(f"Index {i} out of range (total={cum})")
    
    @staticmethod
    def unrank_cross_product(left, right, budget, local_i):
        left_losses = sorted(left.objectives) # keys of the map keys
        right_losses = sorted(right.objectives)
        right_counts = [right.objectives[r] for r in right_losses] # PDF
        right_cumsum = [0]
        for cnt in right_counts: # CDF
            right_cumsum.append(right_cumsum[-1] + cnt)

        # now enumerate pairs in left_outer, right_inner order
        pair_counter = 0
        for l_obj in left_losses:
            l_count = left.objectives[l_obj]
            # how many right losses can pair with this l_obj?
            max_r = budget - l_obj # all r_obj <= max_r are allowed
            # right_losses is sorted, so binary search:
            r_hi = bisect.bisect_right(right_losses, max_r) # finds the index in right to put max_r to maintain sorted order with insertion to the right of any existing entries equal to max_r.
            if r_hi == 0: # everything to the left is include, so nothing to the left means nothing is valid
                continue
            total_pairs = l_count * right_cumsum[r_hi] # how many things at that L times how many things at that biggest R that fits or a smaller R
            if pair_counter + total_pairs > local_i: # the pair is in this left_obj bucket. so we've identified the left loss, but not yet the left index without that loss, or anything about the right
                rel_i = local_i - pair_counter # shift to be local to this pair. you know it is occuring in this left loss but not what right loss.
                # imagine a table of size l_count * right_cumsum[r_hi]. mod is giving you the column, division the row.
                left_sub_idx = rel_i // right_cumsum[r_hi]
                right_sub_idx = rel_i % right_cumsum[r_hi]
                # now, which actual right_obj does right_sub_idx fall into? We want the right_sub_idx out of all of them
                # i.e. we have a flattened index (that goes the list with duplicates) and need to map it to an x and y position on the histogram
                # rcum = 0
                # for r_idx in range(r_hi): # iterate eligible right objectives
                #     cnt = right_counts[r_idx] # right PDF at that objective that corresponds to the index
                #     if rcum + cnt > right_sub_idx: # we want to return what objective it is that the tree is in, and which numbered tree of that same objective (for left and right, that's why there are 4 returns)
                #         # if our index falls in this bucket, we shift it to start at 0, and also return the index for the objective it fell in in the histogram (and same for left)
                #         return (l_obj, left_sub_idx, right_losses[r_idx], right_sub_idx - rcum)
                #     rcum += cnt
                # binary search to find which right objective bucket
                r_idx = bisect.bisect_right(right_cumsum, right_sub_idx) - 1 # finds the first CDF entry strictly greater than right_sub_idx and subtracts 1. remember, the idx is a tree number for a flattened list.
                r_obj = right_losses[r_idx] # subtracting 1 means you get the CDF index to look in (that range contains right_sub_idx), so we store the objective.
                idx_in_r = right_sub_idx - right_cumsum[r_idx] # shift to count trees from the start of it
                return (l_obj, left_sub_idx, r_obj, idx_in_r)
            pair_counter += total_pairs
        raise IndexError("Index out of range in cross-product unranking")
    
    def get_kth_tree_with_objective(self, target_obj, k):
        for child in self.children:
            if isinstance(child, Leaf):
                if child.loss == target_obj:
                    if k == 0:
                        return child
                    k -= 1  # skip this one
            else: # SplitNode
                L, R = child.left, child.right
                # count how many pairs under this split achieve exactly target_obj
                total_here = 0
                for l_obj in sorted(L.objectives): # sorts the keys [objectives]
                    r_obj = target_obj - l_obj # need to add up exactly to objective
                    rc = R.objectives.get(r_obj, 0)
                    if rc: # non-zero
                        total_here += L.objectives[l_obj] * rc

                if k < total_here: # the desired tree lies under this split. need to find which left-loss bucket l_obj it falls into.
                    # now we know the split
                    running = 0
                    for l_obj in sorted(L.objectives):
                        r_obj = target_obj - l_obj
                        rc = R.objectives.get(r_obj, 0)
                        if not rc:
                            continue
                        pairs = L.objectives[l_obj] * rc # for an l loss, how many exact matches.
                        if running + pairs > k: # if adding the number of trees with this objective exceeds the index we are looking, then it must be in what we just added.
                            # local index within this (l_obj, r_obj) block/bucket
                            # now we know both the left and the right objective - we just need the "density" number
                            rel = k - running
                            # consider a table of lc by rc.
                            left_idx  = rel // rc     # which left-tree among those with loss l_obj
                            right_idx = rel %  rc     # which right-tree among those with loss r_obj
                            # now we want the left_idx tree
                            left_tree  = L.get_kth_tree_with_objective(l_obj,  left_idx)
                            right_tree = R.get_kth_tree_with_objective(r_obj, right_idx)
                            return Node(feature=child.feature,
                                        left_child=left_tree,
                                        right_child=right_tree,
                                        loss=target_obj)
                        running += pairs
                else:
                    k -= total_here  # skip all trees from this split that achieve target_obj, too soon
        raise IndexError(f"Index {k} out of range for objective {target_obj} " f"(total={self.objectives.get(target_obj, 0)})")

    def get_ith_tree(self, i): # i think probably i can make this be in order of objective
        total = self.count_trees()
        if i < 0 or i >= total:
            raise IndexError(f"Index {i} out of range (total={total})")

        # what split holds the ith tree that fits in the budget.
        child, local_i = self.pick_child_for_ith_tree(i)

        if isinstance(child, Leaf):
            return child

        # child is a SplitNode. find the objective and number within that objective for both the L and R to be the local_ith tree at that split.
        # you want the local_ith tree from that split. so figure out what objetive and number within that objective the tree would be for the trees in LTrieNode and RTrieNode.
        L, R = child.left, child.right
        l_obj, left_sub_idx, r_obj, right_sub_idx = self.unrank_cross_product(L, R, self.budget, local_i)

        # given the objective and tree number for a subproblem, we can figure out what split it was, what the L and R objectives are, and the density number to recurse and get the whole tree.
        left_tree  = L.get_kth_tree_with_objective(l_obj,  left_sub_idx)
        right_tree = R.get_kth_tree_with_objective(r_obj, right_sub_idx)

        return Node(feature=child.feature, left_child=left_tree, right_child=right_tree, loss=l_obj + r_obj)

    def list_trees(self):
        result = []
        for child in self.children:
            if isinstance(child, Leaf):
                result.append(child)
            else:
                L, R = child.left, child.right
                lefts  = L.list_trees()
                rights = R.list_trees()
                for l_sub, r_sub in product(lefts, rights):
                    tot = l_sub.loss + r_sub.loss
                    if tot <= self.budget:
                        node = Node(feature=child.feature,
                                    left_child=l_sub,
                                    right_child=r_sub, loss=tot)
                        result.append(node)
        return result

    def get_predictions(self, i, X):
        tree = self.get_ith_tree(i)
        return self._predict_tree(tree, X)

    @staticmethod
    def _predict_tree(tree, X):
        n = X.shape[0]
        out = np.empty(n, dtype=np.uint8)

        def rec(node, idx):
            if isinstance(node, Leaf):
                out[idx] = np.uint8(node.prediction)
                return
            f = node.feature
            left_mask = X[idx, f]
            if np.any(left_mask):
                rec(node.left_child, idx[left_mask])
            if np.any(~left_mask):
                rec(node.right_child, idx[~left_mask])
        
        rec(tree, np.arange(n, dtype=np.int64))
        return out

    def iter_predictions(self, X):
        total = self.count_trees()
        for i in range(total):
            yield self.get_predictions(i, X)

    def get_all_predictions(self, X, stack = False):
        preds = [self.get_predictions(i, X) for i in range(self.count_trees())]
        return np.stack(preds, axis=0) if stack else preds


def structurally_equal(a, b):
    if type(a) != type(b):
        return False

    if hasattr(a, "prediction") and hasattr(b, "prediction"):
        return a.prediction == b.prediction

    if hasattr(a, "feature") and hasattr(b, "feature"):
        if a.feature != b.feature:
            return False
        if not structurally_equal(a.left_child, b.left_child):
            return False
        if not structurally_equal(a.right_child, b.right_child):
            return False
        return True

    return False

def tree_structure_signature(tree): # nested tuples can be hashed
    if hasattr(tree, "prediction"): 
        return ('leaf', tree.prediction)
    elif hasattr(tree, "feature"): 
        return (
            'node',
            tree.feature,
            tree_structure_signature(tree.left_child),
            tree_structure_signature(tree.right_child)
        )
    else:
        raise ValueError("Unknown tree type: %s" % type(tree))

    