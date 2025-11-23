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

    def count_trees_with_objective(self, target_obj):
        return self.objectives.get(target_obj, 0)

    def count_trees_within_objective(self, max_obj, min_obj = None, inclusive = True):
        if min_obj is None:
            if inclusive:
                return sum(cnt for obj, cnt in self.objectives.items() if obj <= max_obj)
            else:
                return sum(cnt for obj, cnt in self.objectives.items() if obj < max_obj)
        else:
            if inclusive:
                return sum(cnt for obj, cnt in self.objectives.items() if min_obj <= obj <= max_obj)
            else:
                return sum(cnt for obj, cnt in self.objectives.items() if min_obj < obj < max_obj)


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

    def iter_trees_within_objective(self, max_obj, min_obj=None, inclusive = True):
        total = self.count_trees()
        for i in range(total):
            tree = self.get_ith_tree(i)
            obj = tree.loss
            if min_obj is None:
                # only upper bound
                if inclusive:
                    if obj <= max_obj:
                        yield tree
                    else:
                        return  # early stop (trees sorted by objective)
                else:
                    if obj < max_obj:
                        yield tree
                    else:
                        return  # early stop
            else:
                # lower & upper bounds
                lower_ok = (obj >= min_obj) if inclusive else (obj > min_obj)
                upper_ok = (obj <= max_obj) if inclusive else (obj < max_obj)

                if not upper_ok:
                    return  # exceeded upper bound => done (sorted)
                if lower_ok:
                    yield tree
                # else: obj is still below min bound; keep scanning until we enter the band

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

    def get_ith_tree_old(self, i):
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

    def get_ith_tree(self, i):
        total = self.count_trees()
        if i < 0 or i >= total:
            raise IndexError(f"Index {i} out of range (total={total})")

        cum = 0
        target_obj = None
        k_within = None
        for obj in sorted(self.objectives):  # ascending objective
            cnt = self.objectives[obj]
            if i < cum + cnt: # if i=5, cum=2, cnt=3, i want the 6th tree, i have seen 5 so far, so i go to the next objective and then get the 0th tree there.
                target_obj = obj
                k_within = i - cum
                break
            cum += cnt

        if target_obj is None:
            raise RuntimeError("Tree out of bounds")

        return self.get_kth_tree_with_objective(target_obj, k_within)


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

    def list_trees_within_objective(self, max_obj, min_obj = None, inclusive = True):
        def _in_band(val):
            if min_obj is None:
                return (val <= max_obj) if inclusive else (val < max_obj)
            if inclusive:
                return (min_obj <= val <= max_obj)
            else:
                return (min_obj < val < max_obj)

        result = []
        for child in self.children:
            if isinstance(child, Leaf):
                if _in_band(child.loss):
                    result.append(child)
            else:
                L, R = child.left, child.right
                lefts  = L.list_trees()
                rights = R.list_trees()
                for l_sub, r_sub in product(lefts, rights):
                    tot = l_sub.loss + r_sub.loss
                    if _in_band(tot):
                        node = Node(
                            feature=child.feature,
                            left_child=l_sub,
                            right_child=r_sub,
                            loss=tot
                        )
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

    def count_unique_prediction_vectors(self, X=None, preds=None):      
        if preds is None:
            if X is None:
                raise ValueError("Provide either `preds` or `X`.")
            preds = self.get_all_predictions(X, stack=True)

        if isinstance(preds, list):
            preds = np.stack(preds, axis=0)
        preds = np.asarray(preds, dtype=np.uint8)

        uniq = np.unique(preds, axis=0)
        return uniq.shape[0]

    def truncated_copy(self, max_depth, budget = None):
        # returns a deep-copied trie truncated to max_depth splits (0, leaves only) and recomputed under a smaller budget
        new_budget = self.budget if budget is None else int(budget)
        out = TreeTrieNode(budget=new_budget)

        # keep all leaves that fit the (possibly new) budget
        for child in self.children:
            if isinstance(child, Leaf):
                if child.loss <= new_budget:
                    out.add_leaf(Leaf(prediction=child.prediction, loss=child.loss))

        # if no more splits allowed, stop here
        if max_depth <= 0:
            return out

        # otherwise, recursively truncate SplitNode children
        for child in self.children:
            if isinstance(child, SplitNode):
                # recursively truncate both sides to (max_depth-1)
                right_budget = new_budget - child.left.min_objective
                if right_budget <= 0:
                    continue
                R_trunc = child.right.truncated_copy(max_depth - 1, right_budget)
                if R_trunc.is_empty():
                    continue

                left_budget = new_budget - R_trunc.min_objective
                if left_budget <= 0:
                    continue
                L_trunc = child.left.truncated_copy(max_depth - 1, left_budget)
                if L_trunc.is_empty():
                    continue

                out.add_split(child.feature, L_trunc, R_trunc)

        return out


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


class SpecializedDepth2Node:
    __slots__ = ("budget", "depth", "lamN", "n_pos", "n_neg",
                 "P1", "N1", "P2", "N2",
                 "objectives", "min_objective", "_active", '_obj_by_depth', '_min_by_depth')
    def __init__(self, budget, depth, lamN, n_pos, n_neg, P1, N1, P2=None, N2=None, top_k_idx=None):
        assert depth in (1, 2)
        
        self.budget = int(budget)
        self.depth = int(depth)
        self.lamN = int(lamN)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.P1 = np.asarray(P1, dtype=np.int32)
        self.N1 = np.asarray(N1, dtype=np.int32)
        self.P2 = None if P2 is None else np.asarray(P2, dtype=np.int32)
        self.N2 = None if N2 is None else np.asarray(N2, dtype=np.int32)
        d = self.P1.shape[0]
        self._active = np.arange(d, dtype=np.int32)

        self.objectives = Counter()
        self.min_objective = float("inf")

        self._obj_by_depth = {0: Counter(), 1: Counter(), 2: Counter()}
        self._min_by_depth = {0: float("inf"), 1: float("inf"), 2: float("inf")}

        self._enumerate_depth0()
        if self.depth >= 1:
            self._enumerate_depth1_fast()
        if self.depth >= 2:
            self._enumerate_depth2_fast()

        self.detach_stats()

    def detach_stats(self):
        self.P1 = None
        self.N1 = None
        self.P2 = None
        self.N2 = None

    def _bulk_add_losses(self, losses, level):
        if losses.size == 0:
            return
        losses = losses[losses <= self.budget]
        if losses.size == 0:
            return
        # overall histogram and per-depth update    
        vals, cnts = np.unique(losses.astype(np.int64, copy=False), return_counts=True)
        self.objectives.update(dict(zip(vals.tolist(), cnts.tolist())))
        m = int(vals.min())
        if m < self.min_objective:
            self.min_objective = m

        self._obj_by_depth[level].update(dict(zip(vals.tolist(), cnts.tolist())))
        if m < self._min_by_depth[level]:
            self._min_by_depth[level] = m

    def _enumerate_depth0(self):
        # two labelings: predict 0 or 1
        base = self.lamN
        # losses: lamN + errors
        losses = np.array([base + self.n_pos, base + self.n_neg], dtype=np.int64)
        self._bulk_add_losses(losses, level=0)

    def _enumerate_depth1_fast(self):
        lam = self.lamN
        d_idx = self._active
        posL = self.P1[d_idx]
        negL = self.N1[d_idx]
        posR = self.n_pos - posL
        negR = self.n_neg - negL

        # valid split: both branches non-empty
        valid = (posL + negL > 0) & (posR + negR > 0)
        if not np.any(valid):
            return

        posL = posL[valid]; negL = negL[valid]
        posR = posR[valid]; negR = negR[valid]

        # all 2x2 labelings in one go (4 combos)
        base = 2 * lam
        # broadcasting
        losses = np.stack([
            base + posL + posR,
            base + posL + negR,
            base + negL + posR,
            base + negL + negR
        ], axis=0).ravel()

        self._bulk_add_losses(losses, level=1)

    def _enumerate_depth2_fast(self):
        lam = self.lamN
        B = self.budget
        d_idx = self._active
        P1, N1, P2, N2 = self.P1, self.N1, self.P2, self.N2

        posL_all = P1[d_idx]
        negL_all = N1[d_idx]
        posR_all = self.n_pos - posL_all
        negR_all = self.n_neg - negL_all

        # only split left twice (3 leaves)
        base3 = 3 * lam
        for jf, f in enumerate(d_idx):
            posL = posL_all[jf]; negL = negL_all[jf]
            posR = posR_all[jf]; negR = negR_all[jf]

            pos11_row = P2[f, d_idx]
            neg11_row = N2[f, d_idx]
            pos10_row = posL - pos11_row
            neg10_row = negL - neg11_row

            valid_h = (pos11_row + neg11_row > 0) & (pos10_row + neg10_row > 0) & (d_idx != f)
            if not np.any(valid_h):
                continue

            pos11 = pos11_row[valid_h]; neg11 = neg11_row[valid_h]
            pos10 = pos10_row[valid_h]; neg10 = neg10_row[valid_h]

            # early prune whole block if min possible loss already too much by doing min over label choices: choose min error per leaf
            min_block = base3 + np.minimum(pos11, neg11) + np.minimum(pos10, neg10) + min(posR, negR)
            if np.all(min_block > B):
                continue

            # build the 2^3=8 labelings in vector form
            e0_choices = np.array([posR, negR], dtype=np.int32)
            left4 = np.stack([
                pos11 + pos10,
                pos11 + neg10,
                neg11 + pos10,
                neg11 + neg10
            ], axis=0)
            losses = (base3 + left4[:, None, :] + e0_choices[None, :, None]).reshape(-1)
            self._bulk_add_losses(losses, level=2)

        # split right twice (3 leaves)
        for jf, f in enumerate(d_idx):
            posL = posL_all[jf]; negL = negL_all[jf]
            posR = posR_all[jf]; negR = negR_all[jf]

            pos01_row = P1[d_idx] - P2[f, d_idx]
            neg01_row = N1[d_idx] - N2[f, d_idx]
            pos00_row = posR - pos01_row
            neg00_row = negR - neg01_row

            valid_g = (pos01_row + neg01_row > 0) & (pos00_row + neg00_row > 0) & (d_idx != f)
            if not np.any(valid_g):
                continue

            pos01 = pos01_row[valid_g]; neg01 = neg01_row[valid_g]
            pos00 = pos00_row[valid_g]; neg00 = neg00_row[valid_g]

            min_block = base3 + np.minimum(pos01, neg01) + np.minimum(pos00, neg00) + min(posL, negL)
            if np.all(min_block > B):
                continue

            e1_choices = np.array([posL, negL], dtype=np.int32)
            right4 = np.stack([
                pos01 + pos00,
                pos01 + neg00,
                neg01 + pos00,
                neg01 + neg00
            ], axis=0)
            losses = (base3 + right4[:, None, :] + e1_choices[None, :, None]).reshape(-1)
            self._bulk_add_losses(losses, level=2)

        # split both twice (4 leaves)
        base4 = 4 * lam
        for jf, f in enumerate(d_idx):
            posL = posL_all[jf]; negL = negL_all[jf]
            posR = posR_all[jf]; negR = negR_all[jf]

            pos11_row = P2[f, d_idx]
            neg11_row = N2[f, d_idx]
            pos10_row = posL - pos11_row
            neg10_row = negL - neg11_row
            valid_h = (pos11_row + neg11_row > 0) & (pos10_row + neg10_row > 0) & (d_idx != f)
            if not np.any(valid_h):
                continue
            pos11 = pos11_row[valid_h]; neg11 = neg11_row[valid_h]
            pos10 = pos10_row[valid_h]; neg10 = neg10_row[valid_h]
            h4 = np.stack([
                pos11 + pos10,
                pos11 + neg10,
                neg11 + pos10,
                neg11 + neg10
            ], axis=0)

            pos01_row = P1[d_idx] - P2[f, d_idx]
            neg01_row = N1[d_idx] - N2[f, d_idx]
            pos00_row = posR - pos01_row
            neg00_row = negR - neg01_row
            valid_g = (pos01_row + neg01_row > 0) & (pos00_row + neg00_row > 0) & (d_idx != f)
            if not np.any(valid_g):
                continue
            pos01 = pos01_row[valid_g]; neg01 = neg01_row[valid_g]
            pos00 = pos00_row[valid_g]; neg00 = neg00_row[valid_g]
            g4 = np.stack([
                pos01 + pos00,
                pos01 + neg00,
                neg01 + pos00,
                neg01 + neg00
            ], axis=0)

            # early optimal prune - min possible over (h,g)
            min_h = np.minimum.reduce(h4) 
            min_g = np.minimum.reduce(g4)
            if (base4 + min_h.min() + min_g.min()) > B:
                continue

            # combine all 2^4=16 combos
            for kh in range(4):
                Hk = h4[kh][:, None]  # (H,1)
                for kg in range(4):
                    Gk = g4[kg][None, :]  # (1,G)
                    sums = base4 + (Hk + Gk)  # (H,G)
                    self._bulk_add_losses(sums.ravel(), level=2)

    def truncated_copy(self, max_depth, budget = None):
        budget = self.budget if budget is None else int(budget)

        obj_by_depth_new = {0: Counter(), 1: Counter(), 2: Counter()}
        min_by_depth_new = {0: float("inf"), 1: float("inf"), 2: float("inf")}
        merged = Counter()
        global_min = float("inf")

        for lvl in range(max_depth + 1):
            src = self._obj_by_depth[lvl]
            if not src:
                continue
            # filter keys by budget
            kept = {k: v for k, v in src.items() if k <= budget}
            if kept:
                obj_by_depth_new[lvl].update(kept)
                lvl_min = min(kept.keys())
                min_by_depth_new[lvl] = lvl_min
                merged.update(kept)
                if lvl_min < global_min:
                    global_min = lvl_min

        # construct a new instance without re-running __init__ enumeration
        new = object.__new__(SpecializedDepth2Node)

        new.budget = budget
        new.depth = max_depth
        new.lamN = self.lamN
        new.n_pos = self.n_pos
        new.n_neg = self.n_neg
        new.P1 = self.P1
        new.N1 = self.N1
        new.P2 = self.P2
        new.N2 = self.N2
        new._active = self._active

        new.objectives = merged
        new.min_objective = global_min
        new._obj_by_depth = obj_by_depth_new
        new._min_by_depth = min_by_depth_new

        return new
    def is_empty(self):
        return not self.objectives

    def __bool__(self):
        return not self.is_empty()

    def count_trees(self):
        return sum(self.objectives.values())

    def count_trees_with_objective(self, target_obj):
        return self.objectives.get(target_obj, 0)

    def count_trees_within_objective(self, max_obj, min_obj=None, inclusive=True):
        if min_obj is None:
            if inclusive:
                return sum(cnt for obj, cnt in self.objectives.items() if obj <= max_obj)
            else:
                return sum(cnt for obj, cnt in self.objectives.items() if obj < max_obj)
        else:
            if inclusive:
                return sum(cnt for obj, cnt in self.objectives.items() if min_obj <= obj <= max_obj)
            else:
                return sum(cnt for obj, cnt in self.objectives.items() if min_obj < obj < max_obj)
