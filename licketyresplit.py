import builtins
import sys
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from split import GOSDTClassifier
#from split._tree import Node, Leaf
from Tree import Node, Leaf
from split import SPLIT, LicketySPLIT
from split import ThresholdGuessBinarizer, GOSDTClassifier
from Trie import TreeTrieNode, SplitNode
import bisect
import math

# TODO
# Add lookahead "consistency" flag. The idea is right now, you will be solving the same problems but with a different lookahead.
# If you start with depth 6 and lookahead 2, when you recurse in create_trie, you will call with depth 5 and lookahead 2, and so on.
# The idea is to make a function depth->lookahead_used that isn't neccessarily injective.
# D8 L2, D7 L1, D6 L2, D5 L1, D4 L2 and so on if lookahead was 2. So cycle through every possibly lookahead value so it is consistent, instead of solving the same lookahead optimization just offset.

def count_leaves(node):
    if isinstance(node, Leaf):
        return 1
    else:
        return count_leaves(node.left_child) + count_leaves(node.right_child)
    
def get_tree_depth(node):
    if isinstance(node, Leaf):
        return 0
    elif isinstance(node, Node):
        return 1 + max(get_tree_depth(node.left_child), get_tree_depth(node.right_child))

class LicketyRESPLIT:
    '''
    LicketyRESPLIT: polynomial-per-tree Rashomon set approximation
    config : dict
            A dictionary of hyperparameters and settings. Must include keys such as:
                - 'regularization': float
                - 'depth_budget': int
                - 'rashomon_bound_multiplier': float
                - 'best_objective': float (optional)
    binarize : bool
            If True, uses ThresholdGuessBinarizer for binarization.
            If False, requires binary input data.
    '''
    def __init__(self, config, binarize=False, lookahead=1, multipass=True, consistent_lookahead = False, prune_style = "H", gbdt_n_est=50, gbdt_max_depth=1, optimal=False, pruning=True, better_than_greedy=False, try_greedy_first=False, trie_cache_strategy="compact", multiplicative_slack=0, cache_greedy=True, cache_lickety=True, cache_key_mode="bitvector", cache_packbits=None):
        self.config = config
        self.domultipass = multipass
        self.lookahead = lookahead
        self.consistent_lookahead = consistent_lookahead # only for Z right now, could maybe expand to H
        self.lookahead_map = None
        self.prune_style = prune_style
        self.better_than_greedy = better_than_greedy
        self._n = None
        self.best = None
        self.obj_bound = None
        #self.X_full = None
        self.X_bool = None
        self.y_full = None
        self.greedy_cache = dict()
        self.trie_cache = dict()
        self.lickety_cache = dict()
        self.cache_greedy = cache_greedy
        self.cache_lickety = cache_lickety
        self.trie_cache_strategy = trie_cache_strategy
        self._pack_cache = {}
        self.binarize = binarize
        if self.binarize: # binarize using threshold guessing
            print("Starting binarizing")
            self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, max_depth=gbdt_max_depth, random_state=42)
            self.enc.set_output(transform="pandas")
            print("Finished binarization")
        self.optimal = optimal
        self.pruning = pruning
        self.try_greedy_first = try_greedy_first
        self.multiplicative_slack = multiplicative_slack
        self.cache_key_mode = cache_key_mode
        if not (cache_greedy or cache_lickety or trie_cache_strategy):
            self.cache_packbits = False
        elif cache_packbits is not None:
            self.cache_packbits = cache_packbits
        else:
            self.cache_packbits = True

    #@profile
    def fit(self, X, y):
        if self.binarize: # train binarizer
            X = self.enc.fit_transform(X, y)

        if isinstance(X, np.ndarray) and X.dtype == np.bool_ and X.flags.f_contiguous:
            self.X_bool = X  # no copy
        else:
            # single allocation; computes bool + makes Fortran layout
            self.X_bool = np.asfortranarray(np.asarray(X) != 0, dtype=np.bool_)

        # keep one copy (uint8) and view as bool without copying
        y_arr = np.asarray(y)
        if y_arr.dtype == np.uint8:
            self.y_full = y_arr  # no copy
        else:
            # 0/1 uint8
            self.y_full = (y_arr != 0).astype(np.uint8, copy=False)

        # bool view shares memory with uint8 since itemsize==1
        self.y_bool = self.y_full.view(np.bool_)

        # --- unpack config ---
        self._n = len(y)
        lam   = self.config['regularization']
        depth = self.config['depth_budget']
        mult  = self.config['rashomon_bound_multiplier']
        self.lamN = int(round(lam * self._n))
        root_bitvector = np.ones(self._n, dtype=bool)
        best = None
        t0 = time.time()
        if self.consistent_lookahead: self.lookahead_map = self._build_lookahead_map(depth)
        root_path = None if self.cache_key_mode == "bitvector" else frozenset()
        if self.better_than_greedy:
            greedy_root_obj = self.train_greedy(root_bitvector, depth, path_key=root_path)
            obj_bound = int(greedy_root_obj)      # we'll say better or equal to than greedy
            print(f"[better_than_greedy] greedy objective at root = {greedy_root_obj} ") 
        else:
            best = self.config.get('best_objective') # compute best_objective if not provided
            if best is None:
                best = self.lickety_split(root_bitvector, depth, max(self.lookahead,1), path_key=root_path)  # use the lickety_split method to get the best objective, caches things, and does so faster
                print(f"Best objective on root: {best:.6f} or {(best/self._n):.6f} and {time.time() - t0:.2f} seconds using lickety")
                # best should be an integer now
            else:
                if best > 1:
                    best = int(round(best))
                else:
                    best = int(round(best * self._n))
            # final Rashomon bound
            obj_bound = round(best * (1+mult)) # the objectives are round so we will also round here
            #obj_bound = math.ceil(best * (1 + mult)) # pretty arbituary, round may be more technically correct but expanding doesn't hurt - in practice things have just barely been outside that treefarms finds

        self.best = int(best) if best is not None else None
        self.obj_bound = int(obj_bound)
        
        self.trie = self.construct_trie(root_bitvector, depth, int(round(obj_bound * (1 + self.multiplicative_slack))), path_key=root_path)
        t1 = time.time()
        print(f"[LicketyRESPLIT] Finished in {t1 - t0:.3f} seconds with {self.trie.count_trees()} trees (truncated to {self.count_trees()})")
        return self
    
    def count_trees(self, max_obj=None, min_obj=None, inclusive=True):
        # works even with multiplicative_slack
        if max_obj is None: max_obj = int(round(self.trie.min_objective*(1+self.config['rashomon_bound_multiplier']))) # more authentic because best could have improved
        return self.trie.count_trees_within_objective(max_obj, min_obj=min_obj, inclusive=inclusive)

    def list_trees(self, max_obj=None, min_obj=None, inclusive=True):
        if max_obj is None: max_obj = int(round(self.trie.min_objective*(1+self.config['rashomon_bound_multiplier'])))
        return self.trie.list_trees_within_objective(max_obj=max_obj, min_obj=min_obj, inclusive=inclusive)

    def iter_trees(self, max_obj=None, min_obj=None, inclusive=True):
        if max_obj is None: max_obj = int(round(self.trie.min_objective*(1+self.config['rashomon_bound_multiplier'])))
        return self.trie.iter_trees_within_objective(max_obj=max_obj, min_obj=min_obj, inclusive=inclusive)

    def iter_predictions(self, X=None, max_obj=None, min_obj=None, inclusive=True):
        if X is None: X = self.X_bool
        for tree in self.iter_trees(max_obj=max_obj, min_obj=min_obj, inclusive=inclusive):
            yield TreeTrieNode._predict_tree(tree, X)
    
    def get_all_predictions(self, X=None, max_obj=None, min_obj=None, inclusive=True, stack=False):
        if X is None: X = self.X_bool
        preds = [p for p in self.iter_predictions(X, max_obj=max_obj, min_obj=min_obj, inclusive=inclusive)]
        return np.stack(preds, axis=0) if stack and preds else preds

    def count_unique_prediction_vectors(self, X=None, preds=None, max_obj=None, min_obj=None, inclusive=True):
        if preds is None:
            if X is None:
                if X is None: X = self.X_bool
            preds = self.get_all_predictions(X, max_obj=max_obj, min_obj=min_obj, inclusive=inclusive, stack=True)
        if isinstance(preds, list):
            preds = np.stack(preds, axis=0)
        preds = np.asarray(preds, dtype=np.uint8)
        return np.unique(preds, axis=0).shape[0]

    #@profile
    def _key_bytes(self, bv, where): # for memory-efficiency - if ive seen this exact byte string b before, give me the one I stored earlier.
        bv = np.ascontiguousarray(bv, dtype=np.bool_)
        b = np.packbits(bv, bitorder="little").tobytes()
        cache_on = (
            (where == "greedy"  and self.cache_greedy) or
            (where == "lickety" and self.cache_lickety) or
            (where == "trie"    and bool(self.trie_cache_strategy))
        )
        if self.cache_packbits and cache_on:
            return self._pack_cache.setdefault(b, b)
        return b

    def _greedy_cache_get(self, key):
        return self.greedy_cache.get(key) if self.cache_greedy else None


    def _greedy_cache_set(self, key, val):
        if self.cache_greedy:
            self.greedy_cache[key] = val

    def _lickety_cache_get(self, key):
        return self.lickety_cache.get(key) if self.cache_lickety else None

    def _lickety_cache_set(self, key, val):
        if self.cache_lickety:
            self.lickety_cache[key] = val

    def _encode_lit(self, feat, val_bit): # doubling the feature index and maybe adding 1 or not, so mapping (feature, on/off) to an integer
        return (int(feat) << 1) | (int(val_bit) & 1)

    def _path_add(self, path_key, feat, val_bit): # frozen-set is order invariant
        lit = self._encode_lit(feat, val_bit)
        if path_key is None:
            return frozenset((lit,))
        return frozenset((*path_key, lit))

    def _cache_key(self, bitvector, path_key, where):
        """
        - 'bitvector': canonical packed-bytes (interned)
        - 'literal'  : order-invariant frozenset of encoded literals
        """
        if self.cache_key_mode == "literal":
            return path_key if path_key is not None else frozenset()
        # default: bitvector
        return self._key_bytes(bitvector, where=where)

    #@profile
    def construct_trie(self, bitvector, depth, budget, path_key=None):
        base_key = self._cache_key(bitvector, path_key, where="trie")  # bytes when "bitvector", frozenset when "literal"

        if self.trie_cache_strategy:
            if self.trie_cache_strategy != "compact":
                key = (base_key, depth, budget)
                hit = self.trie_cache.get(key)
                if hit is not None:
                    return hit
            else:
                entry = self.trie_cache.get(base_key)
                if entry is not None:
                    (d_max, b_at_d, trieD), (d_at_b, b_max, trieB) = entry
                    # prefer whichever qualifies as a superset; either is fine
                    if d_max >= depth and b_at_d >= budget:
                        return trieD.truncated_copy(max_depth=depth, budget=budget)
                    if d_at_b >= depth and b_max >= budget:
                        return trieB.truncated_copy(max_depth=depth, budget=budget)

            # optional superset probing (works with either key mode)
            if self.trie_cache_strategy == "superset":
                max_depth_cfg = int(self.config.get('depth_budget', depth))
                max_budget_101 = int(math.ceil(budget * 1.01))
                # same depth, bigger budget
                if max_budget_101 > budget:
                    for b2 in range(budget + 1, max_budget_101 + 1):
                        k2 = (base_key, depth, b2)
                        hit = self.trie_cache.get(k2)
                        if hit is not None:
                            return hit.truncated_copy(max_depth=depth, budget=budget)
                # bigger depth, same budget
                if max_depth_cfg > depth:
                    for d2 in range(depth + 1, max_depth_cfg + 1):
                        k2 = (base_key, d2, budget)
                        hit = self.trie_cache.get(k2)
                        if hit is not None:
                            return hit.truncated_copy(max_depth=depth, budget=budget)
                # bigger depth AND bigger budget band
                if max_depth_cfg > depth and max_budget_101 > budget:
                    for d2 in range(depth + 1, max_depth_cfg + 1):
                        for b2 in range(budget + 1, max_budget_101 + 1):
                            k2 = (base_key, d2, b2)
                            hit = self.trie_cache.get(k2)
                            if hit is not None:
                                return hit.truncated_copy(max_depth=depth, budget=budget)

        trie = TreeTrieNode(budget=budget)
        N  = self._n
        
        y = self.y_full[bitvector]
        #rashomon = []

        guaranteed_expense = self.lamN # can add to this guaranteed misclassifications (conflicting labels)

        if budget < guaranteed_expense: # cannot do anything at all
            return trie # should never happen given that we prune
        
        # consider leaf predictions for both classes
        for pred in [0, 1]:
            #miscls = (y != pred).sum() / N # we are going to simplify things and use a misclassification cost relative to the global dataset size
            # we are going to scale all losss by N to eliminate floating point issues
            miscls = (y != pred).sum()
            cost   = self.lamN + miscls
            if cost <= budget:
                #rashomon.append(Leaf(prediction=pred, loss=cost))
                trie.add_leaf(Leaf(prediction=pred, loss=cost))

        if depth == 0 or budget < guaranteed_expense+self.lamN: # now 2lambda+optionalconflictinglabels. this now says a split is pointless.
            return trie
                     
        # try every feature split
        d = self.X_bool.shape[1]
        for feat in range(d):
            bf = self.X_bool[:, feat]
            # left_bitvector  = bitvector & bf
            # right_bitvector = bitvector & ~bf
            left_bitvector = np.logical_and(bitvector, bf)
            right_bitvector = np.logical_and(bitvector, np.logical_not(bf))
            if not left_bitvector.any() or not right_bitvector.any():
                continue
            if self.cache_key_mode == "literal":
                left_path_key  = self._path_add(path_key, feat, 1)
                right_path_key = self._path_add(path_key, feat, 0)
            else:
                left_path_key = right_path_key = None

            # only explore things where some oracle would get within the set. maybe this means we should add some slack to the initial call and then trim after
            # it may be worth it to try greedy, and if it it within the set, you don't need to ask a more expensive oracle.
            if self.try_greedy_first:
                loss_l  = self.train_greedy(left_bitvector,  depth - 1, path_key=left_path_key)
                loss_r = self.train_greedy(right_bitvector, depth - 1, path_key=right_path_key)
            if (not self.try_greedy_first) or loss_l + loss_r > budget: # if greedy is bad, we need to try the other methods, but otherwise don't.
                if self.optimal:
                    loss_l = self.objective_optimal(left_bitvector,  depth - 1)
                    loss_r = self.objective_optimal(right_bitvector, depth - 1)
                else:
                    if self.lookahead <= 0:
                        loss_l = self.train_greedy(left_bitvector,  depth - 1, path_key=left_path_key)
                        loss_r = self.train_greedy(right_bitvector, depth - 1, path_key=right_path_key)
                    else:
                        loss_l = self.lickety_split(left_bitvector,  depth - 1, k=self.lookahead, path_key=left_path_key)
                        loss_r = self.lickety_split(right_bitvector, depth - 1, k=self.lookahead, path_key=right_path_key)

            # pruning should always always be on
            if self.pruning and loss_l + loss_r > budget: # if greedy isn't within budget (which starts out epsilon loose), then we can skip this feature. greedy pruning to approxmation r-set
                 continue
            
            left_trie, right_trie = self.multipass(
                loss_l, loss_r,
                left_bitvector, right_bitvector,
                budget, depth,
                left_path_key=left_path_key, right_path_key=right_path_key
            )
        
            trie.add_split(feat, left_trie, right_trie)

        if self.trie_cache_strategy is not None:
            if self.trie_cache_strategy != "compact":
                self.trie_cache[(base_key, depth, budget)] = trie
            else:
                old = self.trie_cache.get(base_key)
                if old is None:
                    # initialize both slots with this trie; ok if identical
                    self.trie_cache[base_key] = ((depth, budget, trie), (depth, budget, trie))
                else:
                    (d_max, b_at_d, trieD), (d_at_b, b_max, trieB) = old
                    # update max-depth slot
                    if depth > d_max:
                        d_max, b_at_d, trieD = depth, budget, trie
                    # update max-budget slot
                    if budget > b_max:
                        d_at_b, b_max, trieB = depth, budget, trie
                    self.trie_cache[base_key] = ((d_max, b_at_d, trieD), (d_at_b, b_max, trieB))

        return trie

    #@profile
    def multipass(self, loss_l, loss_r, left_bitvector, right_bitvector, budget, depth, left_path_key=None, right_path_key=None):
        overly_tight_right_expense = loss_r # overly tight, will only find a subset of what we want
        left_subset_budget = budget - overly_tight_right_expense # too restrictive, but will find a subset of what we want (careful that empty doesn't mean anything, but this is somewhat just heuristics to see how to cut down search space)
      
        # enumerate left subtree set
        left_subtrie = self.construct_trie(left_bitvector, depth-1, left_subset_budget, path_key=left_path_key) # a subset of what we want 
    
        # compute max right budget
        min_left = left_subtrie.min_objective # the smaller the left objective, the more we can spend on the right.
        right_budget = budget - min_left
        
        # enumerate right subtree set
        right_trie = self.construct_trie(right_bitvector, depth-1, right_budget, path_key=right_path_key)

        min_right = right_trie.min_objective # assuming there was a tree, this should be the best possible tree that one can get
        left_budget = budget - min_right # nowe we have a not overly tight budget for the left side, so we can enumerate it correctly

        if left_budget > left_subset_budget: # TODO: CAN SHARE INFORMATION BETWEEN SUBTRIE AND TRIE, BUT THIS HARDLY EVER RUNS SO DOESN'T MATTER MUCH
            left_trie = self.construct_trie(left_bitvector, depth-1, left_budget, path_key=left_path_key) # a subset of what we want
        else: 
            left_trie = left_subtrie # budgets were equal, will just return the same thing
            #print("LEFT SUBSET = LEFT SET") # this happens a lot and saves time
        #left_set = left_subset # just to see if this is faster. indeed it is by a little bit
        
        return left_trie, right_trie

    def _build_lookahead_map(self, max_depth: int):
        # build a depth->k map using a cycling pattern from depth=max_depth down to 0 (goal: help in caching)
        K = int(self.lookahead)
        if K <= 1 or max_depth <= 0:
            return {d: 1 for d in range(max_depth + 1)}

        m = {}
        k = K
        for d in range(max_depth, -1, -1):
            m[d] = min(d,k)
            # cycle k downward, wrap back to K after hitting 1
            k = (k - 1) if k > 1 else K
        return m

    def _next_k(self, k):
        """Cycle K,K-1,...,1,K,... where K=self.lookahead."""
        if self.prune_style == "H": return k
        elif self.prune_style == "Z":
            K = self.lookahead
            if K <= 0:   # degenerate: always greedy. this is only used for licketysplit initial objective so we want to always keep k 1.
                return 1
            return (k - 1) if k > 1 else K
        raise ValueError("prune_style must be 'Z' or 'H'")

    #@profile
    def train_greedy(self, bitvector, depth_budget, path_key=None):
        '''
        Requires X_train to be binary
        '''
        N = self._n
        reg = self.lamN # scaled integer
        key_base = self._cache_key(bitvector, path_key, where="greedy") # bytes or frozenset
        key = (key_base, depth_budget)
        cached = self._greedy_cache_get(key)
        if cached is not None:
            return cached
        y_train = self.y_full[bitvector]
        n_sub = y_train.size # needed for a better way to check if the mean of the subproblem is above 0.5 or not

        #node = Node(feature = None, left_child = None, right_child = None)

        if n_sub == 0:
            # empty node contributes nothing
            #self._greedy_cache_set(key, 0) # not worth caching
            return 0
            
        # take majority label
        pos = int(y_train.sum())
        errors = min(pos, n_sub - pos)
        loss = reg + errors # integer

        if depth_budget <= 0: # I HAVE ADJUSTED THIS TO TAKE DEPTH 0 BEING ROOT CONVENTION
            self._greedy_cache_set(key, loss)
            return loss

        if loss <= 2 * reg: # errors==0 is a special case of this
            self._greedy_cache_set(key, loss)
            return loss


        best_feature = self.find_best_feature_to_split_on(bitvector)
        if best_feature is None:
            self._greedy_cache_set(key, loss)
            return loss 
        bf = self.X_bool[:, best_feature]
        # left_bitvector  = bitvector & bf
        # right_bitvector = bitvector & ~bf
        left_bitvector = np.logical_and(bitvector, bf)
        right_bitvector = np.logical_and(bitvector, np.logical_not(bf))
        if self.cache_key_mode == "literal":
            left_path  = self._path_add(path_key, best_feature, 1)
            right_path = self._path_add(path_key, best_feature, 0)
        else:
            left_path = right_path = None


        if left_bitvector.any() and right_bitvector.any():
            left_loss = self.train_greedy(left_bitvector, depth_budget-1, path_key=left_path) # no rescaling needed given global dataset size normalization
            right_loss = self.train_greedy(right_bitvector, depth_budget-1, path_key=right_path)
                
            if left_loss + right_loss < loss: # only split if it improves the loss
                loss = left_loss + right_loss
            
        self._greedy_cache_set(key, loss)
        return loss

    #@profile
    def lickety_split(self, bitvector, depth_budget, k=1, path_key=None):
        key_base = self._cache_key(bitvector, path_key, where="lickety")
        if self.consistent_lookahead and self.lookahead_map is not None:
            k = self.lookahead_map.get(depth_budget, k)
            key = (key_base, int(depth_budget))
        else:
            k = min(k, depth_budget)
            key = (key_base, depth_budget, k) if self.lookahead > 1 else (key_base, depth_budget)

        cached = self._lickety_cache_get(key)
        if cached is not None:
            return cached
            
        if depth_budget <= 0: # adopt leaf logic
            a = self.train_greedy(bitvector, 0, path_key=path_key)
            #self._lickety_cache_set(key, a) # just recompute, not expensive, especially because greedy will be cached
            return a

        y_subset = self.y_full[bitvector]
        n_sub = y_subset.size
        pos = int(y_subset.sum())
        errors = min(pos, n_sub - pos)
        leaf_loss = self.lamN + errors # absolute
        #leaf_node = Leaf(prediction=pred, loss=leaf_loss)

        if leaf_loss <= 2 * self.lamN:
            self._lickety_cache_set(key, leaf_loss)
            return leaf_loss

        
        best_loss = float('inf')
        best_feat = None
        best_lr = (None, None)

        d = self.X_bool.shape[1]
        for feat in range(d):
            bf = self.X_bool[:, feat]
            left = np.logical_and(bitvector, bf)
            right = np.logical_and(bitvector, np.logical_not(bf))
            if not left.any() or not right.any():
                continue
            
            if self.cache_key_mode == "literal":
                left_path  = self._path_add(path_key, feat, 1)
                right_path = self._path_add(path_key, feat, 0)
            else:
                left_path = right_path = None


            if k>1: # consider k=2. we want to choose the next split based on the best split that follows given greedy after.
                loss_l = self.lickety_split(left,  depth_budget - 1, k-1, path_key=left_path)
                loss_r = self.lickety_split(right, depth_budget - 1, k-1, path_key=right_path)
            else: # k=1 lookahead is standard licketysplit, which chooses the best split based on greedy
                loss_l = self.train_greedy(left,  depth_budget - 1, path_key=left_path)
                loss_r = self.train_greedy(right, depth_budget - 1, path_key=right_path)
            total = loss_l + loss_r

            if total < best_loss:
                best_loss = total
                best_feat = feat
                best_lr = (left, right)

        if best_feat is None or best_loss >= leaf_loss:
            # no useful split (i.e the X are constant for every split, so every split was meaningless) OR split doesn't beat the leaf 
            self._lickety_cache_set(key, leaf_loss)
            return leaf_loss

        if self.consistent_lookahead and self.lookahead_map is not None:
            child_k = self.lookahead  # ignored; next call uses map
        # now, if k=2, we want to recurse on exactly what we just did, with 1 lookahead remaining
        # if k=1, we just chose the best split based on greedy completitions,  so now we want to start the cycle over
        else: child_k =  self._next_k(k) # if self.lookahead=1, this is always just 1. if k=1 and self.lookahead>1, we will restart the cycle.
        # if lookahead=0, child_k becomes 1 because we want to keep doing standard lickety split (only used in root objective calculation initially).
        
   
        left_loss  = self.lickety_split(best_lr[0], depth_budget - 1, child_k, path_key=self._path_add(path_key, best_feat, 1))
        right_loss = self.lickety_split(best_lr[1], depth_budget - 1, child_k, path_key=self._path_add(path_key, best_feat, 0))
        lickety_loss = left_loss + right_loss

        #node = Node(feature=best_feat, left_child=left_node, right_child=right_node)
        #node.loss = lickety_loss

        self._lickety_cache_set(key, lickety_loss)
        return lickety_loss

    #@profile
    def find_best_feature_to_split_on(self, bitvector, criterion: str = "entropy"):
        bv = np.asarray(bitvector, dtype=bool)
        n_sub = int(bv.sum())
        if n_sub <= 1:
            return None

        X = self.X_bool
        y = self.y_bool
        left_counts = X[bv].sum(axis=0).astype(np.int64)
        pos_total   = int((bv & y).sum())
        pos_left    = X[bv & y].sum(axis=0).astype(np.int64)

        right_counts = n_sub - left_counts
        pos_right    = pos_total - pos_left

        valid = (left_counts > 0) & (right_counts > 0)
        if not np.any(valid):
            return None

        p_orig  = pos_total / n_sub
        p_left  = pos_left[valid]  / left_counts[valid]
        p_right = pos_right[valid] / right_counts[valid]

        if criterion == "gini":
            g = lambda p: 2.0 * p * (1.0 - p)
            impur_orig  = g(p_orig)
            impur_left  = g(p_left)
            impur_right = g(p_right)
        elif criterion == "entropy":
            eps = 1e-12
            def H(p):
                p = np.clip(p, eps, 1.0 - eps)
                return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
            impur_orig  = H(p_orig)
            impur_left  = H(p_left)
            impur_right = H(p_right)
        else:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        wl = left_counts[valid]  / n_sub
        wr = right_counts[valid] / n_sub
        gain_valid = impur_orig - (wl * impur_left + wr * impur_right)

        valid_idx = np.where(valid)[0]
        best_idx  = int(valid_idx[np.argmax(gain_valid)])
        return best_idx

    def objective_optimal(self, bitvector, depth_budget):
        if not np.any(bitvector):
            return 0

        X_sub = self.X_bool[bitvector].astype(np.uint8, copy=False)
        y_sub = self.y_full[bitvector].astype(np.uint8, copy=False)
        n_sub = int(y_sub.size)

        pos = int(y_sub.sum())
        errors = min(pos, n_sub - pos)
        leaf_obj = self.lamN + errors

        if depth_budget <= 0 or n_sub <= 2 or errors == 0:
            return leaf_obj


        lam_global = float(self.config['regularization'])  
        N_global = int(self._n)
        lamN_global = self.lamN

        lambda_sub = lam_global * (N_global / max(1, n_sub)) # scale up lambda to penalize smaller subproblems more instead of proportion of local
        lambda_sub = float(min(0.48, max(1e-9, lambda_sub)))

        # run GOSDT with lambda (NOT lamN), then rescale leaves by lamN
        clf = GOSDTClassifier(
            regularization=lambda_sub,
            time_limit=6000,
            depth_budget=int(depth_budget),
            verbose=True
        )
        clf.fit(X_sub, y_sub)

        model_loss = int(round(clf.result_.model_loss))  # errors on subproblem
        raw_model = clf.result_.model

        def _count_leaves(node):
            import json as _json
            if isinstance(node, str):
                node = _json.loads(node)
            if isinstance(node, list):
                return _count_leaves(node[0])
            if "true" not in node and "false" not in node:
                return 1
            return sum(_count_leaves(node[br]) for br in ("true", "false") if br in node)

        n_leaves = _count_leaves(raw_model)
        obj = model_loss + lamN_global * n_leaves
        return obj