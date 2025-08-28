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
    def __init__(self, config, binarize=False, lookahead=1, multipass=True, consistent_lookahead = False, prune_style = "H", gbdt_n_est=50, gbdt_max_depth=1, optimal=False, pruning=True):
        self.config = config
        self.domultipass = multipass
        self.lookahead = lookahead
        self.consistent_lookahead = consistent_lookahead # only for Z right now, could maybe expand to H
        self.lookahead_map = None
        self.prune_style = prune_style
        self.models = None
        self._n = None
        #self.X_full = None
        self.X_bool = None
        self.y_full = None
        self.greedy_cache = dict()
        self.trie_cache = dict()
        self.lickety_cache = dict()
        self._pack_cache = {}
        self.binarize = binarize
        if self.binarize: # binarize using threshold guessing
            print("Starting binarizing")
            self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, max_depth=gbdt_max_depth, random_state=42)
            self.enc.set_output(transform="pandas")
            print("Finished binarization")
        self.call_counter = 0
        self.leaf_counter = 0
        self.prune_counter = 0
        self.entire_greedy_time = 0.0
        self.greedy_cache_hits = 0
        self.enumerate_cache_hits = 0
        self.enumerate_depth_cache_hits = 0
        self.left_budget_equal_count = 0
        self.left_budget_tight_count = 0
        self.optimal = optimal
        self.pruning = pruning

    #@profile
    def fit(self, X, y):
        if self.binarize: # train binarizer
            X = self.enc.fit_transform(X, y)
        Xv = X.reset_index(drop=True).values if hasattr(X, 'values') else X
        yv = y.reset_index(drop=True).values if hasattr(y, 'values') else y
        self.y_bool = (yv != 0)
        self.y_full = self.y_bool.astype(np.uint8)
        
        if True: 
            self.X_bool = np.empty(Xv.shape, dtype=bool, order='F')
            np.not_equal(Xv, 0, out=self.X_bool)
        else:
            self.X_bool = np.empty(Xv.shape, dtype=bool, order='C')
            np.not_equal(Xv, 0, out=self.X_bool)
        t0 = time.time()

        # --- unpack config ---
        self._n = len(y)
        lam   = self.config['regularization']
        depth = self.config['depth_budget']
        mult  = self.config['rashomon_bound_multiplier']
        self.lamN = int(round(lam * self._n))
        root_bitvector = np.ones(self._n, dtype=bool)
        if self.consistent_lookahead: self.lookahead_map = self._build_lookahead_map(depth)
        # compute best_objective if not provided
        best = self.config.get('best_objective')
        if best is None:
            # technically N / len(y) is how much you rescale by, but we only run this at the root currently
            t0 = time.time()
            if self.prune_style == "Z":
                best = self.lickety_split(root_bitvector, depth, max(self.lookahead,1))  # use the lickety_split method to get the best objective, caches things, and does so faster
            elif self.prune_style == "H":
                best = self.lickety_tree_learner(root_bitvector, depth, max(self.lookahead-1, 0)) # if self.lookahead=0, that means greedy, but we want to do licketysplit objective initially, so we max it to 0, and 0 here actually means licketysplit. if self.lookahead=1, we want licketysplit, so we subtract 1 to get it instead of licketylicketysplit.
            else:
                raise ValueError("prune_style must be 'Z' or 'H'")
            print(f"Best objective on root: {best:.6f} or {(best/self._n):.6f} and {time.time() - t0:.2f} seconds using lickety")
            # best should be an integer now
        else:
            if best > 1:
                best = int(round(best))
            else:
                best = int(round(best * self._n))
        # final Rashomon bound
        #obj_bound = round(best * (1+mult)) # the objectives are round so we will also round here
        obj_bound = math.ceil(best * (1 + mult)) # pretty arbituary, round may be more technically correct but expanding doesn't hurt - in practice things have just barely been outside that treefarms finds

        #self.models = self.construct_trie(bitvector, depth, obj_bound)
        self.trie = self.construct_trie(root_bitvector, depth, obj_bound)
        t1 = time.time()
        print(f"[LicketyRESPLIT] Finished in {t1 - t0:.3f} seconds with {self.trie.count_trees()} trees")
        return self

    #@profile
    def _key_bytes(self, bv: np.ndarray):
        bv = np.ascontiguousarray(bv, dtype=np.bool_)
        b = np.packbits(bv, bitorder="little").tobytes()
        return self._pack_cache.setdefault(b, b)

    #@profile
    def construct_trie(self, bitvector, depth, budget):
        self.call_counter += 1
        key = (self._key_bytes(bitvector), depth, budget)
        if key in self.trie_cache:
            return self.trie_cache[key]
        # if you have a greater budget or a smaller depth than an existing solution to the subproblem, you could also return that (it is subset)
        
        trie = TreeTrieNode(budget=budget)
        N  = self._n
        
        y = self.y_full[bitvector]
        #rashomon = []

        guaranteed_expense = self.lamN # can add to this guaranteed misclassifications (conflicting labels)

        if budget < guaranteed_expense: # cannot do anything at all
            return trie
        
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
                     
        greedy_total_time = 0.0 
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
            # only explore things where greedy would get within the set
            t_feat = time.time()
            if self.optimal:
                loss_l = self.objective_optimal(left_bitvector,  depth - 1)
                loss_r = self.objective_optimal(right_bitvector, depth - 1)
            elif self.prune_style == "Z":
                if self.lookahead <= 0:
                    loss_l = self.train_greedy(left_bitvector,  depth - 1)
                    loss_r = self.train_greedy(right_bitvector, depth - 1)
                else:
                    loss_l = self.lickety_split(left_bitvector,  depth - 1, k=self.lookahead)
                    loss_r = self.lickety_split(right_bitvector, depth - 1, k=self.lookahead)
            elif self.prune_style == "H":
                if self.lookahead <= 0:
                    loss_l = self.train_greedy(left_bitvector,  depth - 1)
                    loss_r = self.train_greedy(right_bitvector, depth - 1)
                else:
                    # full H objective for each side (good for pruning; cache makes it cheap)
                    loss_l = self.lickety_tree_learner(left_bitvector,  depth - 1, self.lookahead-1) # subtracting 1 on lookahead because here lookahead of 0 is actually licketysplit because we defined oracle 0 to be greedy instead of entropy
                    loss_r = self.lickety_tree_learner(right_bitvector, depth - 1, self.lookahead-1)
            else:
                raise ValueError("prune_style must be 'Z' or 'H'")

            greedy_total_time += (time.time() - t_feat)
            # pruning should always always be on
            if self.pruning and loss_l + loss_r > budget: # if greedy isn't within budget (which starts out epsilon loose), then we can skip this feature. greedy pruning to approxmation r-set
                 continue
            
            if self.domultipass:
                result = self.multipass(loss_l, loss_r, left_bitvector, right_bitvector, budget, depth)
            else: 
                #result = self.singlepass(loss_l, loss_r, left_bitvector, right_bitvector, budget, depth)
                #result = self.conservative_singlepass(loss_l, loss_r, left_bitvector, right_bitvector, budget, depth)
                result = self.extremely_conservative_singlepass(loss_l, loss_r, left_bitvector, right_bitvector, budget, depth)

            left_trie, right_trie = result
            trie.add_split(feat, left_trie, right_trie)

           
        self.entire_greedy_time += greedy_total_time
        #print(f"[enumerate] call {self.call_counter} | greedy_total_time = {greedy_total_time:.3f} s | len(rashomon) = {len(rashomon)}")
        self.trie_cache[key] = trie
        return trie

    #@profile
    def multipass(self, loss_l, loss_r, left_bitvector, right_bitvector, budget, depth):
        overly_tight_right_expense = loss_r # overly tight, will only find a subset of what we want
        left_subset_budget = budget - overly_tight_right_expense # too restrictive, but will find a subset of what we want (careful that empty doesn't mean anything, but this is somewhat just heuristics to see how to cut down search space)
      
        # enumerate left subtree set
        left_subtrie = self.construct_trie(left_bitvector, depth-1, left_subset_budget) # a subset of what we want 
    
        # compute max right budget
        min_left = left_subtrie.min_objective # the smaller the left objective, the more we can spend on the right.
        right_budget = budget - min_left
        
        # enumerate right subtree set
        right_trie = self.construct_trie(right_bitvector, depth-1, right_budget)

        min_right = right_trie.min_objective # assuming there was a tree, this should be the best possible tree that one can get
        left_budget = budget - min_right # nowe we have a not overly tight budget for the left side, so we can enumerate it correctly

        if left_budget > left_subset_budget: # TODO: CAN SHARE INFORMATION BETWEEN SUBTRIE AND TRIE, BUT THIS HARDLY EVER RUNS SO DOESN'T MATTER MUCH
            left_trie = self.construct_trie(left_bitvector, depth-1, left_budget) # a subset of what we want
            self.left_budget_tight_count += 1
        else: 
            left_trie = left_subtrie # budgets were equal, will just return the same thing
            self.left_budget_equal_count += 1
            #print("LEFT SUBSET = LEFT SET") # this happens a lot and saves time
        #left_set = left_subset # just to see if this is faster. indeed it is by a little bit
        
        return left_trie, right_trie

    # test against multipass to ensure correctness
    def conservative_singlepass(self, loss_l, loss_r, left_bitvector, right_bitvector, budget, depth):
        min_other_side = self.lamN

        left_subset_budget = budget - min_other_side # very slow to decrease
        left_trie = self.construct_trie(left_bitvector, depth-1, left_subset_budget)

        right_budget = budget - left_trie.min_objective # exact and faster
        if right_budget < 0:
            right_budget = 0

        right_trie = self.construct_trie(right_bitvector, depth-1, right_budget)

        return left_trie, right_trie

    def extremely_conservative_singlepass(self, loss_l, loss_r, left_bitvector, right_bitvector, budget, depth):
        left_subset_budget = budget - self.lamN # no decrement
        left_trie = self.construct_trie(left_bitvector, depth-1, left_subset_budget)

        right_budget = budget - self.lamN # no decrement
        if right_budget < 0:
            right_budget = 0

        right_trie = self.construct_trie(right_bitvector, depth-1, right_budget)

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
        K = self.lookahead
        if K <= 0:   # degenerate: always greedy. this is only used for licketysplit initial objective so we want to always keep k 1.
            return 1
        return (k - 1) if k > 1 else K

    #@profile
    def train_greedy(self, bitvector, depth_budget):
        '''
        Requires X_train to be binary
        '''
        N = self._n
        reg = self.lamN # scaled integer
        key = (self._key_bytes(bitvector), depth_budget)
        if key in self.greedy_cache:
            self.greedy_cache_hits += 1 
            return self.greedy_cache[key]
        y_train = self.y_full[bitvector]
        n_sub = y_train.size # needed for a better way to check if the mean of the subproblem is above 0.5 or not

        #node = Node(feature = None, left_child = None, right_child = None)

        if n_sub == 0:
            # empty node contributes nothing
            self.greedy_cache[key] = 0
            return 0
            
        # take majority label
        pos = int(y_train.sum())
        errors = min(pos, n_sub - pos)
        loss = reg + errors # integer

        if depth_budget <= 0: # I HAVE ADJUSTED THIS TO TAKE DEPTH 0 BEING ROOT CONVENTION
            self.greedy_cache[key] = loss
            return loss

        if loss <= 2 * reg: # errors==0 is a special case of this
            self.greedy_cache[key] = loss
            return loss


        best_feature = self.find_best_feature_to_split_on(bitvector)
        if best_feature is None:
            self.greedy_cache[key] = loss
            return loss 
        bf = self.X_bool[:, best_feature]
        # left_bitvector  = bitvector & bf
        # right_bitvector = bitvector & ~bf
        left_bitvector = np.logical_and(bitvector, bf)
        right_bitvector = np.logical_and(bitvector, np.logical_not(bf))
            
        if left_bitvector.any() and right_bitvector.any():
            left_loss = self.train_greedy(left_bitvector, depth_budget-1) # no rescaling needed given global dataset size normalization
            right_loss = self.train_greedy(right_bitvector, depth_budget-1)
                
            if left_loss + right_loss < loss: # only split if it improves the loss
                loss = left_loss + right_loss
            
        self.greedy_cache[key] = loss
        return loss

    #@profile
    def lickety_split(self, bitvector, depth_budget, k=1):
        if self.consistent_lookahead and self.lookahead_map is not None:
            k = self.lookahead_map.get(depth_budget, k)
            # cache key is now independent of input k when consistent
            key = (self._key_bytes(bitvector), int(depth_budget))
        else:
            # original behavior (cap k by depth for caching)
            k = min(k, depth_budget)
            if self.lookahead > 1:
                key = (self._key_bytes(bitvector), depth_budget, k)
            else:
                key = (self._key_bytes(bitvector), depth_budget)
                
        if key in self.lickety_cache:
            return self.lickety_cache[key]
            
        if depth_budget <= 0: # adopt leaf logic
            a = self.train_greedy(bitvector, 0)
            self.lickety_cache[key] = a
            return a

        y_subset = self.y_full[bitvector]
        n_sub = y_subset.size
        pos = int(y_subset.sum())
        errors = min(pos, n_sub - pos)
        leaf_loss = self.lamN + errors # absolute
        #leaf_node = Leaf(prediction=pred, loss=leaf_loss)

        if leaf_loss <= 2 * self.lamN:
            self.lickety_cache[key] = leaf_loss
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


            if k>1: # consider k=2. we want to choose the next split based on the best split that follows given greedy after.
                loss_l = self.lickety_split(left,  depth_budget - 1, k-1)
                loss_r = self.lickety_split(right, depth_budget - 1, k-1)
            else: # k=1 lookahead is standard licketysplit, which chooses the best split based on greedy
                loss_l = self.train_greedy(left,  depth_budget - 1)
                loss_r = self.train_greedy(right, depth_budget - 1)
            total = loss_l + loss_r

            if total < best_loss:
                best_loss = total
                best_feat = feat
                best_lr = (left, right)

        if best_feat is None or best_loss >= leaf_loss:
            # no useful split (i.e the X are constant for every split, so every split was meaningless) OR split doesn't beat the leaf 
            self.lickety_cache[key] = leaf_loss
            return leaf_loss

        if self.consistent_lookahead and self.lookahead_map is not None:
            child_k = self.lookahead  # ignored; next call uses map
        # now, if k=2, we want to recurse on exactly what we just did, with 1 lookahead remaining
        # if k=1, we just chose the best split based on greedy completitions,  so now we want to start the cycle over
        else: child_k =  self._next_k(k) # if self.lookahead=1, this is always just 1. if k=1 and self.lookahead>1, we will restart the cycle.
        # if lookahead=0, child_k becomes 1 because we want to keep doing standard lickety split (only used in root objective calculation initially).
        
   
        left_loss  = self.lickety_split(best_lr[0],  depth_budget - 1, child_k)
        right_loss = self.lickety_split(best_lr[1], depth_budget - 1, child_k)
        lickety_loss = left_loss + right_loss

        #node = Node(feature=best_feat, left_child=left_node, right_child=right_node)
        #node.loss = lickety_loss

        self.lickety_cache[key] = lickety_loss
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

    def oracle_obj(self, bitvector, depth_budget, L): # the choice of oracle is based on L
        # L=0 is greedy. L=1 is licketysplit. L=2 is choosing the best split based on how LicketySPLIT would finish it, ...
        if depth_budget <= 0:
            return self.train_greedy(bitvector, 0)
        if L <= 0:
            return self.train_greedy(bitvector, depth_budget)

        return self.lickety_tree_learner(bitvector, depth_budget, L-1)

    def lickety_tree_learner(self, bitvector, depth_budget, L):
        # need to think more on if we can clamp L in the same way for caching purposes
        if depth_budget <= 0:
            return self.train_greedy(bitvector, 0)

        key = (self._key_bytes(bitvector), depth_budget, L)
        if key in self.lickety_cache:
            return self.lickety_cache[key]

        y_subset = self.y_full[bitvector]
        n_sub = y_subset.size
        pos = int(y_subset.sum())
        errors = min(pos, n_sub - pos)
        leaf_loss = self.lamN + errors

        if leaf_loss <= 2 * self.lamN:
            self.lickety_cache[key] = leaf_loss
            return leaf_loss

        best_feat = None
        best_loss_est = float("inf")
        best_lr = (None, None)

        D = self.X_bool.shape[1]
        for feat in range(D):
            bf = self.X_bool[:, feat]
            left  = np.logical_and(bitvector, bf)
            right = np.logical_and(bitvector, np.logical_not(bf))
            if not left.any() or not right.any():
                continue

            # score candidate split using the oracle (note oracle uses L-1)
            est_l = self.oracle_obj(left,  depth_budget - 1, L)
            est_r = self.oracle_obj(right, depth_budget - 1, L)
            total_est = est_l + est_r

            if total_est < best_loss_est:
                best_loss_est = total_est
                best_feat = feat
                best_lr = (left, right)

        if best_feat is None or best_loss_est >= leaf_loss:
            self.lickety_cache[key] = leaf_loss
            return leaf_loss

        # commit the chosen split and recurse on both sides with the SAME L (it just specifies the oracle)
        left_loss  = self.lickety_tree_learner(best_lr[0], depth_budget - 1, L)
        right_loss = self.lickety_tree_learner(best_lr[1], depth_budget - 1, L)
        obj = left_loss + right_loss

        self.lickety_cache[key] = obj
        return obj



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