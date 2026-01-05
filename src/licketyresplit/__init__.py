import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from ._core import LicketyRESPLIT as _LicketyCore

__all__ = ["LicketyRESPLIT"]

class LicketyRESPLIT:
    def __init__(self):
        # self._model = _core.LicketyRESPLIT()
        self._model = _LicketyCore()

    def fit(
        self,
        X,
        y,
        lambda_reg=0.01,
        depth_budget=5,
        rashomon_mult=0.01,
        multiplicative_slack=0.0,
        key_mode="hash",
        trie_cache_enabled=False,
        lookahead_k=1,
        oracle_style=0, 
        root_budget=None, # be weary - expected integerized already
        use_multipass=True, 
        rule_list_mode=False,
        majority_leaf_only=False,
        cache_cheap_subproblems=False,
        greedy_split_mode=1,
    ):
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=int)
        
        if root_budget is None:
            root_budget_int = -1
        else:
            root_budget_int = int(root_budget)
        self._model.fit(
            X,
            y,
            lambda_reg,
            depth_budget,
            rashomon_mult,
            multiplicative_slack,
            key_mode,
            trie_cache_enabled,
            lookahead_k,
            root_budget_int,
            bool(use_multipass), 
            bool(rule_list_mode), 
            int(oracle_style), 
            bool(majority_leaf_only),
            bool(cache_cheap_subproblems),
            int(greedy_split_mode),
        )

    def count_trees(self):
        return self._model.count_trees()

    def get_min_objective(self):
        return self._model.get_min_objective()

    def get_root_histogram(self):
        return self._model.get_root_histogram()
    
    def get_tree_objective(self, tree_index: int):

        obj, obj_norm = self._model.get_tree_objective(int(tree_index))
        return obj, obj_norm

    # WARNING: 1-indexed unlike features
    def get_tree_paths(self, tree_index: int):
        """
        returns (paths, predictions):
        - paths: list of lists of signed feature indices. these are 1-indexed but features are 0-indexed so must subtract 1.
          +f means "go left / True on feature f-1"
          -f means "go right / False on feature f-1".
        - predictions: list of 0/1 labels for each leaf.
        """
        return self._model.get_tree_paths(int(tree_index))
    
    def get_tree_paths_str(self, tree_index: int):
        """
        returns (paths_str, predictions) where:
        - paths_str is a list of strings like "[+0, -1, +2]"
        - indices are shifted by -1 so features are 0-indexed as one would expect
        """
        paths, preds = self.get_tree_paths(tree_index)

        out = []
        for p in paths:
            converted = []
            for v in p:
                if v >= 0:
                    converted.append(f"+{v - 1}")
                else:
                    converted.append(f"-{abs(v) - 1}")
            path_str = "[" + ", ".join(converted) + "]"
            out.append(path_str)

        return out, preds
    
    def get_predictions(self, tree_index: int, X):
        X = np.asarray(X, dtype=np.uint8)
        return self._model.get_predictions(int(tree_index), X)

    def get_all_predictions(self, X, stack: bool = False):
        X = np.asarray(X, dtype=np.uint8)
        return self._model.get_all_predictions(X, bool(stack))
    
    def plot_tree(self, tree_index: int, feature_names=None, figsize=(8, 6), ax=None):

        paths, preds = self.get_tree_paths(tree_index)

        # infer feature names
        if feature_names is None:
            # collect all encoded feature indices (1-based in paths)
            encodings = [abs(v) for path in paths for v in path]
            if encodings:
                max_f = max(encodings) - 1  # convert back to 0-based
            else:
                # single-leaf tree (no splits): we never use feature_names
                max_f = -1
            feature_names = [f"f{j}" for j in range(max_f + 1)] #f0 through fk-1 if k features


        # convert path representation into an explicit tree structure
        class Node:
            __slots__ = ("feature", "left", "right", "prediction")
            def __init__(self):
                self.feature = None   # integer feature index
                self.left = None # node
                self.right = None # node
                self.prediction = None  # only for leaves

        root = Node()

        # build tree
        for path, pred in zip(paths, preds):
            cur = root
            for signed_f in path:
                f = abs(signed_f) - 1 # get 0-based index
                go_left = signed_f > 0

                # if no split recorded yet, set it (not always true as every internal node is traversed through on paths to many leaves)
                if cur.feature is None:
                    cur.feature = f
                    cur.left = Node()
                    cur.right = Node()

                # move downward
                if go_left:
                    cur = cur.left
                else:
                    cur = cur.right

            # leaf node
            cur.prediction = pred

        def count_leaves(node):
            if node is None:
                return 0
            if node.prediction is not None or (node.left is None and node.right is None):
                return 1
            return count_leaves(node.left) + count_leaves(node.right)

        def collect_leaves_in_order(node, leaves):
            """Left-to-right list of leaf nodes."""
            if node is None:
                return
            if node.prediction is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def assign_positions_tree(root, positions):
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]

            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                # leaf
                if node.prediction is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                # internal: recurse on children first
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                x = 0.5 * (x_left + x_right)
                positions[node] = (x, -depth)

            dfs(root, 0)

        positions = {}
        assign_positions_tree(root, positions)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.set_axis_off()

        def draw_node(node):
            x, y = positions[node]

            if node.left:
                x2, y2 = positions[node.left]
                ax.add_line(Line2D([x, x2], [y, y2], color="black"))
                draw_node(node.left)
            if node.right:
                x2, y2 = positions[node.right]
                ax.add_line(Line2D([x, x2], [y, y2], color="black"))
                draw_node(node.right)

            if node.prediction is None:
                label = f"{feature_names[node.feature]}"
                color = "#ddeeff"
            else:
                label = f"{node.prediction}"
                color = "#e0ffd8"

            radius = 0.3
            circle = Circle((x, y), radius, facecolor=color, edgecolor="black")
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=10)

        draw_node(root)

        xs, ys = zip(*positions.values())
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        plt.title(f"LicketyRESPLIT Tree {tree_index}")
        plt.show()
        
    def get_tree_frontier_scores(self, tree_index: int, depth_budget: int):
        # returns a list of (depth_from_root, frontier_score) for each internal node of the specified tree. Root has depth 0.
        return self._model.get_tree_frontier_scores(int(tree_index), int(depth_budget))

    def root_lickety_objective_lookahead1(self, depth_budget: int):
        return int(self._model.root_lickety_objective_lookahead1(int(depth_budget)))

