import os
import math
import numpy as np
import pickle
import json
import time
import warnings
import pandas as pd
from treefarms_rid_utils import construct_tree_rset
from lickety_vi_utils import get_model_reliances
from sklearn.utils import resample
from licketyresplit import LicketyRESPLIT

#from gosdt.model.threshold_guess import compute_thresholds
from split import ThresholdGuessBinarizer
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

def compute_thresholds(X_all, y, n_est=40, max_depth=1):
    """
    Drop-in replacement for gosdt.model.threshold_guess.compute_thresholds
    implemented with split.ThresholdGuessBinarizer.

    Returns:
      X_binned_full (pd.DataFrame of 0/1 ints),
      thresholds (unused -> None),
      header (list of column names like 'feat<=thr'),
      threshold_guess_time (float seconds)
    """
    t0 = time.time()
    enc = ThresholdGuessBinarizer(n_estimators=n_est, max_depth=max_depth, random_state=42)
    enc.set_output(transform="pandas")
    X_binned = enc.fit_transform(X_all, y)
    X_binned = (X_binned != 0).astype(int) # 0/1 not bool
    header = [str(c) for c in X_binned.columns]  # e.g., 'age<=37.5' (double check that the columns are that)
    return X_binned, None, header, time.time() - t0

class RashomonImportanceDistribution:
    '''
    A class to compute and interface with the
    Rashomon Importance Distribution. Note that this
    implementation makes heavy use of caching.

    Attributes
    ----------
    input_df : pd.DataFrame
        A pandas DataFrame containing a binarized
        version of the dataset we seek to explain
    binning_map : dict
        A dictionary of the form
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8]}
        describing which variables in the unbinned version of
        input_df map to which columns of the binarized version
    db : int
        The depth bound to use when computing rashomon sets
    lam : float
        The regularization weight to use when computing
        rashomon sets
    eps : float
        The threshold to use when computing rashomon sets
        (i.e., models within eps of optimal are included)
    dataset_name : string
        The name of the datset being analyzed. Used to determine
        where to cache various files
    n_resamples : int
        The number of bootstrap samples to compute
    cache_dir_root : string
        The root file path at which all cached files should be stored
    rashomon_output_dir : string
        The name of the subfolder of cache_dir_root in which rashomon
        sets will be stored
    verbose : bool
        Whether to produce extra logging
    vi_metric : string
        The VI metric to use for this RID; should be one of 
        ['sub_mr', 'div_mr', 'sub_cmr', 'div_cmr', 'shap']
    max_par_for_gosdt : int
        The maximum number of instances of GOSDT to run
        in parallell; reduce this number if memory issues
        occur
    allow_binarize_internally : bool
        Whether to allow RID to binarize data internally
        using threshold guessing
    '''
    def __init__(self, 
            input_df,
            db, lam, eps, 
            binning_map=None,
            dataset_name='dataset_1', 
            n_resamples=100,
            cache_dir_root='./cached_files',
            rashomon_output_dir='rashomon_outputs',
            verbose=False,
            vi_metric='sub_mr',
            max_par_for_gosdt=5,
            allow_binarize_internally=False, lickety_lookahead = 1):

        supported_vis = ['sub_mr', 'div_mr', 'sub_cmr', 'div_cmr', 'shap']
        assert vi_metric in supported_vis, \
            f"Error: VI metric {vi_metric} not recognized. Supported VI metrics are {supported_vis}"

        if input_df.isin([0, 1]).all().all():
            assert binning_map, "Error: Binning map must not be None if binary data is given."
            self.input_df = input_df
            self.binning_map = binning_map
        elif allow_binarize_internally:
            warnings.warn("Non-binarized data detected, binarizing internally using guesses.")
            binarized_df, binning_map = self._binarize_data_guesses(input_df)
            self.input_df = binarized_df
            self.binning_map = binning_map
        else:
            raise Exception(
                """
                Non-binarized data was given, but allow_binarize_internally is set to False. 
                If you would like RID to binarize data internally, you can specify allow_binarize_internally=True.
                Otherwise, binarize your data before passing it to RID.
                """
            )
        self.input_df = self.input_df.astype(int)

        self.vi_metric = vi_metric
        self.n_vars = len(binning_map)
        self.n_resamples = n_resamples
        self.db = db
        self.lam = lam
        self.eps = eps
        self.dataset_name = dataset_name
        self.rashomon_output_dir = rashomon_output_dir
        self.verbose = verbose
        self.max_par_for_gosdt = max_par_for_gosdt
        self.lickety_lookahead = lickety_lookahead

        try:
            self.num_cpus = os.cpu_count()
        except:
            self.num_cpus = 1

        # Create the cache directory if necessary
        if not os.path.exists(os.path.join(cache_dir_root, dataset_name)):
            os.makedirs(os.path.join(cache_dir_root, dataset_name))
        self.cache_dir = os.path.join(cache_dir_root, dataset_name)

        if not os.path.exists(os.path.join(self.cache_dir, self.rashomon_output_dir)):
            os.makedirs(os.path.join(self.cache_dir, self.rashomon_output_dir))

        # First, compute each necessary bootstrapped dataset --------------
        for i in range(self.n_resamples):
            self._construct_bootstrap_datasets(i)

        # Second, compute each necessary Rashomon set ---------------------
        with Pool(min(self.num_cpus, self.max_par_for_gosdt)) as p:
            p.map(RashomonImportanceDistribution._construct_rashomon_sets, 
                [self]*self.n_resamples,
                [i for i in range(self.n_resamples)])

        # Third, compute the variable importance for each ----------------
        # model in each bootstrap Rashomon set ---------------------------
        self._compute_and_aggregate_vis()

        self.vi_dataframe = self._read_vis_to_construct_rid(
            file_paths=[os.path.join(self.cache_dir, f'lickety_{self.vi_metric}s_bootstrap_{i}_eps_{eps}_db_{db}_reg_{lam}_lh_{self.lickety_lookahead}.pickle') for i in range(n_resamples)],
            n_vars=self.n_vars
        )
        self.rid_with_counts = self._get_df_with_counts()

    def _construct_bootstrap_datasets(self, bootstrap_ind):
        '''
        Constructs and stores the bootstrapped dataset for the given index

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        if not os.path.isfile(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv')):
            for _ in range(bootstrap_ind):
                resampled_df = resample(self.input_df)
            resampled_df = resample(self.input_df)
            
            resampled_df.to_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'), index=False)

    def _construct_rashomon_sets(self, bootstrap_ind):
        """
        Constructs and stores the Rashomon set for the given bootstrap index,
        using LicketyRESPLIT instead of construct_tree_rset. Saves a pickle
        (not JSON) containing the trie and minimal metadata.
        """
        rset_path = os.path.join(
            self.cache_dir,
            self.rashomon_output_dir,
            f'lickety_trie_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pkl'
        )

        if os.path.isfile(rset_path):
            if self.verbose:
                print(f"[RID] Rashomon set already exists: {rset_path}")
            return

        if self.verbose:
            print(f"[RID] Generating Rashomon set with LicketyRESPLIT for bootstrap {bootstrap_ind}")

        df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        config = {
            "regularization": float(self.lam),
            "depth_budget":   int(self.db-1), # depth-0 is root for licketyresplit
            "rashomon_bound_multiplier": float(self.eps),
        }
        model = LicketyRESPLIT(
            config=config,
            binarize=False, # we’re already binarized upstream
            lookahead=self.lickety_lookahead,
            multipass=True
        )
        model.fit(X, y)  # builds model.trie (TreeTrieNode)

        trie_trunc = model.trie.truncated_copy(max_depth=int(self.db - 1), budget=int(round(model.trie.min_objective * (1.0 + self.eps))))

        # some of this stuff may not be needed
        artifact = {
            "trie": trie_trunc,
            "feature_names": list(X.columns),
            "config": config,
            "lamN": getattr(model, "lamN", None),
            "n": len(y),
        }

        with open(rset_path, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print(f"[RID] Saved LicketyRESPLIT trie to {rset_path}")


    def _binarize_data_guesses(self, df_unbinned):
        '''
        Converts a non-binarized dataset to a binarized version
        that is compliant with RID

        Parameters
        ----------
            df_unbinned : pd.DataFrame
                The original, non-binarized dataset provided
        '''
        n_est = 40
        max_depth = 1
        X_all = df_unbinned.iloc[:, :-1]
        y = df_unbinned.iloc[:, -1]
        X_binned_full, thresholds, header, threshold_guess_time = compute_thresholds(X_all.copy(), y.copy(), n_est, max_depth)
        df = pd.concat((X_binned_full, y), axis=1)

        col_map = {str(c).strip(): i for i, c in enumerate(df_unbinned.columns)}

        bins = {b: [] for b in range(len(df_unbinned.columns) - 1)}

        counter = 0
        for h in header:
            hs = str(h).strip()
            base = hs.split('<=', 1)[0].strip()
            cur_var = col_map[base]
            bins[cur_var].append(counter)
            counter += 1    

        bin_map = bins

        return df, bin_map

    def _compute_and_aggregate_vis(self):
        '''
        Computes and stores the variable importance metric
        for this RID for each model in each Rashomon set
        '''
        if 'cmr' in self.vi_metric:
             self._compute_and_aggregate_cmrs()
        elif 'mr' in self.vi_metric:
             self._compute_and_aggregate_mrs()
        elif 'shap' in self.vi_metric:
             self._compute_and_aggregate_shaps()
    
    def _compute_and_aggregate_cmrs(self):
        '''
        Constructs and stores the CMR for all models. 
        Not currently implemented.
        '''
        assert False, "Error: CMR not yet implemented"

    def _compute_and_aggregate_mrs(self):
        '''
        Computes and stores model reliance (sub and div) for
        each model in all bootstrapped Rashomon sets
        '''
        with Pool(self.num_cpus) as p:
            results_list = p.map(RashomonImportanceDistribution._get_mrs_for_dataset, 
                [self]*self.n_resamples,
                [i for i in range(self.n_resamples)])

        target_div_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        target_sub_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        
        start = time.time()
        for bootstrap_ind, val in enumerate(results_list):
            # A bit hacky, but this allows us to skip datasets for which
            # we've already found our VIs
            if val is None:
                continue

            cur_model_reliance = val[0]
            for var in range(self.n_vars):
                for mr in cur_model_reliance[var].keys():
                    if mr == 'mean':
                        target_div_model_reliances[var]['means'].append(cur_model_reliance[mr])
                    elif mr in target_div_model_reliances[var].keys():
                        target_div_model_reliances[var][mr] += cur_model_reliance[var][mr]
                    else:
                        target_div_model_reliances[var][mr] = cur_model_reliance[var][mr]

            with open(os.path.join(self.cache_dir, f'lickety_div_mrs_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pickle'), 'wb') as f:
                pickle.dump(target_div_model_reliances, f, protocol=pickle.HIGHEST_PROTOCOL)

            cur_model_reliance = val[1]
            for var in range(self.n_vars):
                for mr in cur_model_reliance[var].keys():
                    if mr == 'mean':
                        target_sub_model_reliances[var]['means'].append(cur_model_reliance[mr])
                    elif mr in target_sub_model_reliances[var].keys():
                        target_sub_model_reliances[var][mr] += cur_model_reliance[var][mr]
                    else:
                        target_sub_model_reliances[var][mr] = cur_model_reliance[var][mr]

            with open(os.path.join(self.cache_dir, f'lickety_sub_mrs_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pickle'), 'wb') as f:
                pickle.dump(target_sub_model_reliances, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        if self.verbose:
            print("Completed final processing in {} seconds".format(time.time() - start))
    
    def _compute_and_aggregate_shaps(self):
        '''
        Computes and stores SHAP values for
        each model in all bootstrapped Rashomon sets
        Not currently implemented.
        '''
        assert False, "Error: SHAP not yet implemented"
        
    def _get_mrs_for_dataset(self, bootstrap_ind):
        '''
        Computes and stores the model reliance (sub and div) for
        each model in the Rashomon set corresponding to the given index.

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        if os.path.isfile(os.path.join(self.cache_dir, f'lickety_div_mrs_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pickle'))\
            and os.path.isfile(os.path.join(self.cache_dir, f'lickety_sub_mrs_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pickle')):
            return None

        div_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        sub_model_reliances = [{'means':[]} for i in range(self.n_vars)]

        resampled_df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'))
        trie_path = os.path.join(self.cache_dir, 
                        self.rashomon_output_dir, 
                        f'lickety_trie_bootstrap_{bootstrap_ind}_eps_{self.eps}_db_{self.db}_reg_{self.lam}_lh_{self.lickety_lookahead}.pkl')
                        
        with open(trie_path, 'rb') as f:
            artifact = pickle.load(f)
            running_trie = artifact['trie'] 

            if self.verbose:
                cur_iterator = tqdm(self.binning_map)
            else:
                cur_iterator = self.binning_map
            for var in cur_iterator:
                tmp_div_model_reliances, tmp_sub_model_reliances, num_models = get_model_reliances(running_trie, resampled_df, 
                    var_of_interest=self.binning_map[var])

                div_model_reliances[var] = tmp_div_model_reliances
                sub_model_reliances[var] = tmp_sub_model_reliances
            
        return (div_model_reliances, sub_model_reliances)

    def _read_vis_to_construct_rid(self, file_paths, n_vars):
        '''
        Reads through each set of variable importance values specified by
        file_paths, combining them into the RID for this dataset

        Parameters
        ----------
            file_paths : list(string)
                A list of file paths pointing to each stored set of variable
                importance values
            n_vars : int
                The number of variables in the original version of the
                relevant dataset
        '''
        n_bootstraps = 0
        combined_mrs = [{'means':[]} for i in range(n_vars)]
        skips = []
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                
                    model_reliances = pickle.load(f)
                    # Add the information from this R-set to one mega trie
                    # For each variable
                    for var in range(n_vars):
                        # For each value in ['mean', observed_mr_1, observed_mr_2, ...]
                        for key in model_reliances[var].keys():
                            if key == 'means':
                                if 'cmr' in file_path:
                                    combined_mrs[var]['means'].append(np.mean(model_reliances[var]['means']))
                                else:
                                    combined_mrs[var]['means'] = combined_mrs[var]['means'] + np.mean(model_reliances[var]['means'])
                                continue
                            # If we've already seen this MR value, add the probability in it's R-set
                            # to our running list
                            elif key in combined_mrs[var].keys():
                                if 'cmr' in file_path:
                                    combined_mrs[var][key] = combined_mrs[var][key] + model_reliances[var][key]
                                else:
                                    combined_mrs[var][key].append(model_reliances[var][key])
                            # Otherwise, start a new running list for it
                            else:
                                if 'cmr' in file_path:
                                    combined_mrs[var][key] = model_reliances[var][key]
                                else:
                                    combined_mrs[var][key] = [model_reliances[var][key]]
                # Track how many successful rashomon sets we loaded
                n_bootstraps += 1
            except:
                skips.append(file_path)
                continue

        '''
        combined_mrs is now a dict of the form
        {
            var_1: {
                "means": [mean_1, mean_2, ...],
                observe_mr_1: [p_1, p_2, ...],
                ...
            }
            var_2: {
                "means": [mean_1, mean_2, ...],
                observe_mr_1: [p_1, p_2, ...],
                ...
            }
        }
        '''

        # The overall probability of observing each MR is then the mean over all
        # observed datasets
        model_reliance_df = pd.DataFrame()

        vars = []
        values = []
        probabilities = []
        # For each variable
        for var in range(n_vars):
            # For each observed MR
            for key in combined_mrs[var].keys():
                if key == 'means':
                    continue
                vars.append(var)
                values.append(key)
                probabilities.append(combined_mrs[var][key])

        # We now have vars, a list, values, a list, and probabilities,
        # a jagged list of lists
        model_reliance_df['var'] = vars
        model_reliance_df['val'] = values

        # Take the mean across rashomon sets for each probability
        true_probabilities = []
        for p in probabilities:
            true_probabilities.append(sum(p) / n_bootstraps)
        model_reliance_df['prob'] = true_probabilities

        model_reliance_df['count'] = 0
        for var in range(n_vars):
            cur_prob = model_reliance_df[model_reliance_df['var'] == var]['prob']
            model_reliance_df.loc[model_reliance_df['var'] == var, 'count'] = (cur_prob / cur_prob.min()).round().astype(int)
            
        return model_reliance_df

    def _get_df_with_counts(self):
        '''
        Gathers all computed sets of variable importances to compute
        a easily interfaced with datafram
        '''
        rid_with_counts = {}
        if self.verbose:
            print("Processing ours with counts")
        for var in range(self.n_vars):
            if self.verbose:
                print(f"Starting var {var}")
            rid_with_counts[var] = self.vi_dataframe[self.vi_dataframe['var'] == var]['val'].values
            rid_with_counts[var] = np.repeat(self.vi_dataframe[self.vi_dataframe['var'] == var]['val'].values,
                                                        self.vi_dataframe[self.vi_dataframe['var'] == var]['count'].values)
        return rid_with_counts


    def eval_cdf(self, k, var):
        '''
        Computes the value of the CDF for var at k

        Parameters
        ----------
            k : float
                The point in the CDF to evaluate
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        return (self.rid_with_counts[var] <= k).mean()

    def mean(self, var):
        '''
        Computes the mean variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        return self.rid_with_counts[var].mean()

    def median(self, var):
        '''
        Computes the median variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        return np.median(self.rid_with_counts[var])

    def bwr(self, var):
        '''
        Computes the box and whiskers range of variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        lower_q = np.quantile(self.rid_with_counts[var], 0.25)
        upper_q = np.quantile(self.rid_with_counts[var], 0.75)
        iqr = upper_q - lower_q
        return (lower_q - 1.5*iqr, upper_q + 1.5*iqr)