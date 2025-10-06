import pandas as pd
import numpy

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/imodels/compas-recidivism/" + splits["train"])
# drop age since already in categories, bin priors into 3 categories
# may want to explore <= values and/or threshold guessing
df['priors<=0'] = df['priors_count'] <= 0
df['priors=1-2'] = df['priors_count'].apply(lambda x: x > 0 and x < 3)
df['priors=3+'] = df['priors_count'] >= 3
df.drop(['age', 'priors_count', 'days_b_screening_arrest', 'c_jail_time'], axis=1, inplace=True)
# encode 
df['juv_fel_count=0'] = df['juv_fel_count'] == 0
df['juv_other_count=0'] = df['juv_other_count'] == 0
df['juv_misd_count=0'] = df['juv_misd_count'] == 0
df.drop(['juv_fel_count', 'juv_other_count', 'juv_misd_count'], axis=1, inplace=True)
df.drop(['c_charge_degree:M'], axis=1, inplace=True) # just the negation of c_charge_degree:F
df.drop(['sex:Female'], axis=1, inplace=True) # this data measured sex as a binary, so sex:Male and sex:Female are negations of each other
df['target'] = df['is_recid'] # put target at the end
df.drop(['is_recid'], axis=1, inplace=True)

df.to_csv('compas_w_demographics.csv', index=False)