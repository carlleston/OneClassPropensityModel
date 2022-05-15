import spectral_score
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import kurtosis


def spectral_ranking(df):
    spectral_scores = spectral_score.spec(df.values) ## quanto maior o score melhor,default adota second feature ranking function (if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is)
    features_name = list(df.columns)
    df_spectral_feature = pd.DataFrame(spectral_scores,features_name).reset_index(drop = False)
    df_spectral_feature.columns = ['Features','spectral_scores']
    df_spectral_feature = df_spectral_feature.sort_values('spectral_scores', ascending = False)
    df_spectral_feature['spectral_ranking'] = df_spectral_feature['spectral_scores'].rank(ascending= False, method = 'min').astype(int)
    return df_spectral_feature

#Using Pearson Correlation
def pearson_ranking(df_train): 
    """df_train: features"""
    cor = df_train.corr()
    pearson_ranking = pd.DataFrame(abs(cor).sum() - 1).reset_index(drop=False)
    pearson_ranking.columns = ['Features','Pearson_sum']
    pearson_ranking = pearson_ranking.sort_values('Pearson_sum', ascending = True)
    df_rank = pearson_ranking[pearson_ranking['Pearson_sum'] != -1].reset_index(drop = True)
    df_rank['pearson_ranking'] = df_rank.index+1
    pearson_ranking_df = pearson_ranking.merge(df_rank[['Features','pearson_ranking']],on = 'Features', how = 'left')
    pearson_ranking_df = pearson_ranking_df.fillna(df_rank['pearson_ranking'].max()+1).sort_values('pearson_ranking').reset_index(drop= True)
    pearson_ranking_df['pearson_ranking'] = pearson_ranking_df['pearson_ranking'].astype(int)
    return pearson_ranking_df


def anova_ranking(X,y):
    selector = SelectKBest(score_func=f_classif, k='all').fit(X,y)  ### utiliza teste Anova F-score que varia de 0 a um "grande numero", quanto maior o score, menores as chances de a variavel intervir na target pelo acaso
    x_new = selector.transform(X) 
    scores = selector.scores_
    features = list(X.columns)
    feature_score = list(scores)
    df_feature_score = pd.DataFrame([features,feature_score]).T
    df_feature_score.columns = ['Features','anova_score']
    df_feature_score = df_feature_score.fillna(0)
    df_feature_score = df_feature_score.sort_values('anova_score', ascending=False)
    df_feature_score['anova_ranking'] = df_feature_score['anova_score'].rank(ascending= False, method = 'min').astype(int)
    return df_feature_score

def featureImportance_ranking(X,y):
    # Feature Importance with Extra Trees Classifier
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, y)
    FI_df = pd.DataFrame(model.feature_importances_,list(X.columns), ).reset_index(drop=False)
    FI_df.columns = ['Features','Importance']
    FI_df = FI_df.sort_values('Importance',ascending = False)
    FI_df['FI_ranking'] = FI_df['Importance'].rank(ascending= False, method = 'min').astype(int)
    return FI_df

## Calculating kurtosis for each feature
def kurtosis_ranking(X, outlier_isbetter = True):
    df_kurtosis = pd.DataFrame(X.kurt(axis=0)).reset_index(drop= False)
    df_kurtosis.columns = ['Features','kurtosis']
    if  not outlier_isbetter:
        asc = True
    else: asc = False
    df_kurtosis['kurtosis_ranking'] = df_kurtosis['kurtosis'].rank(ascending= asc, method = 'min').astype(int)
    df_kurtosis = df_kurtosis.sort_values('kurtosis_ranking')
    return df_kurtosis

def FS_OC_borda(X,y,outlier_isbetter):
    spectral_rank = spectral_ranking(X)
    pearson_rank = pearson_ranking(X)
    anova_df = anova_ranking(X,y)
    FI_df = featureImportance_ranking(X,y)
    kurtosis_df = kurtosis_ranking(X, outlier_isbetter)

    df1 = spectral_rank[['Features','spectral_ranking']]
    df2 =anova_df[['Features','anova_ranking']]
    df3 =pearson_rank[['Features','pearson_ranking']]
    df4 =FI_df[['Features','FI_ranking']]
    df5 = kurtosis_df[['Features','kurtosis_ranking']]

    df_ranks = df1.merge(df2, how='left', on=['Features']).merge(df3, how='left', on=['Features']).merge(df4, how='left', on =['Features']).merge(df5, how='left', on =['Features'])
    df_ranks['Borda_agg'] =  df_ranks.drop('Features', axis=1).sum(axis=1)
    df_ranks['reordered_ranks'] = df_ranks['Borda_agg'].rank(ascending= True, method = 'min').astype(int)
    df_ranks = df_ranks.sort_values('reordered_ranks')
    return df_ranks