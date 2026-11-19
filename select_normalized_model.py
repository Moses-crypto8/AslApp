"""Select estimator using only inner CV on the original TRAIN partition."""
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from database import load_database
from features import dataframe_to_xy
from trainer import split_data
from normalization import SequenceNormalizer
ROOT=Path(__file__).resolve().parent

def select(df):
    ids,_=split_data(df); X,y=dataframe_to_xy(df);X=X.iloc[ids];y=y.iloc[ids]
    candidates={
        'random_forest':RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced',n_jobs=-1),
        'extra_trees':ExtraTreesClassifier(n_estimators=300,random_state=42,class_weight='balanced',n_jobs=-1),
        'svm':Pipeline([('standardize',StandardScaler()),('svm',SVC(C=10,gamma='scale',probability=True,random_state=42,class_weight='balanced'))])}
    results={};pipes={}
    for name,estimator in candidates.items():
        pipe=Pipeline([('normalize',SequenceNormalizer()),('classifier',estimator)])
        scores=cross_val_score(pipe,X,y,cv=StratifiedKFold(3,shuffle=True,random_state=123),scoring='accuracy',n_jobs=1)
        results[name]={'fold_accuracy':scores.tolist(),'mean_accuracy':float(scores.mean())};pipes[name]=pipe
        print(name,results[name],flush=True)
    winner=max(results,key=lambda k:results[k]['mean_accuracy'])
    report={'method':'3-fold stratified CV on the 471 original training samples only; seed 123. Holdout excluded from selection.','candidates':results,'selected':winner}
    return pipes[winner],report

if __name__=='__main__':
    _,report=select(load_database(ROOT/'sign_language.csv'))
    (ROOT/'evaluation/model_selection.json').write_text(json.dumps(report,indent=2))
