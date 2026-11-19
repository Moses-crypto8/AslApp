"""Evaluate unchanged holdout and synthetic uniform position/scale changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from database import load_database
from features import feature_columns
from trainer import split_data, load_model
from normalization import relocate_sequence, normalize_sequence

ROOT=Path(__file__).resolve().parent

def run():
    df=load_database(ROOT/'sign_language.csv');_,ids=split_data(df)
    raw=df.iloc[ids][feature_columns()].to_numpy(dtype=float).reshape(-1,15,84)
    truth=df.iloc[ids].label.to_numpy()
    models={name:load_model(ROOT/'models'/file)['model'] for name,file in [('normalized','asl_model.joblib'),('pixel_baseline','asl_pixel_baseline.joblib')]}
    cases=[('unchanged',1.,(0,0)),('shift_right_down',1.,(400,250)),('shift_left_up',1.,(-250,-150)),('half_size_shifted',.5,(500,300)),('larger_shifted',1.6,(-300,100)),('double_size',2.,(0,0))]
    result={'samples':len(ids),'evaluation':'Synthetic landmark transforms of the same legacy holdout; not camera images or unseen signers.', 'cases':[]}
    base={k:m.predict(pd.DataFrame(raw.reshape(-1,1260),columns=feature_columns())) for k,m in models.items()}
    for name,scale,offset in cases:
        transformed=np.stack([relocate_sequence(s,scale,offset) for s in raw])
        row={'name':name,'scale':scale,'offset':list(offset)}
        row['maximum_normalized_feature_change']=float(np.max(np.abs(np.stack([normalize_sequence(s) for s in raw])-np.stack([normalize_sequence(s) for s in transformed]))))
        for k,m in models.items():
            pred=m.predict(pd.DataFrame(transformed.reshape(-1,1260),columns=feature_columns()))
            row[k]={'correct':int(np.sum(pred==truth)),'accuracy':float(np.mean(pred==truth)),'prediction_consistency':float(np.mean(pred==base[k]))}
        result['cases'].append(row)
    path=ROOT/'evaluation/position_scale_report.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
