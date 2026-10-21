"""Ray Tune hyperparameter search for phishing URL detection."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.data_loader import load_d2
from src.models import MODEL_NAMES, get_model

OUT = Path("results/ray")

def train_trial(config, X_fit, X_valid, y_fit, y_valid):
    cfg = dict(config)
    key = cfg.pop("model")
    model = get_model(key, **cfg)
    model.fit(X_fit, y_fit)
    pred = model.predict(X_valid)
    return {"f1": float(f1_score(y_valid, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_valid, pred)),
            "precision": float(precision_score(y_valid, pred, zero_division=0)),
            "recall": float(recall_score(y_valid, pred, zero_division=0))}

def main():
    p = argparse.ArgumentParser(description="Ray Tune experiments for phishing URL detection")
    p.add_argument("--d2", default="data/raw/phiusiil_phishing_url.csv")
    p.add_argument("--sample-size", type=int, default=20000, help="0 uses all rows")
    p.add_argument("--num-samples", type=int, default=12, help="Total configurations across models")
    p.add_argument("--models", nargs="+", choices=["rf","xgb","lgbm"], default=["rf","xgb","lgbm"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpus-per-trial", type=float, default=2.0)
    a = p.parse_args()
    if a.num_samples < 1 or a.sample_size < 0: p.error("num-samples must be positive; sample-size non-negative")
    for folder in ("figures","metrics","reports"): (OUT/folder).mkdir(parents=True, exist_ok=True)
    X,y = load_d2(a.d2, sample=a.sample_size or None)
    Xtr,Xtest,ytr,ytest = train_test_split(X,y,test_size=.2,random_state=a.seed,stratify=y)
    Xfit,Xval,yfit,yval = train_test_split(Xtr,ytr,test_size=.2,random_state=a.seed,stratify=ytr)
    spaces = {
      "rf":{"n_estimators":tune.choice([100,180,250,350]),"max_depth":tune.choice([4,6,8,12,None]),"min_samples_leaf":tune.choice([4,8,12,20]),"min_samples_split":tune.choice([8,16,24,40]),"max_features":tune.choice(["sqrt",.5,.7])},
      "xgb":{"n_estimators":tune.choice([100,180,250,350]),"learning_rate":tune.loguniform(.015,.12),"max_depth":tune.randint(2,7),"min_child_weight":tune.choice([3,5,8,12]),"subsample":tune.uniform(.6,.95),"colsample_bytree":tune.uniform(.55,.9),"reg_alpha":tune.choice([.1,.5,1.,2.]),"reg_lambda":tune.choice([1.,3.,5.,10.]),"gamma":tune.choice([0.,.5,1.5,3.])},
      "lgbm":{"n_estimators":tune.choice([100,180,250,350]),"learning_rate":tune.loguniform(.015,.12),"max_depth":tune.randint(3,8),"num_leaves":tune.choice([7,12,16,24,31]),"min_child_samples":tune.choice([10,20,30,50]),"subsample":tune.uniform(.6,.95),"colsample_bytree":tune.uniform(.55,.9),"reg_alpha":tune.choice([.1,.5,1.,2.]),"reg_lambda":tune.choice([1.,3.,5.,10.])}}
    counts=[a.num_samples//len(a.models)]*len(a.models)
    for i in range(a.num_samples%len(a.models)): counts[i]+=1
    rows=[]
    for i,key in enumerate(a.models):
        if not counts[i]: continue
        fn=tune.with_parameters(train_trial,X_fit=Xfit,X_valid=Xval,y_fit=yfit,y_valid=yval)
        fn=tune.with_resources(fn,{"cpu":a.cpus_per_trial})
        tuner=tune.Tuner(fn,param_space={"model":key,**spaces[key]},
          tune_config=tune.TuneConfig(metric="f1",mode="max",num_samples=counts[i],search_alg=BasicVariantGenerator(random_state=a.seed+i)),
          run_config=tune.RunConfig(name="phishing_"+key,storage_path=str((OUT/"ray_storage").resolve()),failure_config=tune.FailureConfig(max_failures=1)))
        for r in tuner.fit():
            if r.error is None and r.metrics.get("f1") is not None:
                rows.append({"model":key,"validation_f1":float(r.metrics["f1"]),"validation_accuracy":float(r.metrics.get("accuracy",0)),"validation_precision":float(r.metrics.get("precision",0)),"validation_recall":float(r.metrics.get("recall",0)),"config":json.dumps(dict(r.config),sort_keys=True)})
    if not rows: raise RuntimeError("No successful Ray trials. Inspect results/ray/ray_storage logs.")
    trials=pd.DataFrame(rows).sort_values("validation_f1",ascending=False)
    trials.to_csv(OUT/"metrics/ray_trials.csv",index=False)
    best=trials.iloc[0]; cfg=json.loads(best["config"]); key=cfg.pop("model")
    model=get_model(key,**cfg); model.fit(Xtr,ytr); pred=model.predict(Xtest)
    prob=model.predict_proba(Xtest)[:,1] if hasattr(model,"predict_proba") else None
    metrics={"best_model":key,"best_model_name":MODEL_NAMES[key],"best_params":cfg,"validation_f1":float(best["validation_f1"]),"test_accuracy":float(accuracy_score(ytest,pred)),"test_precision":float(precision_score(ytest,pred,zero_division=0)),"test_recall":float(recall_score(ytest,pred,zero_division=0)),"test_f1":float(f1_score(ytest,pred,zero_division=0)),"test_roc_auc":float(roc_auc_score(ytest,prob)) if prob is not None else None,"train_rows":len(Xtr),"test_rows":len(Xtest),"successful_trials":len(trials),"seed":a.seed}
    (OUT/"metrics/best_result.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(ytest,pred,labels=[0,1],display_labels=["Legitimate","Phishing"],cmap="Blues",ax=ax,colorbar=False)
    ax.set_title("Ray Tune best trial - "+MODEL_NAMES[key]); fig.tight_layout(); fig.savefig(OUT/"figures/best_confusion_matrix.png",dpi=180); plt.close(fig)
    report=["# Ray Tune Experiment Report","","## Summary",""]+[f"- {k}: {v}" for k,v in metrics.items()]+["","## Classification report","","",classification_report(ytest,pred,target_names=["Legitimate","Phishing"],zero_division=0),"","## Artifacts","- figures/best_confusion_matrix.png","- metrics/ray_trials.csv","- metrics/best_result.json"]
    (OUT/"reports/ray_experiment_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(trials.head(10).to_string(index=False)); print("Ray outputs:",OUT.resolve())

if __name__=="__main__": main()
