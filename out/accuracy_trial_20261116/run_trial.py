"""Run one bounded GPU experiment. Never overwrite/promote the app model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("smoke","train","val"))
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--name",default="obb_patch_v1")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime. This experiment does not train on CPU.")
    if not 1<=args.epochs<=100 or not 1<=args.batch<=32:
        raise ValueError("Invalid epochs/batch")
    root=args.root.resolve()
    manifest=json.loads((root/"input_manifest.json").read_text(encoding="utf-8"))
    if digest(root/"baseline.pt")!=manifest["baseline_model_sha256"]:
        raise ValueError("Baseline checkpoint changed")
    name=args.name+("_smoke" if args.action=="smoke" else "")
    run=args.project/name
    progress=args.project/"progress"/(name+".json")
    progress.parent.mkdir(parents=True,exist_ok=True)

    def save_progress(trainer):
        progress.write_text(json.dumps({"epoch":trainer.epoch+1,"epochs":trainer.epochs,
                                      "metrics":{k:float(v) for k,v in trainer.metrics.items()},
                                      "time":time.time()},indent=2),encoding="utf-8")

    if args.action in ("smoke","train"):
        if args.resume:
            if args.action!="train":
                raise ValueError("Resume is for interrupted train runs only")
            last=run/"weights/last.pt"
            if not last.is_file():
                raise FileNotFoundError(f"No checkpoint to resume: {last}")
            checkpoint=torch.load(last,map_location="cpu",weights_only=False)
            if checkpoint.get("epoch",-1)<0 or checkpoint.get("optimizer") is None:
                raise ValueError("Checkpoint is completed/stripped; choose a NEW run name and RESUME=False")
            del checkpoint
            model=YOLO(str(last))
            model.add_callback("on_fit_epoch_end",save_progress)
            model.train(resume=True,device=0,batch=args.batch)
        else:
            if run.exists():
                raise ValueError(f"Run exists: {run}. Resume an interrupted run or choose another NAME.")
            model=YOLO(str(root/"baseline.pt"))
            model.add_callback("on_fit_epoch_end",save_progress)
            model.train(data=str(root/"data.yaml"),epochs=1 if args.action=="smoke" else args.epochs,
                        imgsz=640,batch=args.batch,device=0,optimizer="AdamW",lr0=.0003,
                        seed=42,patience=7,mosaic=.3,close_mosaic=0 if args.action=="smoke" else 5,
                        degrees=10,flipud=.2,fliplr=.5,workers=2,cache=False,
                        project=str(args.project),name=name,exist_ok=False,plots=True)
        print("Completed",args.action,"weights:",run/"weights",flush=True)
        return

    if args.resume:
        raise ValueError("Set RESUME=False for validation")
    candidate=run/"weights/best.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Train candidate missing: {candidate}")
    report_path=run/"comparison_val.json"
    if report_path.exists():
        raise ValueError("comparison_val.json already exists; preserve the report")
    rows=[]
    for label,path in (("baseline",root/"baseline.pt"),("candidate",candidate)):
        model=YOLO(str(path))
        stats=model.val(data=str(root/"data.yaml"),split="val",imgsz=640,batch=args.batch,
                        conf=.001,iou=.7,device=0,project=str(run/"evaluation"),name=label,exist_ok=False)
        row={"model":label,"sha256":digest(path),"mAP50":float(stats.box.map50),
             "mAP50_95":float(stats.box.map),"AP50":{model.names[int(c)]:float(v)
             for c,v in zip(stats.box.ap_class_index,stats.box.ap50)}}
        count_rows=[]
        selected=[r for r in manifest["files"] if r["origin"]=="submitted_review_geometry_padded" and r["split"]=="val"]
        for item in selected:
            result=model.predict(str(root/item["image"]),imgsz=640,conf=.4,iou=.7,
                                 device=0,agnostic_nms=True,max_det=300,verbose=False)[0]
            truth=Counter(int(line.split()[0]) for line in (root/item["label"]).read_text().splitlines() if line.strip())
            predictions=Counter(int(c) for c in result.obb.cls.tolist())
            count_rows.append({"source":item["source"],"truth":dict(truth),"predicted":dict(predictions),
                               "absolute_error":sum(abs(truth[c]-predictions[c]) for c in (0,1,2)),
                               "exact_all_classes":all(truth[c]==predictions[c] for c in (0,1,2))})
        row["reviewed_banana_val"]={"images":len(count_rows),"mae":sum(r["absolute_error"] for r in count_rows)/len(count_rows),
                                    "exact_rate":sum(r["exact_all_classes"] for r in count_rows)/len(count_rows),"rows":count_rows}
        rows.append(row)
    report={"results":rows,"limitations":manifest["limitations"],
            "protocol":"Raw OBB detector only; no CLIP/teacher/classifier. Same derived legacy val for both models. "
                       "Reviewed banana count scenes are development checks, not a final independent test. "
                       "Do not accept/promote automatically."}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("Report:",report_path,flush=True)


if __name__=="__main__":
    main()
