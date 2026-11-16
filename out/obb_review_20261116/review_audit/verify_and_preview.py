import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
report=json.loads((ROOT/"review_audit/report.json").read_text(encoding="utf-8"))
output=Path(report["output"])
with zipfile.ZipFile(ROOT/"obb_reviewed_v1.zip") as zipped:
    for record in report["items"]:
        original=np.asarray(Image.open(ROOT/"draft_dataset/images"/record["role"]/(record["id"]+".jpg")).convert("RGB"))
        updated=np.asarray(Image.open(output/record["image"]).convert("RGB"))
        h,w=original.shape[:2]
        left,top,right,bottom=record["padding_ltrb"]
        assert np.array_equal(original,updated[top:top+h,left:left+w])
        before=[list(map(float,line.split())) for line in zipped.read(record["label"]).decode("utf-8-sig").splitlines() if line.strip()]
        after=[list(map(float,line.split())) for line in (output/record["label"]).read_text().splitlines() if line.strip()]
        assert len(before)==len(after)
        for a,b in zip(before,after):
            assert a[0]==b[0] and all(0<=value<=1 for value in b[1:])
            assert np.allclose(np.array(a[1:]).reshape(4,2)*[w,h],np.array(b[1:]).reshape(4,2)*[updated.shape[1],updated.shape[0]]-[left,top],atol=.001,rtol=0)
assert hashlib.sha256((ROOT/"obb_reviewed_v1.zip").read_bytes()).hexdigest()==report["source_zip_sha256"]
records={record["id"]:record for record in report["items"]}
identifiers=["95369e3dc95bb75bb39d7499","81d5f64b7d06b9ac1ccea6f0","fd1fb0e99efd248fdd05be40"]
fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=130)
colors={0:"lime",1:"red",2:"cyan"}
names={0:"BANANA",1:"APPLE",2:"ORANGE"}
for axis,identifier in zip(axes,identifiers):
    record=records[identifier]
    image=np.asarray(Image.open(output/record["image"]).convert("RGB"))
    h,w=image.shape[:2]
    axis.imshow(image)
    for k,line in enumerate((output/record["label"]).read_text().splitlines(),1):
        values=list(map(float,line.split())); cls=int(values[0])
        points=np.array(values[1:]).reshape(4,2)*[w,h]
        axis.add_patch(Polygon(points,closed=True,fill=False,color=colors[cls],linewidth=1.5))
        axis.text(*points.mean(0),f"{k}:{names[cls]}",fontsize=8,ha="center",bbox=dict(facecolor=colors[cls],alpha=.7,pad=1))
    frame=10 if record["role"]=="train" else 4 if identifier.startswith("81d5") else 7
    role=record["role"]
    axis.set_title(f"{role} frame {frame} - {identifier[:8]}",fontsize=11)
    axis.axis("off")
fig.tight_layout()
fig.savefig(ROOT/"review_audit/can_sua_3_anh.png")
plt.close(fig)
summary={"verified_images":len(records),"original_pixels_preserved":True,
         "classes_and_box_counts_unchanged":True,"obb_pixel_coordinates_preserved_within":.001,
         "all_new_coordinates_normalized":True,"source_zip_unchanged":True,"training_ready":False}
(ROOT/"review_audit/geometry_verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary))
