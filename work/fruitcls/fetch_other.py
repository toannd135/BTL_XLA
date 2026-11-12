"""Tải ảnh vật KHÔNG phải chuối/táo/cam (quả khác, vật tròn hay bị nhận nhầm) từ Wikimedia Commons vào own_other/raw.
Chỉ lấy giấy phép CC/Public domain; ghi nguồn vào own_other/sources.csv. Dùng cho lớp "other" của bộ phân loại loại quả.
Dùng: python work/fruitcls/fetch_other.py [số ảnh mỗi truy vấn=30]"""
import csv, re, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "own_other" / "raw"; OUT.mkdir(parents=True, exist_ok=True)
PER_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 30
API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session(); S.headers["User-Agent"] = "fruit-btl-research/1.0 (student project; image dataset)"
Q = ["strawberries", "strawberry fruit", "grapes bunch", "green grapes", "red grapes", "mango fruit", "pears fruit", "plums fruit",
     "lemons fruit", "limes fruit", "peaches fruit", "nectarines", "apricots", "kiwifruit", "pomegranate fruit", "cherries fruit",
     "tomatoes", "persimmon fruit", "red bell pepper", "yellow bell pepper", "papaya fruit", "pineapple fruit", "passion fruit",
     "dragon fruit", "lychee fruit", "figs fruit", "blueberries", "raspberries", "watermelon slice", "melon fruit",
     "tennis ball", "onions", "potatoes", "eggs in bowl", "baseballs", "christmas baubles", "coconut fruit", "avocado"]
OK = re.compile(r"^(cc[- ]by|cc0|public domain|pd)", re.I)
seen, rows = set(), []
for q in Q:
    got = 0
    for off in (0, 40):
        if got >= PER_Q:
            break
        for _try in range(4):
            resp = S.get(API, params=dict(action="query", generator="search", gsrsearch=f"{q} filetype:bitmap", gsrnamespace=6, gsrlimit=40,
                                          gsroffset=off, prop="imageinfo", iiprop="url|extmetadata|size|mime", iiurlwidth=800, format="json"), timeout=30)
            if resp.status_code == 200 and resp.text.startswith("{"):
                break
            time.sleep(5 * (_try + 1))
        else:
            print("bỏ qua", q, off, resp.status_code); continue
        for p in resp.json().get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii["mime"] not in ("image/jpeg", "image/png") or p["pageid"] in seen or got >= PER_Q:
                continue
            seen.add(p["pageid"]); lic = ii["extmetadata"].get("LicenseShortName", {}).get("value", "")
            if not OK.match(lic) or min(ii["width"], ii["height"]) < 300:
                continue
            rows.append((q, p["pageid"], p["title"], lic, ii["descriptionurl"], ii.get("thumburl") or ii["url"])); got += 1
        time.sleep(0.5)
print("ứng viên:", len(rows))
with open(OUT.parent / "sources.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "query", "title", "license", "page"])
    n = 0
    for q, pid, title, lic, page, url in rows:
        fn = f"oth{pid}.jpg"
        if not (OUT / fn).exists():
            try:
                (OUT / fn).write_bytes(S.get(url, timeout=60).content); time.sleep(0.3)
            except Exception as e:
                print("lỗi", title, e); continue
        w.writerow([fn, q, title, lic, page]); n += 1
print("đã tải:", n, "->", OUT)
