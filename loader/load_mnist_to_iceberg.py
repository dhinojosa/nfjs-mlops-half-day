import os, numpy as np
from datetime import datetime
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T

# --------- Config from env ----------
ROWS         = int(os.getenv("ROWS", "2000"))
NOISE_STD    = float(os.getenv("NOISE_STD", "0.0"))
DIGITS_FILTER = os.getenv("DIGITS_FILTER", "").strip()
EVENT_DATE   = os.getenv("EVENT_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
TARGET_TABLE = os.getenv("TARGET_TABLE", "digits.db.raw")

# --------- Build Spark --------------
spark = SparkSession.builder.appName("MNIST->Iceberg").getOrCreate()

# Ensure DB and table exist with a stable schema
spark.sql("CREATE DATABASE IF NOT EXISTS digits.db")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
  id STRING,
  label INT,
  p0 DOUBLE, p1 DOUBLE, p2 DOUBLE, p3 DOUBLE, p4 DOUBLE, p5 DOUBLE, p6 DOUBLE, p7 DOUBLE,
  p8 DOUBLE, p9 DOUBLE, p10 DOUBLE, p11 DOUBLE, p12 DOUBLE, p13 DOUBLE, p14 DOUBLE, p15 DOUBLE,
  p16 DOUBLE, p17 DOUBLE, p18 DOUBLE, p19 DOUBLE, p20 DOUBLE, p21 DOUBLE, p22 DOUBLE, p23 DOUBLE,
  p24 DOUBLE, p25 DOUBLE, p26 DOUBLE, p27 DOUBLE, p28 DOUBLE, p29 DOUBLE, p30 DOUBLE, p31 DOUBLE,
  p32 DOUBLE, p33 DOUBLE, p34 DOUBLE, p35 DOUBLE, p36 DOUBLE, p37 DOUBLE, p38 DOUBLE, p39 DOUBLE,
  p40 DOUBLE, p41 DOUBLE, p42 DOUBLE, p43 DOUBLE, p44 DOUBLE, p45 DOUBLE, p46 DOUBLE, p47 DOUBLE,
  p48 DOUBLE, p49 DOUBLE, p50 DOUBLE, p51 DOUBLE, p52 DOUBLE, p53 DOUBLE, p54 DOUBLE, p55 DOUBLE,
  p56 DOUBLE, p57 DOUBLE, p58 DOUBLE, p59 DOUBLE, p60 DOUBLE, p61 DOUBLE, p62 DOUBLE, p63 DOUBLE,
  event_date DATE
)
USING iceberg
PARTITIONED BY (event_date)
""")

# --------- Fetch MNIST (torchvision) ----------
# torchvision isn’t in the image, so we load via the raw ubyte files (tiny helper):
import gzip, urllib.request, os, pathlib

DATA_DIR = "/tmp/mnist"
pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

BASE = "http://yann.lecun.com/exdb/mnist/"
FILES = {
  "train_images": "train-images-idx3-ubyte.gz",
  "train_labels": "train-labels-idx1-ubyte.gz",
  "test_images":  "t10k-images-idx3-ubyte.gz",
  "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

for fname in FILES.values():
    dst = os.path.join(DATA_DIR, fname)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(BASE + fname, dst)

def load_images(path):
    with gzip.open(path, "rb") as f:
        _ = f.read(16)  # header
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(-1, 28, 28)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _ = f.read(8)   # header
        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8)

X = np.concatenate([
    load_images(os.path.join(DATA_DIR, FILES["train_images"])),
    load_images(os.path.join(DATA_DIR, FILES["test_images"])),
], axis=0)

y = np.concatenate([
    load_labels(os.path.join(DATA_DIR, FILES["train_labels"])),
    load_labels(os.path.join(DATA_DIR, FILES["test_labels"])),
], axis=0)

# Optional: filter to certain digits to force drift (e.g., "8,9")
if DIGITS_FILTER:
    wanted = set(int(d.strip()) for d in DIGITS_FILTER.split(",") if d.strip())
    mask = np.isin(y, list(wanted))
    X, y = X[mask], y[mask]

# Limit sample count
N = min(ROWS, X.shape[0])
X, y = X[:N], y[:N]

# Resize to 8x8 and flatten -> 64 features similar to sklearn digits
def to_8x8_flat(img28):
    img = Image.fromarray(img28)
    img = img.resize((8, 8), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return (arr / 1.0).reshape(-1)  # keep raw-ish scale; sklearn digits often 0..16 but we can log-scale later

X8 = np.stack([to_8x8_flat(im) for im in X], axis=0)

# Add Gaussian noise if requested
if NOISE_STD > 0:
    X8 = X8 + np.random.normal(0.0, NOISE_STD, X8.shape)

# Build rows for Spark
rows = []
for i in range(X8.shape[0]):
    pixels = X8[i].tolist()
    rows.append([f"mnist-{i}", int(y[i]), *[float(v) for v in pixels], EVENT_DATE])

schema = T.StructType(
    [T.StructField("id", T.StringType(), False),
     T.StructField("label", T.IntegerType(), False)] +
    [T.StructField(f"p{k}", T.DoubleType(), False) for k in range(64)] +
    [T.StructField("event_date", T.DateType(), False)]
)

df = spark.createDataFrame(rows, schema)
df = df.withColumn("event_date", F.to_date("event_date"))

# Append into Iceberg
df.writeTo(TARGET_TABLE).append()

print(f"Wrote {df.count()} rows to {TARGET_TABLE} with event_date={EVENT_DATE}, noise={NOISE_STD}, filter='{DIGITS_FILTER}'")
spark.stop()
