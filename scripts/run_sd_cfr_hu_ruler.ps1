# Short HU SD-CFR ruler (pre Gate A/B)
$env:OMP_NUM_THREADS = "6"
python scripts/train_sd_cfr_hu.py `
  --iterations 10 `
  --traversals 2 `
  --checkpoint-every 5 `
  --checkpoint-dir checkpoints/sd_cfr_hu_ruler `
  --output checkpoints/sd_cfr_hu_ruler_latest.pkl
