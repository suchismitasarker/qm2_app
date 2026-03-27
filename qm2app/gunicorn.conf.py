# gunicorn.conf.py  —  Production configuration for QM2 Data Analysis App
# Usage:  gunicorn -c gunicorn.conf.py app:app

import multiprocessing

# ── Binding ───────────────────────────────────────────────────────────────────
# Listen on localhost only; Nginx will proxy from port 80/443
bind = "127.0.0.1:5000"

# ── Workers ───────────────────────────────────────────────────────────────────
# IO-heavy workload (HDF5 reads, image generation) — use threads not processes
# so the file/memory cache (dicts, locks) is shared across requests.
workers = 1
threads = 8                    # concurrent requests per worker
worker_class = "gthread"
worker_connections = 100

# ── Timeouts ──────────────────────────────────────────────────────────────────
# Slice rendering + peak fitting can take several seconds on large files
timeout       = 120            # kill worker if silent for 2 min
graceful_timeout = 30          # time allowed for in-flight requests on reload
keepalive     = 5

# ── Logging ───────────────────────────────────────────────────────────────────
loglevel      = "info"
accesslog     = "/var/log/qm2app/access.log"
errorlog      = "/var/log/qm2app/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sus'

# ── Process ───────────────────────────────────────────────────────────────────
daemon        = False          # let systemd manage the process
pidfile       = "/run/qm2app/gunicorn.pid"
user          = "qm2app"       # dedicated service account (see README)
group         = "qm2app"

# ── Reload on code change (dev-only; comment out for production) ──────────────
# reload = True
