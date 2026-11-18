"""Smoke/E2E test of a running API: calls /health, /trips and /predict for every trip and prints a summary.
Usage: python scripts/10_test_api.py [http://host:8000]"""
import sys, time, requests
api = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
print(requests.get(api + "/health", timeout=10).json())
trips = requests.get(api + "/trips", timeout=10).json()["trips"]
print(f"{len(trips)} trips")
for t in trips:
    t0 = time.time(); r = requests.post(api + "/predict", json={"trip_id": t["trip_id"]}, timeout=300); r.raise_for_status(); j = r.json()
    m, c = j["metrics"]["hmm"], j["metrics"]["classifier_only"]
    print(f"{t['trip_id']}  pts={j['n_points']:5d}  top1 clf={c['point_top1_accuracy']:.4f} hmm={m['point_top1_accuracy']:.4f}  "
          f"routeF1={m['route_f1']:.4f}  continuity={m['route_continuity']:.4f}  {time.time() - t0:.1f}s")
