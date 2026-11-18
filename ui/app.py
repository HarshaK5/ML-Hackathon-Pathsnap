"""PathSnap evaluator UI (runs on Laptop 2). Talks to the API only; holds no model, network or labels.
Run:  PATHSNAP_API=http://<laptop-1-ip>:8000 streamlit run ui/app.py"""
import os
import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap - Map Matching", layout="wide", page_icon=":world_map:")
API = st.sidebar.text_input("API base URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")

st.title("PathSnap: GPS map matching & route reconstruction")
st.caption("Select an evaluation trip. The server runs candidate generation -> classifier -> HMM/Viterbi and scores the "
           "result against its private reference route; only the prediction and metrics come back here.")


@st.cache_data(ttl=5, show_spinner=False)
def get(path):
    r = requests.get(API + path, timeout=15); r.raise_for_status(); return r.json()


try:
    health = get("/health"); trips = get("/trips")["trips"]
    st.sidebar.success(f"API online | model: {health['model']} v{health['version']}")
except Exception as e:                                         # noqa
    st.error(f"Cannot reach API at {API}: {e}"); st.stop()

st.subheader("Evaluation trips")
if "sel" not in st.session_state:
    st.session_state.sel, st.session_state.res = None, {}
cols = st.columns(5)
for i, t in enumerate(trips):
    if cols[i % 5].button(f"Trip {t['trip_id']}\n{t['n_gps_points']} GPS pts", key=t["trip_id"], use_container_width=True,
                          type="primary" if st.session_state.sel == t["trip_id"] else "secondary"):
        st.session_state.sel = t["trip_id"]
qp = st.query_params                       # optional deep-link: ?trip=00000040&run=1 (used for screenshots)
if qp.get("trip") and st.session_state.sel is None:
    st.session_state.sel = qp.get("trip")
sel = st.session_state.sel
if not sel:
    st.info("Pick a trip above."); st.stop()

if st.button(f"Run PathSnap on trip {sel}", type="primary") or sel in st.session_state.res or qp.get("run") == "1":
    if sel not in st.session_state.res or st.session_state.get("rerun") == sel:
        with st.spinner("API is processing the trip (candidates, classifier, HMM)..."):
            try:
                r = requests.post(API + "/predict", json={"trip_id": sel}, timeout=300); r.raise_for_status()
                st.session_state.res[sel] = r.json()
            except Exception as e:                              # noqa
                st.error(f"Prediction failed: {e}"); st.stop()
    R = st.session_state.res[sel]
    m, c, T = R["metrics"]["hmm"], R["metrics"]["classifier_only"], R["timing_seconds"]
    st.subheader(f"Results - trip {R['trip_id']}")
    k = st.columns(5)
    k[0].metric("Point accuracy (HMM)", f"{100 * m['point_top1_accuracy']:.2f}%", f"{100 * (m['point_top1_accuracy'] - c['point_top1_accuracy']):+.2f} pp vs classifier")
    k[1].metric("Point accuracy (classifier only)", f"{100 * c['point_top1_accuracy']:.2f}%")
    k[2].metric("Route F1 (HMM)", f"{100 * m['route_f1']:.2f}%")
    k[3].metric("Route continuity (HMM)", f"{100 * m['route_continuity']:.2f}%", f"{100 * (m['route_continuity'] - c['route_continuity']):+.2f} pp")
    k[4].metric("Server time", f"{T['request_total_s']:.2f} s")

    gps = R["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, control_scale=True)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    g1 = folium.FeatureGroup("Raw GPS observations (blue)")
    for la, lo in gps:
        folium.CircleMarker([la, lo], radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=.8).add_to(g1)
    g1.add_to(fmap)
    g2 = folium.FeatureGroup("Classifier-only route (orange, dashed)", show=False)
    for ln in R["classifier_only_route"]:
        folium.PolyLine(ln, color="#f28e2b", weight=4, dash_array="6", opacity=.9).add_to(g2)
    g2.add_to(fmap)
    g3 = folium.FeatureGroup("Predicted route - HMM/Viterbi (red)")
    for ln in R["predicted_route"]:
        folium.PolyLine(ln, color="#d62728", weight=4, opacity=.95).add_to(g3)
    g3.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:6px 10px;border:1px solid #888;'
        'border-radius:6px;font-size:12px"><span style="color:#1f77b4">&#9679;</span> GPS &nbsp;'
        '<span style="color:#f28e2b">&#9644;</span> classifier-only &nbsp;<span style="color:#d62728">&#9644;</span> HMM route</div>'))
    html = fmap._repr_html_()                                  # our own generated map (trusted content)
    if hasattr(st, "iframe"):                                  # newer Streamlit
        st.iframe(html, height=620)
    else:                                                      # older Streamlit
        components.html(html, height=620)

    st.markdown("**Classifier-only vs classifier + HMM**")
    rows = {"Point Top-1 accuracy": "point_top1_accuracy", "Route continuity": "route_continuity", "Route precision": "route_precision",
            "Route recall": "route_recall", "Route F1": "route_f1", "Route sequence similarity": "route_seq_sim"}
    st.table({"Metric": list(rows), "Classifier only": [f"{c[v]:.4f}" for v in rows.values()], "Classifier + HMM": [f"{m[v]:.4f}" for v in rows.values()]})
    st.caption(f"{R['n_points']} GPS points ({R['n_label_valid_points']} scored) | server timing: " + ", ".join(f"{k}={v}" for k, v in T.items()))
