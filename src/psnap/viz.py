"""Folium + matplotlib visualisation helpers (used by EDA, results, and the UI)."""
import numpy as np
import folium
from folium.plugins import MeasureControl


def lonlat_of_arcs(trip, arc_ids):
    a = trip.arcs[np.asarray(arc_ids, dtype=int)]
    return [[(trip.nodes[f, 1], trip.nodes[f, 0]), (trip.nodes[t, 1], trip.nodes[t, 0])] for f, t in a]


def route_polyline(trip, arc_ids):
    """Chain consecutive arcs into [lat,lon] polylines, breaking where arcs are not connected."""
    lines, cur, last_to = [], [], None
    for a in arc_ids:
        f, t = trip.arcs[int(a)]
        if cur and last_to != f:
            lines.append(cur); cur = []
        if not cur:
            cur.append([trip.nodes[f, 1], trip.nodes[f, 0]])
        cur.append([trip.nodes[t, 1], trip.nodes[t, 0]])
        last_to = t
    if cur:
        lines.append(cur)
    return lines


def build_folium_map(trip, pred_arcs=None, ref_route=None, classifier_arcs=None, net_radius_deg=None,
                     max_points=4000, title=None, show_network=True, tiles="OpenStreetMap"):
    """Interactive map with layers: GPS points, road network, reference route, classifier-only route,
    predicted (HMM) route. All layers toggleable; legend included."""
    lat0, lon0 = trip.track[:, 1].mean(), trip.track[:, 0].mean()
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles=tiles, control_scale=True)
    lo, la = trip.track[:, 0], trip.track[:, 1]
    m.fit_bounds([[la.min(), lo.min()], [la.max(), lo.max()]])
    if show_network:
        pad = 0.003
        f, t = trip.arcs[:, 0], trip.arcs[:, 1]
        sel = ((trip.nodes[f, 0] > lo.min() - pad) & (trip.nodes[f, 0] < lo.max() + pad) &
               (trip.nodes[f, 1] > la.min() - pad) & (trip.nodes[f, 1] < la.max() + pad))
        ids = np.flatnonzero(sel)
        if len(ids) > 12000:
            ids = ids[:: len(ids) // 12000 + 1]
        fg = folium.FeatureGroup(name="Road network (grey)", show=True)
        for seg in lonlat_of_arcs(trip, ids):
            folium.PolyLine(seg, color="#9aa0a6", weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
    if ref_route is not None:
        fg = folium.FeatureGroup(name="Reference route (green)", show=True)
        for ln in route_polyline(trip, ref_route):
            folium.PolyLine(ln, color="#2e9e44", weight=7, opacity=0.65).add_to(fg)
        fg.add_to(m)
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only route (orange)", show=False)
        for ln in route_polyline(trip, classifier_arcs):
            folium.PolyLine(ln, color="#f28e2b", weight=4, opacity=0.9, dash_array="6").add_to(fg)
        fg.add_to(m)
    if pred_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route - HMM/Viterbi (red)", show=True)
        for ln in route_polyline(trip, pred_arcs):
            folium.PolyLine(ln, color="#d62728", weight=3.5, opacity=0.95).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS observations (blue)", show=True)
    step = max(1, len(la) // max_points)
    for i in range(0, len(la), step):
        folium.CircleMarker([la[i], lo[i]], radius=2.2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {i}  t={trip.track[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker([la[0], lo[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([la[-1], lo[-1]], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(MeasureControl())
    legend = f"""
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white; padding: 8px 12px;
      border: 1px solid #888; border-radius: 6px; font-size: 12px; line-height: 1.5">
      <b>{title or 'PathSnap trip ' + trip.trip_id}</b><br>
      <span style="color:#1f77b4">&#9679;</span> Raw GPS points<br>
      <span style="color:#9aa0a6">&#9644;</span> Road network<br>
      {'<span style="color:#2e9e44">&#9644;</span> Reference route<br>' if ref_route is not None else ''}
      {'<span style="color:#f28e2b">&#9644;</span> Classifier-only route<br>' if classifier_arcs is not None else ''}
      {'<span style="color:#d62728">&#9644;</span> Predicted route (HMM)<br>' if pred_arcs is not None else ''}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
