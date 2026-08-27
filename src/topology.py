import re
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Color constants aligned with central theme
COLOR_CRITICAL = "#FF3B3B"
COLOR_WARNING  = "#FFB81C"
COLOR_HEALTHY  = "#00D9E8"
COLOR_SURFACE  = "#0D1822"
COLOR_BORDER   = "#263743"
COLOR_TEXT_PRI = "#E8EEF2"
COLOR_TEXT_MUT = "#81939F"

def get_location_coords(location_name: str) -> tuple:
    """
    Deterministically computes a fixed 2D (x, y) coordinate from the anonymized location name.
    Uses MD5 hash to ensure identical coordinates across reruns and simulated updates.
    """
    loc_clean = str(location_name).strip().lower()
    h = hashlib.md5(loc_clean.encode('utf-8')).hexdigest()
    # Map first 8 hex characters to x and next 8 to y in range [10, 90]
    val_x = int(h[0:8], 16) % 10000 / 10000.0
    val_y = int(h[8:16], 16) % 10000 / 10000.0
    
    # Scale with margin to prevent edge clipping
    x = 10.0 + (val_x * 80.0)
    y = 10.0 + (val_y * 80.0)
    return round(x, 2), round(y, 2)

def build_topology_figure(df_results: pd.DataFrame, max_nodes: int = 35) -> go.Figure:
    """
    Builds an interactive node-link network topology visualization using the actual
    runtime dataframe (df_results) loaded by the application.
    
    Args:
        df_results (pd.DataFrame): Current active device dataframe containing 'location',
                                   'health_score', 'status', and 'id'.
        max_nodes (int): Number of top active/critical locations to represent in the topology.
        
    Returns:
        go.Figure: High-density Plotly node-link network graph matching NERVE NOC visual layout.
    """
    if df_results.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=COLOR_SURFACE, plot_bgcolor=COLOR_SURFACE)
        return fig
        
    # Aggregate actual runtime metrics per location
    loc_agg = (
        df_results.groupby('location')
        .agg(
            avg_health=('health_score', 'mean'),
            device_count=('id', 'count'),
            critical_count=('status', lambda s: (s == 'Critical').sum()),
            warning_count=('status', lambda s: (s == 'Warning').sum()),
            healthy_count=('status', lambda s: (s == 'Healthy').sum()),
            device_ids=('id', list)
        )
        .reset_index()
    )
    
    loc_agg['priority_score'] = (loc_agg['critical_count'] * 3) + (loc_agg['warning_count'] * 1.5) + (loc_agg['device_count'])
    loc_agg = loc_agg.sort_values(by=['priority_score', 'avg_health'], ascending=[False, True]).head(max_nodes).copy()
    
    def get_status_props(score):
        if score < 40.0:
            return "Critical", COLOR_CRITICAL
        elif score < 70.0:
            return "Warning", COLOR_WARNING
        else:
            return "Healthy", COLOR_HEALTHY
            
    nodes = []
    for _, row in loc_agg.iterrows():
        loc_name = str(row['location'])
        x, y = get_location_coords(loc_name)
        status_label, color = get_status_props(row['avg_health'])
        
        dev_cnt = int(row['device_count'])
        node_size = min(24, max(12, 10 + (dev_cnt * 2.5)))
        
        dev_list_str = ", ".join([f"#{d}" for d in row['device_ids'][:4]])
        if len(row['device_ids']) > 4:
            dev_list_str += f" (+{len(row['device_ids'])-4} more)"
            
        hover_text = (
            f"<b>Node: {loc_name}</b><br>"
            f"Status: <span style='color:{color}; font-weight:bold;'>{status_label}</span><br>"
            f"Avg Health Score: <b>{row['avg_health']:.1f}%</b><br>"
            f"Active Monitored Devices: <b>{dev_cnt}</b><br>"
            f"Critical: {row['critical_count']} | Warning: {row['warning_count']} | Healthy: {row['healthy_count']}<br>"
            f"Devices: {dev_list_str}"
        )
        
        nodes.append({
            'location': loc_name,
            'x': x,
            'y': y,
            'health': row['avg_health'],
            'status': status_label,
            'color': color,
            'size': node_size,
            'hover': hover_text,
            'critical_count': row['critical_count']
        })
        
    edge_x = []
    edge_y = []
    coords = np.array([[n['x'], n['y']] for n in nodes])
    n_nodes = len(nodes)
    
    for i in range(n_nodes):
        dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        dists[i] = np.inf
        k_neighbors = min(3, n_nodes - 1)
        nearest_indices = np.argsort(dists)[:k_neighbors]
        
        for n_idx in nearest_indices:
            if dists[n_idx] < 55.0:
                edge_x.extend([nodes[i]['x'], nodes[n_idx]['x'], None])
                edge_y.extend([nodes[i]['y'], nodes[n_idx]['y'], None])
                
    fig = go.Figure()
    
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1.2, color='rgba(0, 219, 233, 0.22)'),
            hoverinfo='none',
            showlegend=False
        ))
        
    for n in nodes:
        if n['status'] in ('Critical', 'Warning'):
            halo_color = 'rgba(255, 59, 59, 0.35)' if n['status'] == 'Critical' else 'rgba(255, 186, 32, 0.3)'
            fig.add_trace(go.Scatter(
                x=[n['x']],
                y=[n['y']],
                mode='markers',
                marker=dict(
                    size=n['size'] + 10,
                    color=halo_color,
                    line=dict(width=0)
                ),
                hoverinfo='none',
                showlegend=False
            ))
            
    fig.add_trace(go.Scatter(
        x=[n['x'] for n in nodes],
        y=[n['y'] for n in nodes],
        mode='markers+text',
        marker=dict(
            size=[n['size'] for n in nodes],
            color=[n['color'] for n in nodes],
            line=dict(width=1.5, color=COLOR_BORDER),
            opacity=0.95
        ),
        text=[n['location'].replace('location ', 'L') for n in nodes],
        textposition="bottom center",
        textfont=dict(family='JetBrains Mono', size=9, color=COLOR_TEXT_MUT),
        hovertext=[n['hover'] for n in nodes],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        paper_bgcolor='#0D1822',
        plot_bgcolor='#0D1822',
        margin=dict(l=5, r=5, t=5, b=5),
        height=175,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 100],
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 100],
            fixedrange=True
        ),
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE,
            bordercolor=COLOR_BORDER,
            font=dict(family='JetBrains Mono', size=11, color=COLOR_TEXT_PRI)
        )
    )
    
    return fig

def get_location_geo_coords(location_name: str) -> tuple:
    """
    Deterministically computes a fixed global (lat, lon) coordinate from the anonymized location name.
    Uses MD5 hash to ensure identical coordinates across reruns.
    """
    loc_clean = str(location_name).strip().lower()
    h = hashlib.md5(loc_clean.encode('utf-8')).hexdigest()
    val_x = int(h[0:8], 16) % 10000 / 10000.0
    val_y = int(h[8:16], 16) % 10000 / 10000.0
    lat = -32.0 + (val_x * 85.0)
    lon = -115.0 + (val_y * 235.0)
    return round(lat, 2), round(lon, 2)

def build_geo_risk_figure(df_results: pd.DataFrame, max_locations: int = 40) -> go.Figure:
    """
    Builds a decorative dark world-map-style visualization of Risk by Location.
    Uses glowing heat blobs scaled and colored by real location health scores.
    """
    if df_results.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=COLOR_SURFACE, plot_bgcolor=COLOR_SURFACE)
        return fig
        
    loc_agg = (
        df_results.groupby('location')
        .agg(
            avg_health=('health_score', 'mean'),
            device_count=('id', 'count'),
            critical_count=('status', lambda s: (s == 'Critical').sum()),
            warning_count=('status', lambda s: (s == 'Warning').sum()),
            healthy_count=('status', lambda s: (s == 'Healthy').sum())
        )
        .reset_index()
    )
    
    loc_agg['priority'] = (loc_agg['critical_count'] * 3) + (loc_agg['warning_count'] * 1.5) + loc_agg['device_count']
    loc_agg = loc_agg.sort_values(by=['priority', 'avg_health'], ascending=[False, True]).head(max_locations).copy()
    
    lats, lons, hover_texts, halo_colors, core_colors, halo_sizes, core_sizes = [], [], [], [], [], [], []
    
    for _, r in loc_agg.iterrows():
        lat, lon = get_location_geo_coords(str(r['location']))
        health = r['avg_health']
        dev_cnt = int(r['device_count'])
        
        if health < 40.0:
            status_str = "Critical"
            c_color = COLOR_CRITICAL
            h_color = "rgba(255, 59, 59, 0.45)"
            h_size = min(32, max(18, 16 + dev_cnt * 2))
            c_size = 8
        elif health < 70.0:
            status_str = "Warning"
            c_color = COLOR_WARNING
            h_color = "rgba(255, 186, 32, 0.35)"
            h_size = min(26, max(14, 12 + dev_cnt * 1.5))
            c_size = 6
        else:
            status_str = "Healthy"
            c_color = COLOR_HEALTHY
            h_color = "rgba(0, 219, 233, 0.28)"
            h_size = min(22, max(12, 10 + dev_cnt))
            c_size = 5
            
        htext = (
            f"<b>{r['location']}</b><br>"
            f"Status: <b>{status_str}</b><br>"
            f"Avg Health Score: <b>{health:.1f}%</b><br>"
            f"Monitored Devices: <b>{dev_cnt}</b><br>"
            f"Critical: {r['critical_count']} | Warning: {r['warning_count']} | Healthy: {r['healthy_count']}"
        )
        
        lats.append(lat)
        lons.append(lon)
        hover_texts.append(htext)
        halo_colors.append(h_color)
        core_colors.append(c_color)
        halo_sizes.append(h_size)
        core_sizes.append(c_size)
        
    fig = go.Figure()
    
    # Glowing Heat Halo Layer
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=halo_sizes,
            color=halo_colors,
            line=dict(width=0)
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Core Markers Layer
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=core_sizes,
            color=core_colors,
            line=dict(width=1, color='#ffffff')
        ),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False
    ))
    
    fig.update_geos(
        projection_type='natural earth',
        showcoastlines=True,
        coastlinecolor='#263743',
        showland=True,
        landcolor='#111C28',
        showocean=True,
        oceancolor='#070D14',
        showlakes=False,
        showcountries=True,
        countrycolor='#263743',
        bgcolor='#0D1822'
    )
    
    fig.update_layout(
        paper_bgcolor='#0D1822',
        plot_bgcolor='#0D1822',
        margin=dict(l=0, r=0, t=0, b=0),
        height=175,
        hoverlabel=dict(
            bgcolor='#0D1822',
            bordercolor='#263743',
            font=dict(family='JetBrains Mono', size=11, color='#E8EEF2')
        )
    )
    
    return fig

