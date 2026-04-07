#!/usr/bin/env python3
"""
Dash-App zur Visualisierung der vorberechneten Embeddings.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from pathlib import Path

app = dash.Dash(__name__)

# Verfügbare CSV-Dateien finden
data_dir = Path(__file__).parent.parent.parent / "data"
manifold_dir = Path(__file__).parent.parent.parent / "manifold_embeddings"

# Suche in beiden Verzeichnissen
csv_files = list(data_dir.glob("embeddings_*.csv"))
manifold_files = list(manifold_dir.glob("embeddings_*.csv")) if manifold_dir.exists() else []

all_csv_files = csv_files + manifold_files

# Falls keine Dateien vorhanden
if not all_csv_files:
    print("⚠️  Keine Embedding-Dateien gefunden!")
    print("   Bitte zuerst compute_and_save_embeddings.py oder process_activity_logs.py ausführen")
    available_files = []
else:
    # Erstelle Dict: Dateiname -> vollständiger Pfad
    file_dict = {f.name: f for f in all_csv_files}
    available_files = list(file_dict.keys())

print(f"Gefundene Embedding-Dateien: {available_files}")

app.layout = html.Div([
    html.H1("SHD Neuronale Aktivität - Manifold Visualisierung", 
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label("Embedding-Datei:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': f, 'value': f} for f in available_files],
                value=available_files[0] if available_files else None,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Label (Ziffer):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='label-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}] + 
                        [{'label': str(i), 'value': i} for i in range(10)],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}, id='label-filter-container'),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Layer:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='layer-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}],  # Wird dynamisch aktualisiert
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Epoche:", id='epoch-label', style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='epoch-slider',
                min=0, max=100, value=0, step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}, id='epoch-filter-container'),
    ], style={'marginBottom': 20}),
    
    # Metadaten-Filter (werden dynamisch basierend auf verfügbaren Spalten angezeigt)
    html.Div([
        html.Div([
            html.Label("Sprache:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='language-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '1%'}, id='language-filter-container'),
        
        html.Div([
            html.Label("Speaker:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='speaker-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '1%'}, id='speaker-filter-container'),
        
        html.Div([
            html.Label("⚧ Geschlecht:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': 'Alle', 'value': 'all'}],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '32%', 'display': 'inline-block'}, id='gender-filter-container'),
    ], style={'marginBottom': 20}, id='metadata-filters-container'),
    
    html.Div([
        html.Div([
            html.Label("📈 Anzahl Samples:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='n-samples-slider',
                min=1, max=1000, value=10, step=1,
                marks={i: str(i) for i in [1, 10, 25, 50, 100, 250, 500, 1000]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("📊 Visualisierungstyp:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='viz-type',
                options=[
                    {'label': ' Trajektorien (Linien)', 'value': 'lines'},
                    {'label': ' Trajektorien (nach Label)', 'value': 'lines_by_label'},
                    {'label': ' Scatter (Punkte)', 'value': 'scatter'},
                    {'label': ' Animation über Zeit', 'value': 'animation'},
                    {'label': ' Animation über Epochen', 'value': 'epoch_animation'}
                ],
                value='lines',
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'marginBottom': 20}),
    
    # Farbcodierungs-Optionen
    html.Div([
        html.Label("🎨 Farbcodierung (für Scatter & Trajektorien):", style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='color-by',
            options=[
                {'label': ' Label', 'value': 'label'},
                {'label': ' Speaker', 'value': 'speaker'},
                {'label': ' Geschlecht', 'value': 'speaker_gender'},
                {'label': ' Trial', 'value': 'trial'},
                {'label': ' Zeit (default)', 'value': 'time'},
                {'label': ' ⚫ Schwarz (monochrom)', 'value': 'black'}
            ],
            value='time',
            inline=True
        ),
    ], style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label("Zeitbin-Bereich:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='timebin-range-slider',
                min=0, max=100, value=[0, 100], step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}),
    ], style={'marginBottom': 20}),
    
    # Bereichs-Filter für X, Y, Z Achsen
    html.Div([
        html.H4("🔍 Bereichs-Filter (Zoom in bestimmte Regionen):", 
                style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Nutze die Slider um einen bestimmten Bereich im Manifold zu filtern und näher zu betrachten.",
               style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '10px'}),
    ], style={'marginBottom': 10}),
    
    html.Div([
        html.Div([
            html.Label("📏 X-Achsen Bereich:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.RangeSlider(
                id='x-range-slider',
                min=-50, max=50, value=[-50, 50], step=0.5,
                marks={i: str(i) for i in range(-50, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '1%'}),
        
        html.Div([
            html.Label("📏 Y-Achsen Bereich:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.RangeSlider(
                id='y-range-slider',
                min=-50, max=50, value=[-50, 50], step=0.5,
                marks={i: str(i) for i in range(-50, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '32%', 'display': 'inline-block', 'paddingRight': '1%'}),
        
        html.Div([
            html.Label("📏 Z-Achsen Bereich:", style={'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.RangeSlider(
                id='z-range-slider',
                min=-50, max=50, value=[-50, 50], step=0.5,
                marks={i: str(i) for i in range(-50, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),
    ], style={'marginBottom': 10}),
    
    html.Div([
        html.Button('🔄 Bereichs-Filter zurücksetzen', id='reset-range-button', n_clicks=0,
                   style={'marginRight': '10px', 'padding': '8px 15px', 'backgroundColor': '#3498db', 
                          'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        html.Span(id='range-filter-info', style={'fontSize': '0.9em', 'color': '#666'})
    ], style={'marginBottom': 30}),
    
    # Info-Box für Epochen-Animation
    html.Div(id='epoch-animation-info', style={'marginBottom': 20}),
    
    dcc.Graph(id='3d-plot', style={'height': '700px'}),
    
    html.Div([
        html.H3("Statistiken:", style={'color': '#2c3e50'}),
        html.Div(id='stats-output', style={
            'backgroundColor': '#ecf0f1',
            'padding': '15px',
            'borderRadius': '5px',
            'fontFamily': 'monospace'
        })
    ], style={'marginTop': 30})
])

@callback(
    [Output('epoch-slider', 'marks'),
     Output('epoch-slider', 'max'),
     Output('epoch-slider', 'value')],
    Input('file-dropdown', 'value')
)
def update_epoch_slider(filename):
    """Aktualisiert die Epochen-Slider-Optionen basierend auf der ausgewählten Datei."""
    if not filename:
        return {0: '0'}, 0, 0
    
    try:
        filepath = file_dict.get(filename)
        if filepath is None:
            return {0: '0'}, 0, 0
        
        df = pd.read_csv(filepath)
        
        # Prüfe ob 'epoch' Spalte existiert
        if 'epoch' in df.columns:
            epochs = sorted(df['epoch'].unique())
            if epochs:
                # Konvertiere numpy.int64 zu Python int
                epochs = [int(epoch) for epoch in epochs]
                max_epoch = int(max(epochs))
                # Erstelle Marks alle 5 Epochen oder bei wichtigen Epochen
                marks = {}
                for i in range(0, max_epoch + 1, max(1, max_epoch // 10)):
                    marks[i] = str(i)
                # Füge die tatsächlichen Epochen hinzu
                for epoch in epochs:
                    marks[epoch] = str(epoch)
                return marks, max_epoch, epochs[0]
            else:
                return {0: '0'}, 0, 0
        else:
            return {0: '0'}, 0, 0
        
    except:
        return {0: '0'}, 0, 0

@callback(
    [Output('timebin-range-slider', 'min'),
     Output('timebin-range-slider', 'max'),
     Output('timebin-range-slider', 'value'),
     Output('timebin-range-slider', 'marks')],
    Input('file-dropdown', 'value')
)
def update_timebin_slider(filename):
    """Aktualisiert die Zeitbin-Slider-Optionen basierend auf der ausgewählten Datei."""
    if not filename:
        return 0, 100, [0, 100], {i: str(i) for i in range(0, 101, 10)}
    
    try:
        filepath = file_dict.get(filename)
        if filepath is None:
            return 0, 100, [0, 100], {i: str(i) for i in range(0, 101, 10)}
        
        df = pd.read_csv(filepath)
        
        # Prüfe ob 'time_bin' Spalte existiert
        if 'time_bin' in df.columns:
            time_bins = sorted(df['time_bin'].unique())
            if time_bins:
                min_timebin = int(min(time_bins))
                max_timebin = int(max(time_bins))
                
                # Erstelle Marks alle 10 Zeitbins oder bei wichtigen Zeitbins
                marks = {}
                step = max(1, (max_timebin - min_timebin) // 10)
                for i in range(min_timebin, max_timebin + 1, step):
                    marks[i] = str(i)
                # Füge die tatsächlichen Zeitbins hinzu
                for timebin in time_bins:
                    marks[int(timebin)] = str(int(timebin))
                
                return min_timebin, max_timebin, [min_timebin, max_timebin], marks
            else:
                return 0, 100, [0, 100], {i: str(i) for i in range(0, 101, 10)}
        else:
            return 0, 100, [0, 100], {i: str(i) for i in range(0, 101, 10)}
        
    except:
        return 0, 100, [0, 100], {i: str(i) for i in range(0, 101, 10)}

@callback(
    Output('epoch-label', 'children'),
    Input('epoch-slider', 'value')
)
def update_epoch_label(epoch_value):
    """Aktualisiert das Epochen-Label mit dem aktuellen Wert."""
    return f"Epoche: {epoch_value}"

@callback(
    [Output('x-range-slider', 'min'),
     Output('x-range-slider', 'max'),
     Output('x-range-slider', 'value'),
     Output('x-range-slider', 'marks'),
     Output('y-range-slider', 'min'),
     Output('y-range-slider', 'max'),
     Output('y-range-slider', 'value'),
     Output('y-range-slider', 'marks'),
     Output('z-range-slider', 'min'),
     Output('z-range-slider', 'max'),
     Output('z-range-slider', 'value'),
     Output('z-range-slider', 'marks')],
    [Input('file-dropdown', 'value'),
     Input('reset-range-button', 'n_clicks')]
)
def update_range_sliders(filename, n_clicks):
    """Aktualisiert die X, Y, Z Range-Slider basierend auf den tatsächlichen Daten."""
    if not filename:
        default = (-50, 50, [-50, 50], {i: str(i) for i in range(-50, 51, 10)})
        return default * 3  # Für x, y, z
    
    try:
        filepath = file_dict.get(filename)
        if filepath is None:
            default = (-50, 50, [-50, 50], {i: str(i) for i in range(-50, 51, 10)})
            return default * 3
        
        df = pd.read_csv(filepath)
        
        # Berechne Min/Max für jede Achse mit etwas Puffer
        def get_range_with_buffer(values, buffer=0.1):
            vmin, vmax = values.min(), values.max()
            range_size = vmax - vmin
            buffer_size = range_size * buffer
            return (
                float(np.floor(vmin - buffer_size)),
                float(np.ceil(vmax + buffer_size))
            )
        
        x_min, x_max = get_range_with_buffer(df['x'])
        y_min, y_max = get_range_with_buffer(df['y'])
        z_min, z_max = get_range_with_buffer(df['z'])
        
        # Erstelle Marks
        def create_marks(vmin, vmax, n_marks=11):
            step = (vmax - vmin) / (n_marks - 1)
            return {int(vmin + i * step): f'{vmin + i * step:.1f}' 
                    for i in range(n_marks)}
        
        x_marks = create_marks(x_min, x_max)
        y_marks = create_marks(y_min, y_max)
        z_marks = create_marks(z_min, z_max)
        
        return (
            x_min, x_max, [x_min, x_max], x_marks,
            y_min, y_max, [y_min, y_max], y_marks,
            z_min, z_max, [z_min, z_max], z_marks
        )
        
    except Exception as e:
        print(f"Fehler beim Update der Range-Slider: {e}")
        default = (-50, 50, [-50, 50], {i: str(i) for i in range(-50, 51, 10)})
        return default * 3


@callback(
    Output('range-filter-info', 'children'),
    [Input('x-range-slider', 'value'),
     Input('y-range-slider', 'value'),
     Input('z-range-slider', 'value')]
)
def update_range_info(x_range, y_range, z_range):
    """Zeigt Info über aktiven Bereichs-Filter."""
    if x_range and y_range and z_range:
        return f"📊 Aktiver Filter: X=[{x_range[0]:.1f}, {x_range[1]:.1f}], Y=[{y_range[0]:.1f}, {y_range[1]:.1f}], Z=[{z_range[0]:.1f}, {z_range[1]:.1f}]"
    return ""


@callback(
    [Output('layer-dropdown', 'options'),
     Output('language-dropdown', 'options'),
     Output('speaker-dropdown', 'options'),
     Output('gender-dropdown', 'options'),
     Output('metadata-filters-container', 'style')],
    Input('file-dropdown', 'value')
)
def update_metadata_filters(filename):
    """Aktualisiert die Layer- und Metadaten-Filter basierend auf der ausgewählten Datei."""
    if not filename:
        return (
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            {'display': 'none'}
        )
    
    try:
        filepath = file_dict.get(filename)
        if filepath is None:
            return (
                [{'label': 'Alle', 'value': 'all'}],
                [{'label': 'Alle', 'value': 'all'}],
                [{'label': 'Alle', 'value': 'all'}],
                [{'label': 'Alle', 'value': 'all'}],
                {'display': 'none'}
            )
        
        df = pd.read_csv(filepath)
        
        # Layer-Optionen (dynamisch basierend auf verfügbaren Layern)
        layer_opts = [{'label': 'Alle', 'value': 'all'}]
        if 'layer' in df.columns:
            layers = sorted(df['layer'].unique())
            # Erstelle benutzerfreundliche Labels
            layer_labels = {
                'hidden1': 'Hidden Layer 1',
                'hidden2': 'Hidden Layer 2',
                'output': 'Output Layer'
            }
            for layer in layers:
                label = layer_labels.get(layer, layer.capitalize())
                layer_opts.append({'label': label, 'value': layer})
        
        # Prüfe welche Metadaten-Spalten vorhanden sind
        has_metadata = any(col in df.columns for col in ['language', 'speaker', 'speaker_gender'])
        
        if not has_metadata:
            return (
                layer_opts,
                [{'label': 'Alle', 'value': 'all'}],
                [{'label': 'Alle', 'value': 'all'}],
                [{'label': 'Alle', 'value': 'all'}],
                {'display': 'none', 'marginBottom': 20}
            )
        
        # Language-Optionen
        language_opts = [{'label': 'Alle', 'value': 'all'}]
        if 'language' in df.columns:
            languages = sorted(df['language'].unique())
            language_opts.extend([{'label': lang.capitalize(), 'value': lang} for lang in languages])
        
        # Speaker-Optionen
        speaker_opts = [{'label': 'Alle', 'value': 'all'}]
        if 'speaker' in df.columns:
            speakers = sorted(df['speaker'].unique())
            speaker_opts.extend([{'label': f'Speaker {spk}', 'value': int(spk)} for spk in speakers])
        
        # Gender-Optionen (male/female)
        gender_opts = [{'label': 'Alle', 'value': 'all'}]
        if 'speaker_gender' in df.columns:
            genders = sorted(df['speaker_gender'].dropna().unique())
            gender_map = {'male': 'Männlich (male)', 'female': 'Weiblich (female)'}
            gender_opts.extend([{'label': gender_map.get(gender, gender), 'value': gender} for gender in genders])
        
        return (
            layer_opts,
            language_opts,
            speaker_opts,
            gender_opts,
            {'display': 'block', 'marginBottom': 20}
        )
        
    except Exception as e:
        print(f"Fehler beim Laden der Metadaten: {e}")
        return (
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            [{'label': 'Alle', 'value': 'all'}],
            {'display': 'none'}
        )

@callback(
    [Output('epoch-filter-container', 'style'),
     Output('label-filter-container', 'style'),
     Output('epoch-animation-info', 'children')],
    Input('viz-type', 'value')
)
def toggle_filters(viz_type):
    """Deaktiviert nur den Epochen-Filter bei Epochen-Animation."""
    if viz_type == 'epoch_animation':
        # Nur Epochen-Filter verstecken, Label-Filter bleibt sichtbar
        info_box = html.Div([
            html.P("ℹ️ Bei Epochen-Animation können Sie ein Label auswählen, um dessen Entwicklung über alle Epochen zu sehen.",
                   style={'backgroundColor': '#e8f4fd', 'padding': '10px', 'borderRadius': '5px', 
                          'border': '1px solid #bee5eb', 'color': '#0c5460', 'margin': 0})
        ])
        return {'width': '48%', 'display': 'none'}, {'width': '48%', 'display': 'inline-block'}, info_box
    else:
        # Beide Filter anzeigen bei anderen Visualisierungen
        return {'width': '48%', 'display': 'inline-block'}, {'width': '48%', 'display': 'inline-block'}, ""

@callback(
    [Output('3d-plot', 'figure'),
     Output('stats-output', 'children')],
    [Input('file-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('layer-dropdown', 'value'),
     Input('epoch-slider', 'value'),
     Input('n-samples-slider', 'value'),
     Input('viz-type', 'value'),
     Input('timebin-range-slider', 'value'),
     Input('language-dropdown', 'value'),
     Input('speaker-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('color-by', 'value'),
     Input('x-range-slider', 'value'),
     Input('y-range-slider', 'value'),
     Input('z-range-slider', 'value')]
)
def update_graph(filename, label, layer, epoch, n_samples, viz_type, timebin_range, 
                 language, speaker, gender, color_by, x_range, y_range, z_range):
    """
    Robuste Update-Funktion mit Metadaten-Farbcodierung.
    """
    if not filename:
        return go.Figure(), "⚠️ Keine Daten verfügbar"
    
    # Hilfsfunktion: Kontrastreiche Farbpalette
    def get_color_palette(color_variable, data_series):
        """Gibt kontrastreiche Farbpalette für kategorische oder kontinuierliche Daten."""
        categorical_vars = ['label', 'speaker', 'speaker_gender', 'language', 'word', 'digit']
        
        if color_variable in categorical_vars or not pd.api.types.is_numeric_dtype(data_series):
            # Diskrete, kontrastreiche Farben
            n_unique = data_series.nunique()
            if n_unique <= 10:
                return px.colors.qualitative.Plotly  # Sehr kontrastreich
            elif n_unique <= 24:
                return px.colors.qualitative.D3
            else:
                return px.colors.qualitative.Alphabet
        else:
            # Kontinuierliche Skala für numerische Daten
            return 'Viridis'
    
    # Daten laden (aus file_dict, das beide Verzeichnisse enthält)
    filepath = file_dict.get(filename)
    if filepath is None:
        return go.Figure(), f"⚠️ Datei nicht gefunden: {filename}"
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return go.Figure(), f"❌ Fehler beim Laden: {str(e)}"
    
    # Filter nach Label
    if label != 'all':
        df = df[df['label'] == label]
    
    # Filter nach Layer (falls Spalte vorhanden)
    if layer != 'all' and 'layer' in df.columns:
        df = df[df['layer'] == layer]
    
    # Filter nach Epoche (falls Spalte vorhanden) - NICHT bei Epochen-Animation
    if epoch != 0 and 'epoch' in df.columns and viz_type != 'epoch_animation':
        df = df[df['epoch'] == epoch]
    
    # Filter nach Zeitbin-Bereich
    if timebin_range and len(timebin_range) == 2:
        min_timebin, max_timebin = timebin_range
        df = df[(df['time_bin'] >= min_timebin) & (df['time_bin'] <= max_timebin)]
    
    # Metadaten-Filter
    if language != 'all' and 'language' in df.columns:
        df = df[df['language'] == language]
    
    if speaker != 'all' and 'speaker' in df.columns:
        df = df[df['speaker'] == speaker]
    
    if gender != 'all' and 'speaker_gender' in df.columns:
        df = df[df['speaker_gender'] == gender]
    
    # Bereichs-Filter für X, Y, Z Achsen (Zoom in bestimmte Regionen)
    if x_range and len(x_range) == 2:
        x_min, x_max = x_range
        df = df[(df['x'] >= x_min) & (df['x'] <= x_max)]
    
    if y_range and len(y_range) == 2:
        y_min, y_max = y_range
        df = df[(df['y'] >= y_min) & (df['y'] <= y_max)]
    
    if z_range and len(z_range) == 2:
        z_min, z_max = z_range
        df = df[(df['z'] >= z_min) & (df['z'] <= z_max)]
    
    if df.empty:
        filter_info = []
        if label != 'all':
            filter_info.append(f"Label={label}")
        if layer != 'all':
            filter_info.append(f"Layer={layer}")
        if epoch != 0 and viz_type != 'epoch_animation':
            filter_info.append(f"Epoche={epoch}")
        if timebin_range and len(timebin_range) == 2:
            filter_info.append(f"Zeitbins={timebin_range[0]}-{timebin_range[1]}")
        if language != 'all':
            filter_info.append(f"Sprache={language}")
        if speaker != 'all':
            filter_info.append(f"Speaker={speaker}")
        if gender != 'all':
            filter_info.append(f"Geschlecht={gender}")
        if x_range:
            filter_info.append(f"X={x_range[0]:.1f}-{x_range[1]:.1f}")
        if y_range:
            filter_info.append(f"Y={y_range[0]:.1f}-{y_range[1]:.1f}")
        if z_range:
            filter_info.append(f"Z={z_range[0]:.1f}-{z_range[1]:.1f}")
        return go.Figure(), f"⚠️ Keine Daten für Filter: {', '.join(filter_info)}"
    
    # Nur erste n_samples nehmen
    sample_ids = df['sample_id'].unique()[:n_samples]
    df = df[df['sample_id'].isin(sample_ids)]
    
    # Methode extrahieren
    method = df['method'].iloc[0]
    
    # Bestimme Farbvariable für alle Visualisierungen
    if color_by == 'black':
        color_var = 'black'
        color_palette = 'black'  # Spezialfall: monochrom
    elif color_by == 'time':
        color_var = 'time_bin'
        # Hole Farbpalette
        try:
            color_palette = get_color_palette(color_var, df[color_var])
        except:
            color_palette = 'Viridis'
            color_var = 'time_bin'
    else:
        color_var = color_by if color_by in df.columns else 'time_bin'
        # Hole Farbpalette
        try:
            color_palette = get_color_palette(color_var, df[color_var])
        except:
            color_palette = 'Viridis'
            color_var = 'time_bin'
    
    # Visualisierung erstellen
    if viz_type == 'lines':
        # Trajektorien mit Linien und Farbverlauf
        fig = go.Figure()
        
        # Bestimme ob diskrete oder kontinuierliche Färbung
        is_discrete = isinstance(color_palette, list)
        is_black = (color_palette == 'black')
        
        for idx, sample_id in enumerate(sample_ids):
            sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
            if sample_df.empty:
                continue
                
            label_val = sample_df['label'].iloc[0] if 'label' in sample_df.columns else 'N/A'
            
            # Erstelle Hover-Text mit Metadaten
            hover_text = []
            for _, row in sample_df.iterrows():
                text = f"<b>Sample {sample_id}</b><br>Zeit: {row['time_bin']}"
                if 'label' in sample_df.columns:
                    text += f"<br>Label: {row['label']}"
                if 'word' in sample_df.columns:
                    text += f"<br>Wort: {row['word']}"
                if 'language' in sample_df.columns:
                    text += f"<br>Sprache: {row['language']}"
                if 'speaker' in sample_df.columns:
                    text += f"<br>Speaker: {row['speaker']}"
                if 'speaker_gender' in sample_df.columns:
                    text += f"<br>Geschlecht: {row['speaker_gender']}"
                if 'trial' in sample_df.columns:
                    text += f"<br>Trial: {row['trial']}"
                text += f"<br>X: {row['x']:.2f}<br>Y: {row['y']:.2f}<br>Z: {row['z']:.2f}"
                hover_text.append(text)
            
            # Färbung basierend auf color_var
            if is_black:
                # Monochrom schwarz - komplett undurchsichtig, ohne Rand
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Sample {sample_id}',
                    line=dict(color='black', width=1.5),
                    marker=dict(
                        size=3,
                        color='black',
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text
                ))
            elif is_discrete:
                # Diskrete Farbe für ganze Trajektorie
                color_value = sample_df[color_var].iloc[0]
                unique_values = sorted(df[color_var].unique())
                color_idx = unique_values.index(color_value) if color_value in unique_values else 0
                line_color = color_palette[color_idx % len(color_palette)]
                
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Sample {sample_id} ({color_var}: {color_value})',
                    line=dict(color=line_color, width=2.5),
                    marker=dict(
                        size=3,
                        color=line_color,
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text
                ))
            else:
                # Kontinuierlicher Farbverlauf
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Sample {sample_id} (Label {label_val})',
                    line=dict(
                        color=sample_df[color_var].tolist(),
                        colorscale=color_palette,
                        width=2.5
                    ),
                    marker=dict(
                        size=3,
                        color=sample_df[color_var].tolist(),
                        colorscale=color_palette,
                        showscale=bool(idx == 0),  # Nur eine Colorbar
                        colorbar=dict(title=color_var, x=1.1),
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_text
                ))
        
        title = f'{method.upper()} - Trajektorien (Färbung: {"Schwarz (monochrom)" if color_var == "black" else color_var})'
    
    elif viz_type == 'lines_by_label':
        # Trajektorien mit Label-Farbcodierung
        fig = go.Figure()
        
        # Farben für verschiedene Labels
        label_colors = px.colors.qualitative.Set1
        label_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 
                        'circle-open', 'square-open', 'diamond-open']
        
        # Gruppiere Samples nach Labels
        labels_in_data = sorted(df['label'].unique())
        
        for label_idx, label_val in enumerate(labels_in_data):
            label_df = df[df['label'] == label_val]
            label_samples = [sid for sid in sample_ids if sid in label_df['sample_id'].unique()]
            
            if not label_samples:
                continue
                
            color = label_colors[label_idx % len(label_colors)]
            symbol = label_symbols[label_idx % len(label_symbols)]
            
            for sample_id in label_samples:
                sample_df = label_df[label_df['sample_id'] == sample_id].sort_values('time_bin')
                
                # Linie mit Label-Farbe
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Label {label_val} - Sample {sample_id}',
                    legendgroup=f'Label {label_val}',
                    showlegend=True if sample_id == label_samples[0] else False,
                    line=dict(
                        color=color,
                        width=2.5
                    ),
                    marker=dict(
                        size=3,
                        color=color,
                        symbol=symbol,
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate=f'<b>Label {label_val} - Sample {sample_id}</b><br>Zeit: %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
                    text=sample_df['time_bin'].tolist()
                ))
        
        title = f'{method.upper()} - Trajektorien nach Labels (farbcodiert)'
    
    elif viz_type == 'scatter':
        # Scatter Plot mit Metadaten-Farbcodierung
        fig = go.Figure()
        
        # Bestimme Farbvariable
        if color_by == 'black':
            color_var = 'black'
            color_palette = 'black'
        elif color_by == 'time':
            color_var = 'time_bin'
            color_palette = get_color_palette(color_var, df[color_var])
        else:
            color_var = color_by if color_by in df.columns else 'label'
            # Hole Farbpalette
            color_palette = get_color_palette(color_var, df[color_var])
        
        # Konvertiere kategorische zu String für diskrete Färbung
        if color_var in ['label', 'speaker', 'digit']:
            df = df.copy()
            df[color_var] = df[color_var].astype(str)
        
        # Hilfsfunktion zum Erstellen von Hover-Text
        def create_hover_text(row):
            """Erstellt Hover-Text für eine Datenzeile."""
            text = f"<b>Sample {row['sample_id']}</b><br>Zeit: {row['time_bin']}"
            if 'label' in df.columns:
                text += f"<br>Label: {row['label']}"
            if 'word' in df.columns:
                text += f"<br>Wort: {row['word']}"
            if 'language' in df.columns:
                text += f"<br>Sprache: {row['language']}"
            if 'speaker' in df.columns:
                text += f"<br>Speaker: {row['speaker']}"
            if 'speaker_gender' in df.columns:
                text += f"<br>Geschlecht: {row['speaker_gender']}"
            if 'trial' in df.columns:
                text += f"<br>Trial: {row['trial']}"
            if 'digit' in df.columns:
                text += f"<br>Digit: {row['digit']}"
            text += f"<br>X: {row['x']:.2f}<br>Y: {row['y']:.2f}<br>Z: {row['z']:.2f}"
            return text
        
        # Gruppiere nach Farbvariable für diskrete Färbung
        if color_palette == 'black':  # Monochrom schwarz
            # Erstelle Hover-Text für alle Daten
            hover_text = [create_hover_text(row) for _, row in df.iterrows()]
            
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=df['z'],
                mode='markers',
                name='Alle Punkte',
                marker=dict(
                    size=4,
                    color='black',
                    line=dict(width=0),  # Kein Rand
                    opacity=1.0  # Komplett undurchsichtig
                ),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            ))
        elif isinstance(color_palette, list):  # Diskrete Palette
            unique_values = sorted(df[color_var].unique())
            
            for idx, val in enumerate(unique_values):
                subset = df[df[color_var] == val].copy()
                
                # Erstelle Hover-Text für dieses Subset
                subset_hover = [create_hover_text(row) for _, row in subset.iterrows()]
                color = color_palette[idx % len(color_palette)]
                
                fig.add_trace(go.Scatter3d(
                    x=subset['x'],
                    y=subset['y'],
                    z=subset['z'],
                    mode='markers',
                    name=f'{color_var}: {val}',
                    marker=dict(
                        size=4,
                        color=color,
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=subset_hover
                ))
        else:  # Kontinuierliche Skala
            # Erstelle Hover-Text für alle Daten
            hover_text = [create_hover_text(row) for _, row in df.iterrows()]
            
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=df['z'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df[color_var],
                    colorscale=color_palette,
                    colorbar=dict(title=color_var),
                    line=dict(width=0),  # Kein Rand
                    opacity=1.0  # Komplett undurchsichtig
                ),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            ))
        
        title = f'{method.upper()} - Scatter Plot (Färbung: {"Schwarz (monochrom)" if color_var == "black" else color_var})'
    
    elif viz_type == 'animation':
        # Animation: Punkt läuft über komplette Linie mit Farbverlauf
        fig = go.Figure()
        
        all_time_bins = sorted(df['time_bin'].unique())
        
        # 1. Zeichne alle kompletten Linien mit Farbverlauf (statisch)
        for sample_id in sample_ids:
            sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
            label_val = sample_df['label'].iloc[0]
            
            # Linie mit Farbverlauf
            fig.add_trace(go.Scatter3d(
                x=sample_df['x'],
                y=sample_df['y'],
                z=sample_df['z'],
                mode='lines',
                line=dict(
                    color=sample_df['time_bin'].tolist(),  # Farbverlauf!
                    colorscale='Viridis',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip',
                name=f'Line_{sample_id}'
            ))
        
        # 2. Füge initiale Punkt-Trace hinzu
        initial_points_x, initial_points_y, initial_points_z = [], [], []
        initial_colors = []
        initial_hover = []
        
        for sample_id in sample_ids:
            sample_point = df[(df['sample_id'] == sample_id) & (df['time_bin'] == all_time_bins[0])]
            if not sample_point.empty:
                initial_points_x.append(sample_point['x'].iloc[0])
                initial_points_y.append(sample_point['y'].iloc[0])
                initial_points_z.append(sample_point['z'].iloc[0])
                label_val = sample_point['label'].iloc[0]
                initial_colors.append(label_val)
                initial_hover.append(f'Sample {sample_id}<br>Zeit: {all_time_bins[0]}<br>Label: {label_val}')
        
        fig.add_trace(go.Scatter3d(
            x=initial_points_x,
            y=initial_points_y,
            z=initial_points_z,
            mode='markers',
            marker=dict(
                size=8,
                color=initial_colors,
                colorscale='Turbo',
                line=dict(width=0),  # Kein Rand
                showscale=False,
                opacity=1.0  # Komplett undurchsichtig
            ),
            text=initial_hover,
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            name='Aktuelle Position'
        ))
        
        # 3. Erstelle Animations-Frames - Jeder Frame enthält ALLE Traces
        frames = []
        
        for time_bin in all_time_bins:
            frame_data = []
            
            # Füge alle Linien hinzu (gleich bleibend)
            for sample_id in sample_ids:
                sample_df = df[df['sample_id'] == sample_id].sort_values('time_bin')
                
                frame_data.append(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines',
                    line=dict(
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Füge die Punkte für diesen Zeitschritt hinzu
            points_x, points_y, points_z = [], [], []
            colors = []
            hover_texts = []
            
            for sample_id in sample_ids:
                sample_point = df[(df['sample_id'] == sample_id) & (df['time_bin'] == time_bin)]
                if not sample_point.empty:
                    points_x.append(sample_point['x'].iloc[0])
                    points_y.append(sample_point['y'].iloc[0])
                    points_z.append(sample_point['z'].iloc[0])
                    label_val = sample_point['label'].iloc[0]
                    colors.append(label_val)
                    hover_texts.append(f'Sample {sample_id}<br>Zeit: {time_bin}<br>Label: {label_val}')
            
            frame_data.append(go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale='Turbo',
                    line=dict(width=0),  # Kein Rand
                    showscale=False,
                    opacity=1.0  # Komplett undurchsichtig
                ),
                text=hover_texts,
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(time_bin)))
        
        fig.frames = frames
        
        # Animation-Einstellungen
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate', 
                                    'transition': {'duration': 0}}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 
                                      'mode': 'immediate'}]}
                ],
                'x': 0.1, 'y': 1.15
            }],
            sliders=[{
                'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 
                                              'mode': 'immediate', 'transition': {'duration': 0}}],
                          'label': f'Zeit: {f.name}', 'method': 'animate'} for f in frames],
                'active': 0,
                'y': -0.1,
                'len': 0.9,
                'x': 0.1
            }]
        )
        
        title = f'{method.upper()} - Animation: Punkt läuft über Trajektorie'
    
    elif viz_type == 'epoch_animation':
        # Epochen-Animation: Zeigt Entwicklung über Epochen
        if 'epoch' not in df.columns:
            return go.Figure(), "Epochen-Animation benötigt 'epoch' Spalte in den Daten"
        
        fig = go.Figure()
        
        all_epochs = sorted(df['epoch'].unique())
        if len(all_epochs) < 2:
            return go.Figure(), f"Epochen-Animation benötigt mindestens 2 Epochen. Gefunden: {all_epochs}"
        
        # Debug: Zeige verfügbare Epochen
        print(f"DEBUG: Verfügbare Epochen für Animation: {all_epochs}")
        print(f"DEBUG: Erste Epoche (initial_epoch): {min(all_epochs)}")
        print(f"DEBUG: Sample IDs: {sample_ids}")
        print(f"DEBUG: Labels in Daten: {sorted(df['label'].unique())}")
        
        # 1. Zeichne alle Trajektorien für alle Epochen (statisch, transparent)
        for epoch in all_epochs:
            epoch_df = df[df['epoch'] == epoch]
            for sample_id in sample_ids:
                sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                if not sample_df.empty:
                    label_val = sample_df['label'].iloc[0]
                    
                    # Transparente Linien für alle Epochen
                    fig.add_trace(go.Scatter3d(
                        x=sample_df['x'],
                        y=sample_df['y'],
                        z=sample_df['z'],
                        mode='lines',
                        line=dict(
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            width=1
                        ),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'Line_{sample_id}_epoch_{epoch}'
                    ))
        
        # 2. Füge initiale Trajektorien hinzu (erste Epoche, sichtbar)
        # Verwende die kleinste Epoche (normalerweise 0)
        initial_epoch = min(all_epochs)
        initial_epoch_df = df[df['epoch'] == initial_epoch]
        
        for idx, sample_id in enumerate(sample_ids):
            sample_df = initial_epoch_df[initial_epoch_df['sample_id'] == sample_id].sort_values('time_bin')
            if not sample_df.empty:
                label_val = sample_df['label'].iloc[0]
                
                fig.add_trace(go.Scatter3d(
                    x=sample_df['x'],
                    y=sample_df['y'],
                    z=sample_df['z'],
                    mode='lines+markers',
                    name=f'Sample {sample_id} (Label {label_val})',
                    line=dict(
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        width=3
                    ),
                    marker=dict(
                        size=4,
                        color=sample_df['time_bin'].tolist(),
                        colorscale='Viridis',
                        showscale=bool(idx == 0),
                        colorbar=dict(title="Zeit", x=1.1),
                        line=dict(width=0),  # Kein Rand
                        opacity=1.0  # Komplett undurchsichtig
                    ),
                    hovertemplate=f'<b>Sample {sample_id}</b><br>Epoche: {initial_epoch}<br>Zeit: %{{marker.color}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}',
                    visible=True
                ))
        
        # 3. Erstelle Animations-Frames für Epochen (in aufsteigender Reihenfolge)
        frames = []
        
        for epoch in sorted(all_epochs):
            frame_data = []
            
            # Füge alle transparenten Linien hinzu (gleich bleibend)
            for ep in all_epochs:
                epoch_df = df[df['epoch'] == ep]
                for sample_id in sample_ids:
                    sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                    if not sample_df.empty:
                        frame_data.append(go.Scatter3d(
                            x=sample_df['x'],
                            y=sample_df['y'],
                            z=sample_df['z'],
                            mode='lines',
                            line=dict(
                                color=sample_df['time_bin'].tolist(),
                                colorscale='Viridis',
                                width=1
                            ),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Füge die sichtbaren Trajektorien für diese Epoche hinzu
            epoch_df = df[df['epoch'] == epoch]
            for idx, sample_id in enumerate(sample_ids):
                sample_df = epoch_df[epoch_df['sample_id'] == sample_id].sort_values('time_bin')
                if not sample_df.empty:
                    label_val = sample_df['label'].iloc[0]
                    
                    frame_data.append(go.Scatter3d(
                        x=sample_df['x'],
                        y=sample_df['y'],
                        z=sample_df['z'],
                        mode='lines+markers',
                        line=dict(
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            width=3
                        ),
                        marker=dict(
                            size=4,
                            color=sample_df['time_bin'].tolist(),
                            colorscale='Viridis',
                            showscale=bool(idx == 0),
                            colorbar=dict(title="Zeit", x=1.1),
                            line=dict(width=0),  # Kein Rand
                            opacity=1.0  # Komplett undurchsichtig
                        ),
                        hovertemplate=f'<b>Sample {sample_id}</b><br>Epoche: {epoch}<br>Zeit: %{{marker.color}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}'
                    ))
            
            frames.append(go.Frame(data=frame_data, name=str(epoch)))
        
        fig.frames = frames
        
        # Animation-Einstellungen für Epochen
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'buttons': [
                    {'label': '▶ Play Epochen', 'method': 'animate', 
                     'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate', 
                                    'transition': {'duration': 0}}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 
                                      'mode': 'immediate'}]}
                ],
                'x': 0.1, 'y': 1.15
            }],
            sliders=[{
                'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 
                                              'mode': 'immediate', 'transition': {'duration': 0}}],
                          'label': f'Epoche: {f.name}', 'method': 'animate'} for f in frames],
                'active': 0,
                'y': -0.1,
                'len': 0.9,
                'x': 0.1
            }]
        )
        
        title = f'{method.upper()} - Animation: Entwicklung über Epochen'
    
    # Legende-Konfiguration: Übersichtlich über dem Plot
    legend_config = dict(
        orientation="h",  # Horizontal
        yanchor="bottom",
        y=1.02,  # Über dem Plot
        xanchor="left",
        x=0,
        bgcolor="rgba(255, 255, 255, 0.9)",  # Weißer Hintergrund mit Transparenz
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        font=dict(size=10),
        tracegroupgap=5
    )
    
    # Bestimme ob Legende angezeigt werden soll
    # Bei kontinuierlichen Farbskalen (mit Colorbar) keine Legende nötig
    show_legend = True
    if color_palette == 'black':
        # Bei monochromer Färbung keine Legende nötig
        show_legend = False
    elif viz_type in ['lines', 'animation', 'epoch_animation']:
        # Bei Trajektorien mit kontinuierlicher Farbskala keine Legende
        if not isinstance(color_palette, list):
            show_legend = False
    
    fig.update_layout(
        title=dict(
            text=title,
            y=0.98,  # Titel etwas höher
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title=f'{method} Dim 1',
            yaxis_title=f'{method} Dim 2',
            zaxis_title=f'{method} Dim 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=show_legend,
        legend=legend_config,
        margin=dict(t=120, b=50, l=50, r=50)  # Mehr Platz oben für Legende
    )
    
    # Statistiken
    stats_items = [
        html.P(f"📁 Datei: {filename}"),
        html.P(f"🔬 Methode: {method.upper()}"),
    ]
    
    # Füge Layer hinzu falls vorhanden
    if 'layer' in df.columns:
        layers = sorted(df['layer'].unique())
        stats_items.append(html.P(f"🧠 Layer: {', '.join(layers)}"))
    
    # Füge Epoche hinzu falls vorhanden
    if 'epoch' in df.columns:
        epochs = sorted(df['epoch'].unique())
        if viz_type == 'epoch_animation':
            stats_items.append(html.P(f"📅 Epochen für Animation: {epochs}"))
        elif len(epochs) == 1:
            stats_items.append(html.P(f"📅 Epoche: {epochs[0]}"))
        else:
            stats_items.append(html.P(f"📅 Epochen: {epochs}"))
    
    stats_items.extend([
        html.P(f"📊 Anzahl Samples: {len(sample_ids)}"),
        html.P(f"🏷️  Labels: {sorted(df['label'].unique())}"),
        html.P(f"📈 Datenpunkte gesamt: {len(df):,}"),
        html.P(f"⏱️  Zeitbins pro Sample: {df.groupby('sample_id')['time_bin'].count().iloc[0]}"),
    ])
    
    # Metadaten-Statistiken (falls vorhanden)
    metadata_cols = {
        'language': ('🌍', 'Sprachen'),
        'speaker': ('🗣️', 'Speakers'),
        'speaker_gender': ('👤', 'Geschlechter'),
        'word': ('💬', 'Wörter'),
        'trial': ('🔢', 'Trials'),
        'digit': ('🔟', 'Digits')
    }
    
    has_any_metadata = False
    for col, (icon, label) in metadata_cols.items():
        if col in df.columns:
            if not has_any_metadata:
                stats_items.append(html.Hr())
                stats_items.append(html.P("📋 Metadaten:", style={'fontWeight': 'bold', 'marginTop': '10px'}))
                has_any_metadata = True
            
            unique_values = sorted(df[col].unique())
            if len(unique_values) <= 10:
                stats_items.append(html.P(f"{icon} {label}: {unique_values}"))
            else:
                stats_items.append(html.P(f"{icon} {label}: {len(unique_values)} eindeutige Werte"))
    
    # Zusätzliche Metadaten-Details (falls vorhanden)
    if 'speaker_age' in df.columns:
        ages = df.groupby('sample_id')['speaker_age'].first()
        stats_items.append(html.P(f"🎂 Alter-Bereich: {ages.min()} - {ages.max()} Jahre"))
    
    if 'speaker_height' in df.columns:
        heights = df.groupby('sample_id')['speaker_height'].first()
        stats_items.append(html.P(f"📏 Größe-Bereich: {heights.min()} - {heights.max()} cm"))
    
    if 'original_filename' in df.columns:
        stats_items.append(html.P(f"📄 Original-Dateinamen: verfügbar", style={'fontStyle': 'italic'}))
    
    # Zeitbin-Information hinzufügen
    if timebin_range and len(timebin_range) == 2:
        min_timebin, max_timebin = timebin_range
        stats_items.append(html.P(f"🕒 Zeitbin-Bereich: {min_timebin} - {max_timebin}"))
    else:
        time_bins_in_data = sorted(df['time_bin'].unique())
        stats_items.append(html.P(f"🕒 Zeitbins in Daten: {time_bins_in_data[0]} - {time_bins_in_data[-1]}"))
    
    stats_items.extend([
        html.Hr(),
        html.P(f"📊 Daten in gefiltertem Bereich:", style={'fontWeight': 'bold'}),
        html.P(f"X-Bereich: [{df['x'].min():.2f}, {df['x'].max():.2f}]", style={'marginLeft': '10px'}),
        html.P(f"Y-Bereich: [{df['y'].min():.2f}, {df['y'].max():.2f}]", style={'marginLeft': '10px'}),
        html.P(f"Z-Bereich: [{df['z'].min():.2f}, {df['z'].max():.2f}]", style={'marginLeft': '10px'}),
    ])
    
    # Zeige Info wenn Bereichs-Filter aktiv ist
    if x_range and y_range and z_range:
        stats_items.append(html.Hr())
        stats_items.append(html.P("🔍 Aktiver Bereichs-Filter:", style={'fontWeight': 'bold', 'color': '#3498db'}))
        stats_items.append(html.P(f"X-Filter: [{x_range[0]:.2f}, {x_range[1]:.2f}]", style={'marginLeft': '10px', 'color': '#666'}))
        stats_items.append(html.P(f"Y-Filter: [{y_range[0]:.2f}, {y_range[1]:.2f}]", style={'marginLeft': '10px', 'color': '#666'}))
        stats_items.append(html.P(f"Z-Filter: [{z_range[0]:.2f}, {z_range[1]:.2f}]", style={'marginLeft': '10px', 'color': '#666'}))
    
    # Debug-Info für Epochen-Animation
    if viz_type == 'epoch_animation':
        stats_items.append(html.Hr())
        stats_items.append(html.P(f"🔍 DEBUG: Animation startet bei Epoche {min(all_epochs)}"))
        stats_items.append(html.P(f"🔍 DEBUG: Alle Epochen: {all_epochs}"))
    
    stats = html.Div(stats_items)
    
    return fig, stats

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starte Dash-App auf http://127.0.0.1:8050")
    print("="*80 + "\n")
    app.run(debug=True, port=8050)

