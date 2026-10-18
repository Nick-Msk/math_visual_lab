import threading
import time
import webbrowser
import os
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots
from scipy import signal
from scipy.integrate import solve_ivp

# =============================================================================
# Стили вкладок: визуальная иерархия уровней 2 и 3
# =============================================================================

# --- Уровень 1: главные группы (тёмные, крупные) ---
MAIN_TAB_STYLE = {
    'padding': '12px 24px',
    'fontSize': '16px',
    'fontWeight': '700',
    'color': '#455a64',
    'backgroundColor': '#cfd8dc',
    'border': 'none',
    'borderRadius': '8px 8px 0 0',
    'marginRight': '6px',
}
MAIN_TAB_SELECTED_STYLE = {
    'padding': '12px 24px',
    'fontSize': '16px',
    'fontWeight': '700',
    'color': 'white',
    'backgroundColor': '#263238',
    'border': 'none',
    'borderRadius': '8px 8px 0 0',
    'marginRight': '6px',
}
MAIN_TABS_CONTAINER_STYLE = {
    'borderBottom': '3px solid #263238',
}

# --- Уровень 2: подгруппы (заливка, крупные) ---
SUBGROUP_TAB_STYLE = {
    'padding': '10px 18px',
    'fontSize': '14px',
    'fontWeight': '600',
    'color': '#37474f',
    'backgroundColor': '#eceff1',
    'border': '1px solid #b0bec5',
    'borderBottom': 'none',
    'borderRadius': '6px 6px 0 0',
    'marginRight': '4px',
}
SUBGROUP_TAB_SELECTED_STYLE = {
    'padding': '10px 18px',
    'fontSize': '14px',
    'fontWeight': '600',
    'color': 'white',
    'backgroundColor': '#1f77b4',
    'border': '1px solid #1f77b4',
    'borderBottom': 'none',
    'borderRadius': '6px 6px 0 0',
    'marginRight': '4px',
}
SUBGROUP_TABS_CONTAINER_STYLE = {
    'borderBottom': '2px solid #1f77b4',
    'marginBottom': '6px',
}

# --- Уровень 3: темы (подчёркивание, лёгкие) ---
TOPIC_TAB_STYLE = {
    'padding': '7px 14px',
    'fontSize': '13px',
    'color': '#607d8b',
    'backgroundColor': 'transparent',
    'border': 'none',
    'borderBottom': '2px solid transparent',
}
TOPIC_TAB_SELECTED_STYLE = {
    'padding': '7px 14px',
    'fontSize': '13px',
    'fontWeight': '600',
    'color': '#d62728',
    'backgroundColor': 'transparent',
    'border': 'none',
    'borderBottom': '2px solid #d62728',
}
TOPIC_TABS_CONTAINER_STYLE = {
    'borderBottom': '1px solid #e0e0e0',
    'marginBottom': '10px',
}

app = Dash(__name__)

# =============================================================================
# МАКЕТ ПРИЛОЖЕНИЯ (Вкладки)
# =============================================================================
app.layout = html.Div([
     dcc.Tabs(style=MAIN_TABS_CONTAINER_STYLE, children=[

        # =====================================================================
        # ГРУППА 1: ВЕКТОРЫ И БАЗИСЫ
        # =====================================================================
        dcc.Tab(label='1. Векторы и базисы',
                style=MAIN_TAB_STYLE,
                selected_style=MAIN_TAB_SELECTED_STYLE,
                children=[
            dcc.Tabs(style=TOPIC_TABS_CONTAINER_STYLE, children=[

                # --- Комплексные числа ---
                dcc.Tab(label='Комплексные числа',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    # Вводный блок про три формы одного числа
                    html.Div(id='complex-intro', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '15px', 'fontSize': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderLeft': '4px solid #1f77b4',
                        'marginBottom': '10px'
                    }),
                    dcc.Graph(id='plot-complex', style={'height': '550px'}),
                    html.Div([
                        html.Label('|z1|'), dcc.Slider(id='m1', min=0.1, max=3, step=0.1, value=1.5),
                        html.Label('arg z1 (deg)'), dcc.Slider(id='a1', min=0, max=360, step=1, value=23),
                        html.Label('|z2|'), dcc.Slider(id='m2', min=0.1, max=3, step=0.1, value=1.2),
                        html.Label('arg z2 (deg)'), dcc.Slider(id='a2', min=0, max=360, step=1, value=63),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Сигналы и векторы ---
                dcc.Tab(label='Сигналы и векторы',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    html.Div(id='signal-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '16px', 'backgroundColor': '#f8f9fa'
                    }),
                    dcc.Graph(id='plot-signals', style={'height': '400px'}),
                    html.Div([
                        html.Label('Сдвиг сигнала g (roll)'),
                        dcc.Slider(id='signal-shift', min=0, max=7, step=1, value=1),
                        html.Label('Амплитуда шума h'),
                        dcc.Slider(id='noise-scale', min=0, max=2, step=0.1, value=1.0),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Смена базиса ---
                dcc.Tab(label='Смена базиса',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    html.Div(id='basis-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '16px', 'backgroundColor': '#f8f9fa'
                    }),
                    dcc.Graph(id='plot-basis', style={'height': '600px'}),
                    html.Div([
                        html.Label('Координата f_x'), dcc.Slider(id='fx', min=0, max=3, step=0.1, value=2.0),
                        html.Label('Координата f_y'), dcc.Slider(id='fy', min=0, max=3, step=0.1, value=1.0),
                        html.Label('Угол поворота базиса (deg)'), dcc.Slider(id='theta', min=0, max=360, step=1, value=30),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Ортогональность ---
                dcc.Tab(label='Ортогональность',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-gram', style={'height': '500px'}),
                    dcc.Graph(id='plot-basis-functions', style={'height': '350px'}),
                    dcc.Graph(id='plot-product', style={'height': '350px'}),
                    html.Div(id='pair-label', style={'textAlign': 'center', 'fontSize': '16px', 'padding': '10px'}),
                    html.Div([
                        html.Label('Число гармоник N'), dcc.Slider(id='n-harmonics', min=1, max=5, step=1, value=3),
                        html.Label('Нормировка √2'), dcc.RadioItems(id='normalize',
                            options=[{'label': 'Да', 'value': 'yes'}, {'label': 'Нет', 'value': 'no'}],
                            value='yes', inline=True),
                        html.Label('Интегрировать от -L до L, где L = (значение) × π'),
                        dcc.Slider(id='interval', min=0.2, max=1.0, step=0.1, value=1.0),
                        html.Label('Индекс первой функции i'), dcc.Slider(id='pair-i', min=0, max=10, step=1, value=1),
                        html.Label('Индекс второй функции j'), dcc.Slider(id='pair-j', min=0, max=10, step=1, value=2),
                    ], style={'padding': '20px', 'maxWidth': '900px', 'margin': '0 auto'}),
                ]),
                # --- Комплексные фокусы ---
                dcc.Tab(label='Комплексные фокусы',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-complex-tricks', style={'height': '500px'}),
                    html.Div(id='complex-tricks-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderLeft': '4px solid #d62728'
                    }),
                    html.Div([
                        html.Label('Подкоренное выражение для √(z) (отрицательное)'),
                        dcc.Slider(id='ct-sqrt-val', min=-25, max=-1, step=1, value=-9),
                        html.Label('Правая часть cos x = a (больше 1)'),
                        dcc.Slider(id='ct-cos-val', min=1.1, max=5.0, step=0.1, value=2.0),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),
            ])
        ]),
        # =====================================================================
        # ГРУППА 2: РЯДЫ И СПЕКТРЫ
        # =====================================================================
        dcc.Tab(label='2. Ряды и спектры',
                style=MAIN_TAB_STYLE,
                selected_style=MAIN_TAB_SELECTED_STYLE,
                children=[
            dcc.Tabs(style=TOPIC_TABS_CONTAINER_STYLE, children=[

                # --- Ряд Фурье (Гиббс) ---
                dcc.Tab(label='Ряд Фурье (Гиббс)',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-fourier', style={'height': '400px'}),
                    dcc.Graph(id='plot-coeffs', style={'height': '300px'}),
                    html.Div([
                        html.Label('Число гармоник N'),
                        dcc.Slider(id='fourier-n', min=1, max=50, step=1, value=7),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Комплексный спектр ---
                dcc.Tab(label='Комплексный спектр',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    html.Div(id='complex-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                    }),
                    dcc.Graph(id='plot-complex-plane', style={'height': '400px'}),
                    dcc.Graph(id='plot-spectrum', style={'height': '350px'}),
                    html.Div([
                        html.Label('Число гармоник N (от -N до +N)'),
                        dcc.Slider(id='complex-n', min=1, max=10, step=1, value=3),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- ДПФ на пальцах ---
                dcc.Tab(label='ДПФ на пальцах',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-dft-signal', style={'height': '300px'}),
                    dcc.Graph(id='plot-dft-spectrum', style={'height': '400px'}),
                    dcc.Graph(id='plot-dft-polar', style={'height': '500px'}),
                    html.Div(id='dft-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                    }),
                    html.Div([
                        html.Label('N (размер ДПФ)'), dcc.Slider(id='dft-n', min=8, max=32, step=1, value=16),
                        html.Label('Частота тона 1 (k₁)'), dcc.Slider(id='dft-k1', min=0, max=15, step=1, value=2),
                        html.Label('Частота тона 2 (k₂)'), dcc.Slider(id='dft-k2', min=0, max=15, step=1, value=5),
                        html.Label('Амплитуда тона 2'), dcc.Slider(id='dft-amp2', min=0, max=1, step=0.1, value=0.5),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Импульсы и sinc ---
                dcc.Tab(label='Импульсы и sinc',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-pulses', style={'height': '600px'}),
                    html.Div(id='pulse-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                    }),
                    html.Div([
                        html.Label('Скважность Q = T/τ'), dcc.Slider(id='pulse-q', min=1.5, max=10, step=0.5, value=4),
                        html.Label('Длительность импульса τ (сек)'), dcc.Slider(id='pulse-tau', min=0.05, max=0.5, step=0.05, value=0.2),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Непрерывный спектр ---
                dcc.Tab(label='Непрерывный спектр',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-continuous-limit', style={'height': '500px'}),
                    html.Div([
                        html.Label('Скважность Q = T/τ (растёт → спектр сгущается)'),
                        dcc.Slider(id='limit-q', min=2, max=64, step=2, value=4),
                        html.Label('Длительность импульса τ (сек)'),
                        dcc.Slider(id='limit-tau', min=0.05, max=0.5, step=0.05, value=0.2),
                        html.Label('Масштаб гармоник:'),
                        dcc.RadioItems(id='limit-scale',
                            options=[
                                {'label': 'Истинные C_n (падают как 1/Q)', 'value': 'true'},
                                {'label': 'Масштабированные C_n · T (ложатся на огибающую)', 'value': 'scaled'}
                            ], value='scaled', inline=False),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Одиночный импульс ---
                dcc.Tab(label='Одиночный импульс',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-single-pulse', style={'height': '650px'}),
                    html.Div([
                        html.Label('Длительность импульса τ (сек)'),
                        dcc.Slider(id='single-tau', min=0.1, max=2.0, step=0.1, value=0.5),
                        html.Label('Ширина окна наблюдения T_view (сек)'),
                        dcc.Slider(id='single-tview', min=2.0, max=16.0, step=1.0, value=4.0),
                        html.Label('Число точек ДПФ (M)'),
                        dcc.Slider(id='single-m', min=256, max=4096, step=256, value=2048),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Базис Котельникова ---
                dcc.Tab(label='Базис Котельникова',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-sinc-basis', style={'height': '350px'}),
                    dcc.Graph(id='plot-sinc-product', style={'height': '350px'}),
                    html.Div(id='sinc-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                    }),
                    html.Div([
                        html.Label('Индекс первой функции k₁'), dcc.Slider(id='sinc-k1', min=-3, max=3, step=1, value=0),
                        html.Label('Индекс второй функции k₂'), dcc.Slider(id='sinc-k2', min=-3, max=3, step=1, value=2),
                        html.Label('Предел интегрирования L (от -L до L)'), dcc.Slider(id='sinc-limit', min=1, max=20, step=1, value=15),
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),

                # --- Дельта-функция ---
                dcc.Tab(label='Дельта-функция',
                        style=TOPIC_TAB_STYLE,
                        selected_style=TOPIC_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Graph(id='plot-delta-time', style={'height': '350px'}),
                    dcc.Graph(id='plot-delta-spectrum', style={'height': '350px'}),
                    html.Div(id='delta-info', style={
                        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                        'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                    }),
                    html.Div([
                        html.Label('Ширина импульса ε (от 0.1 до 0.001)'),
                        dcc.Slider(id='delta-eps', min=-3, max=-1, step=0.1, value=-1.7),
                        html.Label('Функция f(t) для проверки фильтрующего свойства'),
                        dcc.RadioItems(
                            id='delta-probe-func',
                            options=[
                                {'label': 'f(t) = 1', 'value': 'const'},
                                {'label': 'f(t) = exp(-t²)', 'value': 'gauss'},
                                {'label': 'f(t) = cos(10t)', 'value': 'cos'},
                            ],
                            value='gauss', inline=True
                        )
                    ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                ]),
            ])
        ]),
        # =====================================================================
        # ГРУППА 3: СИСТЕМЫ И ЦЕПИ
        # =====================================================================
        dcc.Tab(label='3. Системы и цепи',
                    style=MAIN_TAB_STYLE,                          # ← добавить
                    selected_style=MAIN_TAB_SELECTED_STYLE,
                    children=[
            dcc.Tabs([

                # =============================================================
                # ПОДГРУППА 3.1: ЦЕПИ И ИМПЕДАНСЫ
                # =============================================================
                dcc.Tab(label='3.1 Цепи и импедансы',
                        style=SUBGROUP_TAB_STYLE,
                        selected_style=SUBGROUP_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Tabs(style=TOPIC_TABS_CONTAINER_STYLE, children=[

                        # --- Импеданс RLC ---
                        dcc.Tab(label='Импеданс RLC',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-rlc-impedance', style={'height': '400px'}),
                            html.Div(id='rlc-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Сопротивление R (Ом)'), dcc.Slider(id='rlc-r', min=1, max=100, step=1, value=10),
                                html.Label('Индуктивность L (мГн)'), dcc.Slider(id='rlc-l', min=0.1, max=10, step=0.1, value=1.0),
                                html.Label('Ёмкость C (мкФ)'), dcc.Slider(id='rlc-c', min=0.1, max=10, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- RC-ФНЧ (1-й порядок) ---
                        dcc.Tab(label='RC-ФНЧ (1-й порядок)',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-rc-bode', style={'height': '400px'}),
                            dcc.Graph(id='plot-rc-step', style={'height': '350px'}),
                            html.Div(id='rc-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('R (кОм)'),
                                dcc.Slider(id='rc-r', min=0.1, max=10, step=0.1, value=1.0),
                                html.Label('C (мкФ)'),
                                dcc.Slider(id='rc-c', min=0.1, max=10, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- RC-делитель (рампа) ---
                        dcc.Tab(label='RC-делитель (рампа)',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-rc-divider', style={'height': '400px'}),
                            dcc.Graph(id='plot-rc-divider-error', style={'height': '300px'}),
                            html.Div(id='rc-divider-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('R1 (кОм)'),
                                dcc.Slider(id='rcd-r1', min=1, max=10, step=0.5, value=3),
                                html.Label('R2 (кОм)'),
                                dcc.Slider(id='rcd-r2', min=0.5, max=5, step=0.5, value=1),
                                html.Label('C (мкФ)'),
                                dcc.Slider(id='rcd-c', min=0.05, max=1.0, step=0.05, value=0.1),
                                html.Label('Крутизна рампы k (В/с)'),
                                dcc.Slider(id='rcd-k', min=0.1, max=2.0, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),
                    ])
                ]),

                # =============================================================
                # ПОДГРУППА 3.2: ДИНАМИКА И ПОЛЮСА
                # =============================================================
                dcc.Tab(label='3.2 Динамика и полюса',
                        style=SUBGROUP_TAB_STYLE,
                        selected_style=SUBGROUP_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Tabs(style=TOPIC_TABS_CONTAINER_STYLE, children=[

                        # --- Механика и Электрика ---
                        dcc.Tab(label='Механика и Электрика',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-mech-elec', style={'height': '400px'}),
                            dcc.Graph(id='plot-phase-portrait', style={'height': '400px'}),
                            html.Div([
                                html.Label('m (масса) / L (индуктивность)'), dcc.Slider(id='me-m', min=0.5, max=5.0, step=0.5, value=1.0),
                                html.Label('b (трение) / R (сопротивление)'), dcc.Slider(id='me-b', min=0.1, max=5.0, step=0.1, value=0.4),
                                html.Label('k (жёсткость пружины) / 1/C (эластанс)'), dcc.Slider(id='me-k', min=5, max=100, step=5, value=25),
                                html.Label('F (толчок силы) / u (напряжение)'), dcc.Slider(id='me-f', min=1, max=10, step=1, value=3),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- H(s) и полюса ---
                        dcc.Tab(label='H(s) и полюса',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            html.Div(id='hs-formula', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '15px', 'fontSize': '16px', 'backgroundColor': '#f8f9fa',
                                'borderLeft': '4px solid #1f77b4'
                            }),
                            dcc.Graph(id='plot-pole-zero', style={'height': '380px'}),
                            dcc.Graph(id='plot-bode', style={'height': '420px'}),
                            dcc.Graph(id='plot-impulse', style={'height': '350px'}),
                            html.Div([
                                html.Label('Тип фильтра'),
                                dcc.RadioItems(id='hs-type',
                                    options=[
                                        {'label': 'ФНЧ (Low-pass)', 'value': 'low'},
                                        {'label': 'ФВЧ (High-pass)', 'value': 'high'},
                                        {'label': 'Полосовой', 'value': 'band'},
                                        {'label': 'Режекторный', 'value': 'notch'},
                                    ], value='low', inline=True),
                                html.Label('ωc (собственная частота)'), dcc.Slider(id='hs-wc', min=0.5, max=20, step=0.5, value=5),
                                html.Label('Q (добротность)'), dcc.Slider(id='hs-q', min=0.1, max=10, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '900px', 'margin': '0 auto'}),
                        ]),

                        # --- Разряд конденсатора ---
                        dcc.Tab(label='Разряд конденсатора',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-discharge', style={'height': '350px'}),
                            dcc.Graph(id='plot-discharge-poles', style={'height': '350px'}),
                            html.Div(id='discharge-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Сопротивление R (Ом)'), dcc.Slider(id='dis-r', min=0.1, max=8, step=0.1, value=0.5),
                                html.Label('Индуктивность L (Гн)'), dcc.Slider(id='dis-l', min=0.5, max=2, step=0.1, value=1.0),
                                html.Label('Ёмкость C (Ф)'), dcc.Slider(id='dis-c', min=0.5, max=2, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- Вычеты ---
                        dcc.Tab(label='Вычеты',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-residue-poles', style={'height': '350px'}),
                            dcc.Graph(id='plot-residue-signal', style={'height': '400px'}),
                            html.Div(id='residue-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Re(p₁)'), dcc.Slider(id='res-re1', min=-2, max=1, step=0.1, value=-0.5),
                                html.Label('Im(p₁)'), dcc.Slider(id='res-im1', min=0, max=5, step=0.5, value=0),
                                html.Label('Re(p₂)'), dcc.Slider(id='res-re2', min=-2, max=1, step=0.1, value=-1.5),
                                html.Label('Im(p₂)'), dcc.Slider(id='res-im2', min=0, max=5, step=0.5, value=0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- Дельта в H(s) ---
                        dcc.Tab(label='Дельта в H(s)',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-feedthrough', style={'height': '450px'}),
                            html.Div(id='feedthrough-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Параметр a (мс)'),
                                dcc.Slider(id='ft-a', min=0.1, max=5.0, step=0.1, value=1.0),
                                html.Label('Тип входного сигнала'),
                                dcc.RadioItems(
                                    id='ft-input',
                                    options=[
                                        {'label': 'Ступенька σ(t)', 'value': 'step'},
                                        {'label': 'Рампа k·t', 'value': 'ramp'},
                                    ],
                                    value='step', inline=True
                                ),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),
                    ])
                ]),

                # =============================================================
                # ПОДГРУППА 3.3: СВЁРТКА И ФИЛЬТРАЦИЯ
                # =============================================================
                dcc.Tab(label='3.3 Свёртка и фильтрация',
                        style=SUBGROUP_TAB_STYLE,
                        selected_style=SUBGROUP_TAB_SELECTED_STYLE,
                        children=[
                    dcc.Tabs(style=TOPIC_TABS_CONTAINER_STYLE, children=[

                        # --- Интеграл Дюамеля ---
                        dcc.Tab(label='Интеграл Дюамеля',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-duhamel', style={'height': '450px'}),
                            html.Div(id='duhamel-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Постоянная времени τ (мс)'),
                                dcc.Slider(id='duh-tau', min=0.1, max=5.0, step=0.1, value=1.0),
                                html.Label('Тип входного сигнала'),
                                dcc.RadioItems(
                                    id='duh-input',
                                    options=[
                                        {'label': 'Ступенька', 'value': 'step'},
                                        {'label': 'Прямоугольный импульс', 'value': 'pulse'},
                                        {'label': 'Синусоида', 'value': 'sine'},
                                    ],
                                    value='step', inline=True
                                ),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- Три метода ---
                        dcc.Tab(label='Три метода',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-three-methods', style={'height': '450px'}),
                            dcc.Graph(id='plot-three-methods-error', style={'height': '300px'}),
                            html.Div(id='three-methods-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Параметр a (мс)'),
                                dcc.Slider(id='tm-a', min=0.1, max=5.0, step=0.1, value=1.0),
                                html.Label('Крутизна рампы k (В/с)'),
                                dcc.Slider(id='tm-k', min=0.1, max=2.0, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- БПФ от h(t) ---
                        dcc.Tab(label='БПФ от h(t)',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-fft-impulse', style={'height': '450px'}),
                            html.Div(id='fft-impulse-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Параметр a (мс)'),
                                dcc.Slider(id='fft-imp-a', min=0.1, max=5.0, step=0.1, value=1.0),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- FFT Blur ---
                        dcc.Tab(label='FFT Blur',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-fft-blur', style={'height': '650px'}),
                            html.Div(id='blur-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Размер окна (нечетное число)'),
                                dcc.Slider(id='blur-window', min=1, max=15, step=2, value=3),
                                html.Label('Амплитуда шума'),
                                dcc.Slider(id='blur-noise', min=0, max=1.5, step=0.1, value=0.6),
                                html.Label('Период пилы'),
                                dcc.Slider(id='blur-period', min=5, max=40, step=1, value=20),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),

                        # --- Свёртка vs Спектр ---
                        dcc.Tab(label='Свёртка vs Спектр',
                                style=TOPIC_TAB_STYLE,
                                selected_style=TOPIC_TAB_SELECTED_STYLE,
                                children=[
                            dcc.Graph(id='plot-conv-spec', style={'height': '550px'}),
                            html.Div(id='conv-spec-info', style={
                                'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap',
                                'padding': '20px', 'fontSize': '15px', 'backgroundColor': '#f8f9fa'
                            }),
                            html.Div([
                                html.Label('Размер окна для нижнего графика (w)'),
                                dcc.Slider(id='conv-window', min=1, max=15, step=2, value=5),
                                html.Label('Амплитуда шума'),
                                dcc.Slider(id='conv-noise', min=0, max=1.5, step=0.1, value=0.6),
                                html.Label('Период пилы'),
                                dcc.Slider(id='conv-period', min=5, max=40, step=1, value=20),
                                html.Label('Обнуление краёв (как в исходной программе)'),
                                dcc.RadioItems(
                                    id='conv-zero-edges',
                                    options=[{'label': 'Да', 'value': 'yes'}, {'label': 'Нет', 'value': 'no'}],
                                    value='yes', inline=True
                                ),
                            ], style={'padding': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
                        ]),
                    ])
                ]),
            ])
        ]),
    ])
])

# =============================================================================
# CALLBACK 1: Комплексные числа
# =============================================================================
@app.callback(
    Output('plot-complex', 'figure'),
    Output('complex-intro', 'children'),
    Input('m1', 'value'), Input('a1', 'value'),
    Input('m2', 'value'), Input('a2', 'value'),
)
def update_complex(m1, a1, m2, a2):
    z1 = m1 * np.exp(1j * np.radians(a1))
    z2 = m2 * np.exp(1j * np.radians(a2))
    z12 = z1 * z2

    # --- Три лица одного числа z1 ---
    r1 = abs(z1)
    phi1 = np.angle(z1)  # в радианах, диапазон [-pi, pi]
    re1 = z1.real
    im1 = z1.imag

    # Проверка формулы Эйлера
    euler_err = abs(np.exp(1j * phi1) - (np.cos(phi1) + 1j * np.sin(phi1)))

    # Форматирование
    sign = '+' if im1 >= 0 else '-'
    trig_sign = '+' if np.sin(phi1) >= 0 else '-'

    intro_text = (
        f"Три лица одного числа z₁:\n\n"
        f"  1. Алгебраическая:      z₁ = {re1:+.3f} {sign} {abs(im1):.3f}j\n"
        f"  2. Тригонометрическая:  z₁ = {r1:.3f}·(cos({np.degrees(phi1):+.1f}°) "
        f"{trig_sign} j·sin({np.degrees(phi1):+.1f}°))\n"
        f"  3. Экспоненциальная:    z₁ = {r1:.3f}·e^(j·{phi1:+.4f} рад)\n\n"
        f"Связи:\n"
        f"  |z₁| = √(Re² + Im²) = {r1:.4f}\n"
        f"  φ₁   = atan2(Im, Re) = {phi1:+.4f} рад = {np.degrees(phi1):+.2f}°\n"
        f"  Re   = |z₁|·cos(φ₁) = {r1 * np.cos(phi1):+.4f}\n"
        f"  Im   = |z₁|·sin(φ₁) = {r1 * np.sin(phi1):+.4f}\n\n"
        f"Формула Эйлера e^(jφ) = cos φ + j·sin φ:\n"
        f"  |e^(jφ₁) − (cos φ₁ + j·sin φ₁)| = {euler_err:.2e}  (машинный ноль — формула верна)\n\n"
        f"Произведение z₁·z₂ = {r1*m2:.3f}·e^(j·({phi1 + np.radians(a2):+.4f}))\n"
        f"  |z₁·z₂| = |z₁|·|z₂| = {r1:.3f}·{m2:.3f} = {r1*m2:.3f}\n"
        f"  arg(z₁·z₂) = arg(z₁) + arg(z₂) = {np.degrees(phi1):+.1f}° + {a2:.1f}° = "
        f"{np.degrees(phi1) + a2:.1f}°"
    )

    # --- Построение графика (без изменений) ---
    fig = go.Figure()
    for z, name, color in [(z1, 'z₁', '#1f77b4'),
                           (z2, 'z₂', '#2ca02c'),
                           (z12, 'z₁·z₂', '#d62728')]:
        fig.add_trace(go.Scatter(
            x=[0, z.real], y=[0, z.imag],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=12),
            name=f'{name}: |·|={abs(z):.2f}, arg={np.degrees(np.angle(z)):.0f}°',
        ))

    fig.update_layout(
        xaxis=dict(range=[-1, 4], title='Re', zeroline=True, zerolinewidth=1),
        yaxis=dict(range=[-1, 4], title='Im', zeroline=True, zerolinewidth=1, scaleanchor='x'),
        title='Умножение: поворот + растяжение',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, intro_text

# =============================================================================
# CALLBACK: 1.1  Комплексные фокусы
# =============================================================================
@app.callback(
    Output('plot-complex-tricks', 'figure'),
    Output('complex-tricks-info', 'children'),
    Input('ct-sqrt-val', 'value'),
    Input('ct-cos-val', 'value'),
)
def update_complex_tricks(sqrt_val, cos_val):
    # --- 1. Корни из отрицательного числа ---
    n = abs(sqrt_val)
    r = np.sqrt(n)
    # √(−n) = √n · e^(j·(π/2 + πk)), k = 0, 1
    roots = [r * np.exp(1j * (np.pi/2 + np.pi*k)) for k in range(2)]

    # --- 2. cos x = a > 1 ---
    # Из e^{jx} + e^{-jx} = 2a  следует  y² − 2a·y + 1 = 0, y = e^{jx}
    # Корни: y = a ± √(a² − 1)
    y1 = cos_val + np.sqrt(cos_val**2 - 1)
    y2 = cos_val - np.sqrt(cos_val**2 - 1)
    # x = −j·ln(y)
    x1 = -1j * np.log(y1)
    x2 = -1j * np.log(y2)

    # Проверка
    check1 = np.cos(x1)
    check2 = np.cos(x2)

    # --- Сборка двух графиков (корни + решения cos) ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Корни из {sqrt_val:.0f}: z² = {sqrt_val:.0f}',
            f'Решения cos x = {cos_val:.1f} (в комплексной плоскости)'
        ),
    )

    # --- Левый график: корни ---
    fig.add_hline(y=0, line_color='black', line_width=1, row=1, col=1)
    fig.add_vline(x=0, line_color='black', line_width=1, row=1, col=1)

    # Окружность радиуса √n (все корни лежат на ней)
    theta = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(
        x=r*np.cos(theta), y=r*np.sin(theta), mode='lines',
        line=dict(color='lightgray', width=1, dash='dot'),
        name=f'|z| = {r:.2f}', showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    for k, zk in enumerate(roots):
        fig.add_trace(go.Scatter(
            x=[zk.real], y=[zk.imag], mode='markers+text',
            marker=dict(size=16, color=['#1f77b4', '#d62728'][k]),
            text=[f'z_{k}'], textposition='top center',
            name=f'z_{k} = {zk.real:+.1f}{zk.imag:+.1f}j'
        ), row=1, col=1)
        # Пунктир от нуля к корню
        fig.add_trace(go.Scatter(
            x=[0, zk.real], y=[0, zk.imag], mode='lines',
            line=dict(color=['#1f77b4', '#d62728'][k], width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

    fig.update_xaxes(title_text='Re(z)', row=1, col=1,
                     range=[-r*1.3, r*1.3], zeroline=False)
    fig.update_yaxes(title_text='Im(z)', row=1, col=1,
                     range=[-r*1.3, r*1.3], zeroline=False,
                     scaleanchor='x', scaleratio=1)

    # --- Правый график: cos x = a ---
    fig.add_hline(y=0, line_color='black', line_width=1, row=1, col=2)
    fig.add_vline(x=0, line_color='black', line_width=1, row=1, col=2)

    # Решения x1, x2 на комплексной плоскости
    solutions = [(x1, 'x₁', '#2ca02c'), (x2, 'x₂', '#9467bd')]
    for xk, name, color in solutions:
        fig.add_trace(go.Scatter(
            x=[xk.real], y=[xk.imag], mode='markers+text',
            marker=dict(size=16, color=color),
            text=[name], textposition='top center',
            name=f'{name} = {xk.real:+.3f}{xk.imag:+.3f}j'
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[0, xk.real], y=[0, xk.imag], mode='lines',
            line=dict(color=color, width=1.5, dash='dash'),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=2)

    # Автоматический масштаб правого графика
    max_abs = max(abs(x1), abs(x2)) * 1.3
    fig.update_xaxes(title_text='Re(x)', row=1, col=2,
                     range=[-max_abs, max_abs], zeroline=False)
    fig.update_yaxes(title_text='Im(x)', row=1, col=2,
                     range=[-max_abs, max_abs], zeroline=False,
                     scaleanchor='x', scaleratio=1)

    fig.update_layout(
        template='plotly_white',
        height=500,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # --- Текстовый блок ---
    info = (
        f"ФОКУС 1: корни из {sqrt_val:.0f}\n\n"
        f"  {sqrt_val:.0f} = {n:.0f}·e^(j(π + 2πk))\n"
        f"  z_k = √{n:.0f}·e^(j(π/2 + πk)) = {r:.2f}·e^(j·(π/2 + πk))\n\n"
    )
    for k, zk in enumerate(roots):
        info += f"  z_{k} = {zk.real:+.4f} {zk.imag:+.4f}j   (z_{k}² = {zk**2:.2f})\n"

    info += (
        f"\n  Оба корня лежат на окружности радиуса |z| = {r:.2f},\n"
        f"  разнесены на угол π (на 180°).\n\n"
        f"ФОКУС 2: cos x = {cos_val:.1f}\n\n"
        f"  Из e^(jx) + e^(−jx) = 2·cos(x) = 2·{cos_val:.1f}\n"
        f"  Обозначим y = e^(jx):  y² − 2·{cos_val:.1f}·y + 1 = 0\n"
        f"  Корни:  y₁ = {y1:.4f},  y₂ = {y2:.4f}\n"
        f"  Тогда x = −j·ln(y):\n"
        f"    x₁ = {x1.real:+.4f} {x1.imag:+.4f}j\n"
        f"    x₂ = {x2.real:+.4f} {x2.imag:+.4f}j\n\n"
        f"  Проверка: cos(x₁) = {check1:.6f},  cos(x₂) = {check2:.6f}\n"
        f"  (оба равны {cos_val:.1f}, но x — комплексные числа!)\n\n"
        f"  На графике видно: x₁ и x₂ отличаются только знаком Im(x),\n"
        f"  то есть x₂ = conj(x₁). Это потому, что cos(conj(x)) = conj(cos(x))\n"
        f"  = cos(x) для вещественного cos(x).\n\n"
        f"  ВЫВОД: когда вещественных решений нет, они находятся в комплексной плоскости.\n"
        f"  Именно так возникли комплексные числа: попытка решить уравнение x² = −1."
    )

    return fig, info

# =============================================================================
# CALLBACK 2: Сигналы и векторы
# =============================================================================
@app.callback(
    Output('plot-signals', 'figure'),
    Output('signal-info', 'children'),
    Input('signal-shift', 'value'),
    Input('noise-scale', 'value'),
)
def update_signals(shift, noise_scale):
    f = np.array([1, 2, 1, 0, -1, -2, -1, 0], float)
    g = np.roll(f, shift)

    # Используем seed для воспроизводимости, но меняем при сдвиге шума
    rng = np.random.default_rng(0)
    h = rng.normal(scale=noise_scale, size=8)

    def vec_info(a, b, name):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            r, ang, dot = 0, 0, 0
        else:
            dot = a @ b
            r = dot / (na * nb)
            ang = np.degrees(np.arccos(np.clip(r, -1, 1)))
        return f"{name}: |f|={na:5.2f}  (f,g)={dot:7.2f}  r={r:+.3f}  угол={ang:5.1f}°"

    info_text = (
        vec_info(f, g, 'f и g (похожие) ') + "\n" +
        vec_info(f, h, 'f и h (шум)      ')
    )

    # Создаем два графика рядом
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Похожие сигналы: r близок к 1", "Сигнал и шум: r близок к 0"))

    x_vals = list(range(8))

    # Левый график (f и g)
    fig.add_trace(go.Scatter(x=x_vals, y=f, mode='lines+markers', name='f',
                             line=dict(color='#1f77b4', width=2), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=g, mode='lines+markers', name=f'g (сдвиг {shift})',
                             line=dict(color='#2ca02c', width=2, dash='dash'), marker=dict(size=8)), row=1, col=1)

    # Правый график (f и h)
    fig.add_trace(go.Scatter(x=x_vals, y=f, mode='lines+markers', name='f',
                             line=dict(color='#1f77b4', width=2), marker=dict(size=8), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=h, mode='lines+markers', name='h (шум)',
                             line=dict(color='#d62728', width=2, dash='dash'), marker=dict(size=8)), row=1, col=2)

    fig.update_layout(template='plotly_white', margin=dict(t=50, b=20, l=20, r=20))
    fig.update_xaxes(title_text="Отсчет", row=1, col=1)
    fig.update_xaxes(title_text="Отсчет", row=1, col=2)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=1)

    return fig, info_text

# =============================================================================
# CALLBACK 3: Смена базиса
# =============================================================================
@app.callback(
    Output('plot-basis', 'figure'),
    Output('basis-info', 'children'),
    Input('fx', 'value'),
    Input('fy', 'value'),
    Input('theta', 'value'),
)
def update_basis(fx, fy, theta_deg):
    teta = np.radians(theta_deg)

    # Вектор
    f = np.array([fx, fy])

    # Стандартный базис
    v1, v2 = np.array([1., 0.]), np.array([0., 1.])

    # Повернутый базис
    u1 = np.array([np.cos(teta), np.sin(teta)])
    u2 = np.array([-np.sin(teta), np.cos(teta)])

    # Координаты
    C = [f @ v1, f @ v2]
    D = [f @ u1, f @ u2]

    # Текст с метриками
    info_text = (
        f"Вектор f: [{f[0]:.1f}, {f[1]:.1f}]\n"
        f"Координаты в базисе (v1, v2): [{C[0]:.2f}, {C[1]:.2f}]\n"
        f"Координаты в базисе (u1, u2): [{D[0]:.2f}, {D[1]:.2f}]\n"
        f"Сборка обратно: [{D[0]*u1[0] + D[1]*u2[0]:.10f}, {D[0]*u1[1] + D[1]*u2[1]:.10f}] = f\n"
        f"Длина вектора: ||f|| = {np.linalg.norm(f):.4f} (не зависит от базиса)"
    )

    fig = go.Figure()

    # 1. Рисуем стандартный базис (v1, v2) — пунктиром
    fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines',
                             line=dict(color='#2ca02c', width=2, dash='dot'), name='v1'))
    fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines',
                             line=dict(color='#2ca02c', width=2, dash='dot'), name='v2', showlegend=False))

    # 2. Рисуем повернутый базис (u1, u2)
    fig.add_trace(go.Scatter(x=[0, u1[0]], y=[0, u1[1]], mode='lines',
                             line=dict(color='#ff7f0e', width=2), name='u1'))
    fig.add_trace(go.Scatter(x=[0, u2[0]], y=[0, u2[1]], mode='lines',
                             line=dict(color='#ff7f0e', width=2), name='u2', showlegend=False))

    # 3. Линии проекций на повернутый базис
    # Точка на u1 и линия от f до этой точки
    p1 = D[0] * u1
    p2 = D[1] * u2
    fig.add_trace(go.Scatter(x=[f[0], p1[0]], y=[f[1], p1[1]], mode='lines',
                             line=dict(color='gray', width=1, dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=[f[0], p2[0]], y=[f[1], p2[1]], mode='lines',
                             line=dict(color='gray', width=1, dash='dash'), showlegend=False))

    # Точки на осях
    fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='markers',
                             marker=dict(color='gray', size=6), showlegend=False))

    # 4. Рисуем сам вектор f
    fig.add_trace(go.Scatter(x=[0, f[0]], y=[0, f[1]], mode='lines+markers',
                             line=dict(color='#1f77b4', width=4),
                             marker=dict(size=10), name='f (вектор)'))

    fig.update_layout(
        xaxis=dict(range=[-1, 4], title='X', zeroline=True, zerolinewidth=1),
        yaxis=dict(range=[-1, 4], title='Y', zeroline=True, zerolinewidth=1, scaleanchor='x'),
        title='Вектор f и его координаты в разных базисах',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig, info_text

# =============================================================================
# CALLBACK 4: Ортогональность
# =============================================================================
@app.callback(
    Output('plot-gram', 'figure'),
    Output('plot-basis-functions', 'figure'),
    Output('plot-product', 'figure'),
    Output('pair-label', 'children'),
    Input('n-harmonics', 'value'),
    Input('normalize', 'value'),
    Input('interval', 'value'),
    Input('pair-i', 'value'),
    Input('pair-j', 'value'),
)
def update_orthogonality(n_harm, normalize_val, interval_mult, pair_i, pair_j):
    L = interval_mult * np.pi
    t = np.linspace(-np.pi, np.pi, 4001)

    # Формируем базис
    scale = np.sqrt(2) if normalize_val == 'yes' else 1.0
    prefix = '√2·' if normalize_val == 'yes' else ''
    basis = [np.ones_like(t)]
    names = ['1']
    for k in range(1, n_harm + 1):
        basis.append(scale * np.cos(k * t))
        basis.append(scale * np.sin(k * t))
        names.append(f'{prefix}cos({k}t)')
        names.append(f'{prefix}sin({k}t)')

    n = len(basis)
    mask = (t >= -L) & (t <= L)

    # Матрица Грама
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = np.trapezoid(basis[i][mask] * basis[j][mask], t[mask]) / (2 * np.pi)

    # График 1: Heatmap матрицы Грама
    fig_gram = go.Figure(data=go.Heatmap(
        z=G, x=names, y=names,
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        text=np.round(G, 3), texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='⟨fᵢ,fⱼ⟩'),
    ))
    fig_gram.update_layout(
        title=f'Матрица Грама (интеграл от -{interval_mult}π до {interval_mult}π)',
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed'),
        template='plotly_white', height=500,
    )

    # График 2: Базисные функции
    fig_basis = go.Figure()
    for b, name in zip(basis, names):
        fig_basis.add_trace(go.Scatter(x=t, y=b, mode='lines', name=name, line=dict(width=2)))
    fig_basis.add_vline(x=L, line_dash='dash', line_color='gray')
    fig_basis.add_vline(x=-L, line_dash='dash', line_color='gray')
    fig_basis.update_layout(
        title='Базисные функции (серые — границы интегрирования)',
        xaxis=dict(title='t'), yaxis=dict(title='f(t)'),
        template='plotly_white', height=350,
    )

    # График 3: Произведение выбранных функций
    pi_ = min(pair_i, n - 1)
    pj_ = min(pair_j, n - 1)
    prod = np.where(mask, basis[pi_] * basis[pj_], 0)

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(
        x=t, y=prod, mode='lines', fill='tozeroy',
        line=dict(color='#d62728', width=2),
        name=f'{names[pi_]} · {names[pj_]}'
    ))
    fig_prod.update_layout(
        title=f'{names[pi_]} · {names[pj_]}, интеграл/(2π) = {G[pi_, pj_]:.4f}',
        xaxis=dict(title='t'), yaxis=dict(title='fᵢ(t) · fⱼ(t)'),
        template='plotly_white', height=350,
    )

    label = f'Выбрано: {names[pi_]} и {names[pj_]} → ⟨fᵢ, fⱼ⟩ = {G[pi_, pj_]:.6f}'
    return fig_gram, fig_basis, fig_prod, label

# =============================================================================
# CALLBACK 5: Ряд Фурье (Меандр и Гиббс)
# =============================================================================
@app.callback(
    Output('plot-fourier', 'figure'),
    Output('plot-coeffs', 'figure'),
    Input('fourier-n', 'value'),
)
def update_fourier(n_val):
    t = np.linspace(-np.pi, np.pi, 4001)

    def sp2pi(f_, g_):
        return np.trapezoid(f_ * g_, t) / (2 * np.pi)

    sq = np.sign(np.sin(t))  # меандр ±1

    # Коэффициенты (считаем один раз)
    C_cos = {k: sp2pi(sq, np.sqrt(2) * np.cos(k * t)) for k in range(n_val + 1)}
    C_sin = {k: sp2pi(sq, np.sqrt(2) * np.sin(k * t)) for k in range(1, n_val + 1)}

    # Частичная сумма
    s = C_cos[0] * np.ones_like(t)
    for k in range(1, n_val + 1):
        s += C_cos.get(k, 0) * np.sqrt(2) * np.cos(k * t) + \
             C_sin.get(k, 0) * np.sqrt(2) * np.sin(k * t)

    # График 1: Меандр и аппроксимация
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=sq, mode='lines', name='меандр',
                                  line=dict(color='lightgray', width=2)))
    fig_wave.add_trace(go.Scatter(x=t, y=s, mode='lines', name=f'Сумма N={n_val}',
                                  line=dict(color='#d62728', width=2)))

    # Найдём максимальное отклонение (для визуализации Гиббса)
    max_val = np.max(s)
    fig_wave.add_hline(y=max_val, line_dash="dot", line_color="gray",
                       annotation_text=f"Макс. выброс: {max_val:.4f}")

    fig_wave.update_layout(
        title=f'Аппроксимация меандра (N = {n_val}). Ушки Гиббса!',
        xaxis=dict(title='t'), yaxis=dict(title='f(t)'),
        template='plotly_white', margin=dict(l=20, r=20, t=40, b=20)
    )

    # График 2: Коэффициенты (Столбики)
    # В ряду Фурье для меандра ненулевые только синусы с нечетными k
    k_vals = []
    b_vals = []
    for k in range(1, n_val + 1):
        if abs(C_sin[k]) > 1e-6:
            k_vals.append(k)
            b_vals.append(C_sin[k])

    fig_coeffs = go.Figure(data=[
        go.Bar(x=k_vals, y=b_vals, marker_color='#1f77b4', name='βk (при sin)')
    ])
    fig_coeffs.update_layout(
        title='Коэффициенты βk (убывают как 1/k)',
        xaxis=dict(title='Номер гармоники k', tickmode='linear'),
        yaxis=dict(title='Амплитуда'),
        template='plotly_white', height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig_wave, fig_coeffs

# =============================================================================
# CALLBACK 6: Комплексный спектр
# =============================================================================
@app.callback(
    Output('plot-complex-plane', 'figure'),
    Output('plot-spectrum', 'figure'),
    Output('complex-info', 'children'),
    Input('complex-n', 'value'),
)
def update_complex_spectrum(n_max):
    t = np.linspace(-np.pi, np.pi, 4001)
    sq = np.sign(np.sin(t))

    def Ck(k):
        return np.trapezoid(sq * np.exp(-1j * k * t), t) / (2 * np.pi)

    # Считаем коэффициенты для k от -N до N
    k_vals = list(range(-n_max, n_max + 1))
    c_vals = [Ck(k) for k in k_vals]

    # --- График 1: Комплексная плоскость ---
    fig_plane = go.Figure()

    # Рисуем оси
    fig_plane.add_hline(y=0, line_color='black', line_width=1)
    fig_plane.add_vline(x=0, line_color='black', line_width=1)

    # Отображаем точки и векторы
    for k, c in zip(k_vals, c_vals):
        if abs(c) < 1e-6:
            continue # пропускаем нулевые

        color = '#d62728' if k > 0 else ('#1f77b4' if k < 0 else 'gray')

        # Стрелка от начала координат до C_k
        fig_plane.add_trace(go.Scatter(
            x=[0, c.real], y=[0, c.imag],
            mode='lines+markers+text',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            text=[None, f'C_{k}'], textposition="top center",
            name=f'k={k}',
            showlegend=False
        ))

    fig_plane.update_layout(
        title='Комплексные коэффициенты C_k (синие: k<0, красные: k>0)',
        xaxis=dict(title='Re(C_k)', range=[-0.6, 0.6], zeroline=False),
        yaxis=dict(title='Im(C_k)', range=[-0.6, 0.6], zeroline=False, scaleanchor='x'),
        template='plotly_white'
    )

    # --- График 2: Амплитудный и фазовый спектр ---
    magnitudes = [abs(c) for c in c_vals]
    phases = [np.degrees(np.angle(c)) for c in c_vals]

    fig_spec = make_subplots(rows=1, cols=2,
                             subplot_titles=("Амплитудный спектр |C_k|", "Фазовый спектр ∠C_k (град)"))

    # Магнитуда
    fig_spec.add_trace(go.Bar(
        x=k_vals, y=magnitudes, marker_color='#2ca02c', name='|C_k|'
    ), row=1, col=1)

    # Фаза
    fig_spec.add_trace(go.Bar(
        x=k_vals, y=phases, marker_color='#9467bd', name='∠C_k'
    ), row=1, col=2)

    fig_spec.update_layout(template='plotly_white', showlegend=False)
    fig_spec.update_xaxes(title_text="k", tickmode='linear', dtick=1)
    fig_spec.update_yaxes(title_text="|C_k|", row=1, col=1)
    fig_spec.update_yaxes(title_text="Градусы", row=1, col=2)

    # --- Текстовая информация ---
    c1 = Ck(1)
    c_1 = Ck(-1)
    beta1 = float(np.trapezoid(sq * np.sqrt(2) * np.sin(1 * t), t) / (2 * np.pi))

    info_text = (
        f"C[+1] = {c1.real:+.4f} + {c1.imag:+.4f}j\n"
        f"C[-1] = {c_1.real:+.4f} + {c_1.imag:+.4f}j  (сопряжён C[+1])\n"
        f"Связь с вещественным рядом: β1 = {beta1:+.4f} = -√2 · Im(C[+1]) = {-np.sqrt(2)*c1.imag:+.4f}"
    )

    return fig_plane, fig_spec, info_text

# =============================================================================
# CALLBACK 7: ДПФ на пальцах
# =============================================================================
@app.callback(
    Output('plot-dft-signal', 'figure'),
    Output('plot-dft-spectrum', 'figure'),
    Output('plot-dft-polar', 'figure'),
    Output('dft-info', 'children'),
    Input('dft-n', 'value'),
    Input('dft-k1', 'value'),
    Input('dft-k2', 'value'),
    Input('dft-amp2', 'value'),
)
def update_dft(N, k1, k2, amp2):
    # Приводим частоты по модулю N (алиасинг!)
    k1 = k1 % N
    k2 = k2 % N

    n = np.arange(N)
    f = np.cos(2 * np.pi * k1 * n / N) + amp2 * np.cos(2 * np.pi * k2 * n / N)

    # Матрица ДПФ
    W = np.exp(2j * np.pi * np.outer(np.arange(N), n) / N)
    C = (W.conj() @ f) / N
    f_rec = (W.T @ C).real

    # --- 1. Сигнал во времени ---
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=n, y=f, mode='lines+markers',
                                  name='f[n]',
                                  line=dict(color='#1f77b4', width=2),
                                  marker=dict(size=8)))
    fig_sig.add_trace(go.Scatter(x=n, y=f_rec, mode='lines',
                                  name='IFFT(FFT(f))',
                                  line=dict(color='#d62728', width=2, dash='dash')))
    fig_sig.update_layout(
        title=f'Сигнал f[n] = cos(2π·{k1}·n/{N}) + {amp2:.1f}·cos(2π·{k2}·n/{N})',
        xaxis_title='n', yaxis_title='f[n]',
        template='plotly_white', height=300,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- 2. Спектр: стандартный и центрированный ---
    mags = np.abs(C)
    fig_spec = make_subplots(rows=1, cols=2,
                              subplot_titles=('Стандарт: k = 0..N-1 (как у FFT)',
                                             'Centered: k = -N/2..N/2-1 (как у людей)'))

    colors = ['#d62728' if m > 1e-6 else '#cccccc' for m in mags]
    fig_spec.add_trace(go.Bar(x=n, y=mags, marker_color=colors), row=1, col=1)

    k_cent = np.fft.fftshift(n)
    C_cent = np.fft.fftshift(C)
    mags_cent = np.abs(C_cent)
    colors_cent = ['#d62728' if m > 1e-6 else '#cccccc' for m in mags_cent]
    fig_spec.add_trace(go.Bar(x=k_cent, y=mags_cent, marker_color=colors_cent),
                       row=1, col=2)

    fig_spec.update_xaxes(title_text='k', tickmode='linear', dtick=1, row=1, col=1)
    fig_spec.update_xaxes(title_text='k (centered)', tickmode='linear', dtick=1, row=1, col=2)
    fig_spec.update_yaxes(title_text='|C_k|', row=1, col=1)
    fig_spec.update_layout(template='plotly_white', showlegend=False,
                            height=400,
                            margin=dict(l=40, r=20, t=60, b=40))

    # --- 3. Полярный спектр (угол = 2πk/N) ---
    angles_deg = 360 * n / N

    # Цвет по фазе для ненулевых, серый для нулевых
    phases_deg = np.degrees(np.angle(C))
    colors_polar = []
    for k in range(N):
        if mags[k] > 1e-6:
            colors_polar.append(phases_deg[k])
        else:
            colors_polar.append(0)  # neutral

    fig_polar = go.Figure(go.Barpolar(
        r=mags,
        theta=angles_deg,
        marker=dict(
            color=colors_polar,
            colorscale='HSV',
            cmin=-180, cmax=180,
            colorbar=dict(title='фаза, °', ticksuffix='°'),
            line=dict(color='black', width=1),
        ),
        text=[f'k={k}' for k in n],
        hovertemplate='k=%{text}<br>|C_k|=%{r:.4f}<br>угол=%{theta:.1f}°<extra></extra>',
    ))
    fig_polar.update_layout(
        title='Полярный спектр: угол = 2π·k/N, радиус = |C_k|, цвет = фаза',
        polar=dict(
            radialaxis=dict(visible=True,
                            range=[0, max(1e-6, np.max(mags)) * 1.2]),
            angularaxis=dict(rotation=0, direction='counterclockwise',
                              tickmode='array',
                              tickvals=list(range(0, 360, 45)),
                              ticktext=[f'{a}°' for a in range(0, 360, 45)]),
        ),
        template='plotly_white', height=500,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # --- 4. Инфо ---
    nonzero = [(k, C[k]) for k in range(N) if abs(C[k]) > 1e-6]
    lines = ['Ненулевые гармоники:']
    for k, c in nonzero:
        alias = f' ≡ {k-N:+d}' if k > N / 2 else ''
        lines.append(f'  k={k:2d}{alias:8s}: |C_k|={abs(c):.4f}, arg={np.angle(c):+.3f} рад')

    fft_ok = np.allclose(C, np.fft.fft(f) / N)
    rec_ok = np.allclose(f_rec, f)
    lines.append('')
    lines.append(f'Совпадает с np.fft.fft/N: {fft_ok}')
    lines.append(f'Обратное ДПФ восстанавливает: {rec_ok}')
    lines.append('')
    lines.append('Обратите внимание на полярный график:')
    lines.append(f'  k={k1} (угол {360*k1/N:.0f}°) и k={N-k1} (угол {360*(N-k1)/N:.0f}°) '
                 f'смотрят в противоположные стороны от вещественной оси — это сопряжённые пары.')

    info = '\n'.join(lines)

    return fig_sig, fig_spec, fig_polar, info

# =============================================================================
# CALLBACK 8: Спектр прямоугольных импульсов
# =============================================================================
@app.callback(
    Output('plot-pulses', 'figure'),
    Output('pulse-info', 'children'),
    Input('pulse-q', 'value'),
    Input('pulse-tau', 'value'),
)
def update_pulses(Q, tau):
    A = 1.0
    T = Q * tau

    # --- 1. Временная диаграмма ---
    t_max = 3 * T
    t = np.linspace(-0.5 * T, t_max - 0.5 * T, 2000)
    signal = np.where((np.mod(t + tau/2, T)) < tau, A, 0.0)

    # --- 2. Спектр (частотная область) ---
    n = np.arange(-40, 41)
    Cn = (A / Q) * np.sinc(n / Q)
    fk = n / T

    f_dense = np.linspace(-40 / T, 40 / T, 1000)
    envelope = A * tau * np.sinc(f_dense * tau)

    # --- 3. Собираем Subplots сразу ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Сигнал во времени (T = {T:.2f}, τ = {tau:.2f})',
            'Спектр (частоты n/T)'
        ),
        vertical_spacing=0.15
    )

    # Трейс для времени
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines', name='Сигнал',
        line=dict(color='#1f77b4', width=2), fill='tozeroy'
    ), row=1, col=1)

    # Трейс для огибающей
    fig.add_trace(go.Scatter(
        x=f_dense, y=envelope, mode='lines',
        name='Огибающая Aτ·sinc(fτ)',
        line=dict(color='#d62728', width=2)
    ), row=2, col=1)

    # Трейсы для палочек (гармоник)
    for x_val, y_val in zip(fk, Cn):
        if abs(y_val) > 1e-4:
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[0, y_val],
                mode='lines', line=dict(color='#2ca02c', width=1),
                showlegend=False, hoverinfo='skip'
            ), row=2, col=1)

    # Точки гармоник
    fig.add_trace(go.Scatter(
        x=fk, y=Cn, mode='markers',
        name='Гармоники C_n',
        marker=dict(color='#2ca02c', size=6)
    ), row=2, col=1)

    # Оформление
    fig.update_layout(
        template='plotly_white', height=600, showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(title_text="t (сек)", row=1, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=1)
    fig.update_xaxes(title_text="f (Гц)", row=2, col=1)
    fig.update_yaxes(title_text="C_n", row=2, col=1)

    # --- 4. Текстовая сводка ---
    info = (
        f"Скважность Q = T/τ = {Q:.1f}\n"
        f"Период T = {T:.3f} с\n"
        f"Частота первой гармоники f1 = 1/T = {1/T:.3f} Гц (расстояние между палочками)\n"
        f"Среднее значение (постоянная составляющая) S(0) = A/Q = {A/Q:.3f}\n\n"
        f"Обратите внимание: когда Q — целое число, каждая Q-я гармоника попадает точно в ноль огибающей sinc."
    )

    return fig, info

# =============================================================================
# CALLBACK 9: Переход к непрерывному спектру
# =============================================================================
@app.callback(
    Output('plot-continuous-limit', 'figure'),
    Input('limit-q', 'value'),
    Input('limit-tau', 'value'),
    Input('limit-scale', 'value'),
)
def update_continuous_limit(Q, tau, scale_mode):
    A = 1.0
    T = Q * tau

    # Берем гармоники от -80 до 80
    k = np.arange(-80, 81)

    # Коэффициенты ряда Фурье
    Cn = (A / Q) * np.sinc(k / Q)
    fk = k / T

    # Огибающая (непрерывное преобразование Фурье одного импульса)
    f_dense = np.linspace(-80 / T, 80 / T, 1200)
    envelope = A * tau * np.sinc(f_dense * tau)

    fig = go.Figure()

    # 1. Огибающая
    fig.add_trace(go.Scatter(
        x=f_dense, y=envelope, mode='lines',
        name='Непрерывная огибающая Aτ·sinc(fτ)',
        line=dict(color='#d62728', width=2)
    ))

    # 2. Гармоники
    if scale_mode == 'scaled':
        y_vals = Cn * T
        marker_name = 'Гармоники C_n · T (масштабированные)'
    else:
        y_vals = Cn
        marker_name = 'Гармоники C_n (истинные)'

    # Рисуем палочки от нуля до y_vals
    for x_val, y_val in zip(fk, y_vals):
        if abs(y_val) > 1e-5:
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[0, y_val],
                mode='lines', line=dict(color='#2ca02c', width=1),
                showlegend=False, hoverinfo='skip'
            ))

    # Точки на вершинах палочек
    fig.add_trace(go.Scatter(
        x=fk, y=y_vals, mode='markers',
        name=marker_name,
        marker=dict(color='#2ca02c', size=5)
    ))

    fig.update_layout(
        title=f'Переход к непрерывному спектру (T/τ = {Q}, T = {T:.2f})',
        xaxis=dict(title='Частота f (Гц)', zeroline=True, zerolinewidth=1),
        yaxis=dict(title='Амплитуда', zeroline=True, zerolinewidth=1),
        template='plotly_white',
        height=500,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return fig

# =============================================================================
# CALLBACK 10: Спектр одиночного импульса
# =============================================================================
@app.callback(
    Output('plot-single-pulse', 'figure'),
    Input('single-tau', 'value'),
    Input('single-tview', 'value'),
    Input('single-m', 'value'),
)
def update_single_pulse(tau, T_view, M):
    A = 1.0

    # --- 1. Подготовка временной оси ---
    t = np.linspace(-T_view/2, T_view/2, M, endpoint=False)
    dt = t[1] - t[0]

    # --- 2. Формирование сигнала ---
    x = np.where(np.abs(t) < tau/2, A, 0.0)

    # --- 3. Вычисление ДПФ (с масштабированием на dt) ---
    # Используем fftshift и ifftshift для правильного центрирования
    X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
    f_ax = np.fft.fftshift(np.fft.fftfreq(M, dt))

    # --- 4. Теоретический спектр ---
    # Ограничим отображение частотами, где sinc еще осмыслен (например, ±5/tau)
    f_limit = 5.0 / tau
    f_dense = np.linspace(-f_limit, f_limit, 1000)
    envelope = A * tau * np.sinc(f_dense * tau)

    # --- 5. Создаем фигуру с двумя сабплотами ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Одиночный прямоугольный импульс (τ = {tau:.2f}, T_view = {T_view:.1f})',
            'Спектр: ДПФ (численно) vs Aτ·sinc(fτ) (теория)'
        ),
        vertical_spacing=0.15
    )

    # Временная область
    fig.add_trace(go.Scatter(
        x=t, y=x, mode='lines', name='Импульс x(t)',
        line=dict(color='#1f77b4', width=2), fill='tozeroy'
    ), row=1, col=1)

    # Частотная область: ДПФ (точки)
    # Для наглядности рисуем маркерами без линий
    fig.add_trace(go.Scatter(
        x=f_ax, y=X.real, mode='markers', name='ДПФ·Δt (численно)',
        marker=dict(color='#2ca02c', size=4)
    ), row=2, col=1)

    # Частотная область: Теория (непрерывная линия)
    fig.add_trace(go.Scatter(
        x=f_dense, y=envelope, mode='lines', name='Aτ·sinc(fτ) (теория)',
        line=dict(color='#d62728', width=2, dash='dash')
    ), row=2, col=1)

    # Настройка осей
    fig.update_layout(
        template='plotly_white', height=650, showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(title_text="Время t (сек)", row=1, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=1)

    fig.update_xaxes(title_text="Частота f (Гц)", range=[-f_limit, f_limit], row=2, col=1)
    fig.update_yaxes(title_text="Спектр X(f)", row=2, col=1)

    return fig

# =============================================================================
# CALLBACK 11: Базис Котельникова
# =============================================================================
@app.callback(
    Output('plot-sinc-basis', 'figure'),
    Output('plot-sinc-product', 'figure'),
    Output('sinc-info', 'children'),
    Input('sinc-k1', 'value'),
    Input('sinc-k2', 'value'),
    Input('sinc-limit', 'value'),
)
def update_sinc_basis(k1, k2, limit):
    # Создаем плотную сетку для точного интегрирования
    num_points = max(2000, int(limit * 400))
    t = np.linspace(-limit, limit, num_points)

    def phi(k):
        # np.sinc(x) уже включает деление на pi: sin(pi*x)/(pi*x)
        return np.sinc(t - k)

    # --- График 1: Сами базисные функции ---
    fig_basis = go.Figure()

    # Рисуем "фоном" все функции от -3 до 3
    for k in range(-3, 4):
        if k not in (k1, k2):
            fig_basis.add_trace(go.Scatter(
                x=t, y=phi(k), mode='lines',
                line=dict(color='lightgray', width=1),
                name=f'φ_{k}', showlegend=False, hoverinfo='skip'
            ))

    # Выделяем выбранные k1 и k2
    fig_basis.add_trace(go.Scatter(
        x=t, y=phi(k1), mode='lines', name=f'φ_{k1} (выбран)',
        line=dict(color='#1f77b4', width=3)
    ))
    fig_basis.add_trace(go.Scatter(
        x=t, y=phi(k2), mode='lines', name=f'φ_{k2} (выбран)',
        line=dict(color='#d62728', width=3, dash='dash')
    ))

    fig_basis.update_layout(
        title='Базис Котельникова: φ_k(t) = sinc(t − k)',
        xaxis_title='t', yaxis_title='Амплитуда',
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- График 2: Произведение и площадь интеграла ---
    prod = phi(k1) * phi(k2)

    # Считаем интеграл (площадь под кривой)
    # np.trapezoid появился в NumPy 2.0, для совместимости используем getattr
    integral = np.trapezoid(prod, t)

    # Разделяем на положительную и отрицательную части для красивого закрашивания
    pos_prod = np.where(prod > 0, prod, 0)
    neg_prod = np.where(prod < 0, prod, 0)

    fig_prod = go.Figure()

    # Положительная площадь (зеленая)
    fig_prod.add_trace(go.Scatter(
        x=t, y=pos_prod, mode='lines', fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.4)', line=dict(color='#2ca02c', width=1),
        name='Положительная площадь'
    ))

    # Отрицательная площадь (красная)
    fig_prod.add_trace(go.Scatter(
        x=t, y=neg_prod, mode='lines', fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.4)', line=dict(color='#d62728', width=1),
        name='Отрицательная площадь'
    ))

    # Сам график произведения
    fig_prod.add_trace(go.Scatter(
        x=t, y=prod, mode='lines',
        line=dict(color='black', width=1.5),
        name=f'φ_{k1} · φ_{k2}'
    ))

    fig_prod.update_layout(
        title=f'Произведение φ_{k1} · φ_{k2} и его интеграл на отрезке [-{limit}, {limit}]',
        xaxis_title='t', yaxis_title='Произведение',
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- Текст с результатом ---
    if k1 == k2:
        status = "Это НОРМИРОВКА: k₁ = k₂. Интеграл квадрата функции стремится к 1."
        color_note = "Площадь везде положительная (кроме нулей), она сходится к 1."
    else:
        status = "Это ОРТОГОНАЛЬНОСТЬ: k₁ ≠ k₂. Интеграл произведения стремится к 0."
        color_note = "Зеленая и красная площади в точности компенсируют друг друга!"

    info = (
        f"Численное значение интеграла: {integral:.6f}\n\n"
        f"{status}\n"
        f"{color_note}\n\n"
        f"Попробуйте уменьшить предел интегрирования L (ползунок ниже).\n"
        f"Вы увидите, что на конечном отрезке площади не компенсируются идеально, \n"
        f"и интеграл перестает быть равным нулю (для k₁ ≠ k₂)."
    )

    return fig_basis, fig_prod, info

# =============================================================================
# CALLBACK 12: Механика и Электрика
# =============================================================================
@app.callback(
    Output('plot-mech-elec', 'figure'),
    Output('plot-phase-portrait', 'figure'),
    Input('me-m', 'value'),
    Input('me-b', 'value'),
    Input('me-k', 'value'),
    Input('me-f', 'value'),
)
def update_mech_elec(m_val, b_val, k_val, f_amp):
    # Дифференциальное уравнение
    def rhs(t, y):
        F = 0.0 if t < 0.5 else f_amp # Ступенька на 0.5 секунде
        return [y[1], (F - b_val * y[1] - k_val * y[0]) / m_val]

    # Решаем от 0 до 8 секунд
    sol = solve_ivp(rhs, [0, 8], [0.0, 0.0], t_eval=np.linspace(0, 8, 900),
                    rtol=1e-9, atol=1e-11)

    # --- График 1: Временные ряды (Механика и Электрика) ---
    fig_time = make_subplots(rows=1, cols=2,
                             subplot_titles=('Механика: груз на пружине x(t)',
                                             'Электрика: заряд q(t) в RLC-контуре'))

    fig_time.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode='lines',
                                  name='x(t)', line=dict(color='#1f77b4', width=2)), row=1, col=1)

    fig_time.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode='lines',
                                  name='q(t)', line=dict(color='#d62728', width=2)), row=1, col=2)

    # Добавляем вертикальные линии включения
    fig_time.add_vline(x=0.5, line_dash="dot", line_color="gray", row=1, col=1, annotation_text="Толчок F")
    fig_time.add_vline(x=0.5, line_dash="dot", line_color="gray", row=1, col=2, annotation_text="Вкл. u")

    fig_time.update_layout(template='plotly_white', showlegend=False,
                           margin=dict(l=40, r=20, t=50, b=40))
    fig_time.update_xaxes(title_text="t (сек)", row=1, col=1)
    fig_time.update_yaxes(title_text="x (координата)", row=1, col=1)
    fig_time.update_xaxes(title_text="t (сек)", row=1, col=2)
    fig_time.update_yaxes(title_text="q (заряд)", row=1, col=2)


    # --- График 2: Фазовый портрет ---
    fig_phase = go.Figure()

    fig_phase.add_trace(go.Scatter(
        x=sol.y[0], y=sol.y[1], mode='lines',
        line=dict(color='#2ca02c', width=2),
        name='Траектория'
    ))

    # Отмечаем начальную и конечную точки
    fig_phase.add_trace(go.Scatter(
        x=[sol.y[0][0]], y=[sol.y[1][0]], mode='markers',
        marker=dict(color='black', size=8, symbol='circle'),
        name='Старт'
    ))
    fig_phase.add_trace(go.Scatter(
        x=[sol.y[0][-1]], y=[sol.y[1][-1]], mode='markers',
        marker=dict(color='#d62728', size=8, symbol='x'),
        name='Конец (t=8)'
    ))

    fig_phase.update_layout(
        title='Фазовый портрет: (x, v) или (q, i)',
        xaxis_title='Координата / Заряд',
        yaxis_title='Скорость / Ток',
        template='plotly_white',
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig_time, fig_phase

# =============================================================================
# CALLBACK 13: Передаточная функция H(s)
# =============================================================================
@app.callback(
    Output('hs-formula', 'children'),
    Output('plot-pole-zero', 'figure'),
    Output('plot-bode', 'figure'),
    Output('plot-impulse', 'figure'),
    Input('hs-type', 'value'),
    Input('hs-wc', 'value'),
    Input('hs-q', 'value'),
)
def update_transfer_function(f_type, wc, Q):
    K = 1.0  # коэффициент усиления

    # --- Коэффициенты числителя и знаменателя ---
    # Знаменатель всегда: s^2 + (wc/Q)*s + wc^2
    a = [1.0, wc / Q, wc * wc]

    if f_type == 'low':
        b = [K * wc * wc, 0.0, 0.0]  # числитель: K*wc^2  (в порядке убывания степеней s)
        formula = "H(s) = K·ωc² / (s² + (ωc/Q)·s + ωc²)"
    elif f_type == 'high':
        b = [K, 0.0, 0.0]           # числитель: K*s²
        formula = "H(s) = K·s² / (s² + (ωc/Q)·s + ωc²)"
    elif f_type == 'band':
        b = [K * wc / Q, 0.0, 0.0]  # числитель: K*(ωc/Q)*s
        formula = "H(s) = K·(ωc/Q)·s / (s² + (ωc/Q)·s + ωc²)"
    else:  # notch
        b = [K, 0.0, K * wc * wc]    # числитель: K*(s² + ωc²)
        formula = "H(s) = K·(s² + ωc²) / (s² + (ωc/Q)·s + ωc²)"

    # scipy.signal.lti ожидает коэффициенты в порядке убывания степеней s
    # (b[0]s^2 + b[1]s + b[2]) / (a[0]s^2 + a[1]s + a[2])
    sys = signal.TransferFunction(b, a)
    zeros, poles, _ = signal.tf2zpk(b, a)

    # --- Текст с формулой ---
    formula_text = (
        f"{formula}\n"
        f"Подстановка: K = {K:.2f}, ωc = {wc:.2f}, Q = {Q:.2f}\n"
        f"Знаменатель: {a[0]:.3f}·s² + {a[1]:.4f}·s + {a[2]:.3f}\n"
        f"Числитель: {b[0]:.3f}·s² + {b[1]:.4f}·s + {b[2]:.3f}"
    )

    # --- 1. Карта полюсов и нулей ---
    fig_pz = go.Figure()

    # Ось мнимых значений
    fig_pz.add_hline(y=0, line_color='black', line_width=1)
    fig_pz.add_vline(x=0, line_color='black', line_width=1)

    # Полюса (символ 'x')
    if len(poles) > 0:
        fig_pz.add_trace(go.Scatter(
            x=poles.real, y=poles.imag, mode='markers',
            marker=dict(symbol='x', size=12, color='#d62728', line=dict(width=2)),
            name='Полюса (poles)',
            hovertemplate='%{x:.3f} + %{y:.3f}j<extra>Полюс</extra>'
        ))

    # Нули (символ 'o')
    if len(zeros) > 0:
        fig_pz.add_trace(go.Scatter(
            x=zeros.real, y=zeros.imag, mode='markers',
            marker=dict(symbol='circle-open', size=12, color='#2ca02c', line=dict(width=2)),
            name='Нули (zeros)',
            hovertemplate='%{x:.3f} + %{y:.3f}j<extra>Ноль</extra>'
        ))

    fig_pz.update_layout(
        title='Карта нулей и полюсов на комплексной плоскости (s = σ + jω)',
        xaxis_title='Re(s)', yaxis_title='Im(s)',
        template='plotly_white',
        xaxis=dict(zeroline=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40),
        height=380
    )

    # --- 2. Диаграмма Боде (АЧХ и ФЧХ) ---
    w = np.logspace(np.log10(wc/100), np.log10(wc*100), 1000)
    w, H_jw = signal.freqs(b, a, worN=w)

    magnitude_db = 20 * np.log10(np.abs(H_jw) + 1e-12)  # +eps для лога
    phase_deg = np.degrees(np.angle(H_jw))

    fig_bode = make_subplots(rows=2, cols=1,
                              subplot_titles=('АЧХ (Амплитудно-частотная)', 'ФЧХ (Фазо-частотная)'),
                              shared_xaxes=True,
                              vertical_spacing=0.1)

    fig_bode.add_trace(go.Scatter(x=w, y=magnitude_db, mode='lines',
                                   line=dict(color='#1f77b4', width=2),
                                   name='|H(jω)| (дБ)'), row=1, col=1)

    fig_bode.add_trace(go.Scatter(x=w, y=phase_deg, mode='lines',
                                   line=dict(color='#d62728', width=2),
                                   name='∠H(jω) (град)'), row=2, col=1)

    fig_bode.update_xaxes(type="log", title_text="ω (рад/с)", row=2, col=1)
    fig_bode.update_yaxes(title_text="дБ", row=1, col=1)
    fig_bode.update_yaxes(title_text="градусы", row=2, col=1)

    # Добавляем вертикальную линию wc
    fig_bode.add_vline(x=wc, line_dash="dash", line_color="gray", row=1, col=1, annotation_text="ωc")
    fig_bode.add_vline(x=wc, line_dash="dash", line_color="gray", row=2, col=1)

    fig_bode.update_layout(
        template='plotly_white', showlegend=False,
        height=420, margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- 3. Импульсная и переходная характеристики ---
    # Импульсная h(t) и ступенька s(t)
    t_vec, y_imp = signal.impulse(sys, N=1000)  # симулируем 1000 точек
    t_step, y_step = signal.step(sys, N=1000)

    fig_imp = make_subplots(rows=1, cols=2,
                             subplot_titles=('Импульсная характеристика h(t)',
                                             'Переходная характеристика s(t)'))

    fig_imp.add_trace(go.Scatter(x=t_vec, y=y_imp, mode='lines',
                                  line=dict(color='#1f77b4', width=2),
                                  name='h(t)'), row=1, col=1)

    fig_imp.add_trace(go.Scatter(x=t_step, y=y_step, mode='lines',
                                  line=dict(color='#2ca02c', width=2),
                                  name='s(t)'), row=1, col=2)

    fig_imp.update_layout(template='plotly_white', showlegend=False,
                           height=350, margin=dict(l=40, r=20, t=40, b=40))
    fig_imp.update_xaxes(title_text="t (сек)", row=1, col=1)
    fig_imp.update_xaxes(title_text="t (сек)", row=1, col=2)
    fig_imp.update_yaxes(title_text="h(t)", row=1, col=1)
    fig_imp.update_yaxes(title_text="s(t)", row=1, col=2)

    return formula_text, fig_pz, fig_bode, fig_imp

# =============================================================================
# CALLBACK 14: Импеданс RLC
# =============================================================================
@app.callback(
    Output('plot-rlc-impedance', 'figure'),
    Output('rlc-info', 'children'),
    Input('rlc-r', 'value'),
    Input('rlc-l', 'value'),
    Input('rlc-c', 'value'),
)
def update_rlc(R, L_mH, C_uF):
    # Переводим в базовые единицы СИ
    L = L_mH * 1e-3
    C = C_uF * 1e-6

    # Диапазон частот (от 10^2 до 10^6 рад/с)
    w = np.logspace(2, 6, 1500)

    # Комплексный импеданс Z = R + jωL + 1/(jωC)
    Z = R + 1j * w * L + 1 / (1j * w * C)

    # Резонансная частота
    w0 = 1 / np.sqrt(L * C)
    f0 = w0 / (2 * np.pi)

    # Добротность
    Q_factor = (1 / R) * np.sqrt(L / C)

    # --- График с двумя осями Y (двойная ось для АЧХ и ФЧХ) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. |Z| — логарифмический масштаб
    fig.add_trace(
        go.Scatter(x=w, y=np.abs(Z), mode='lines',
                   name='|Z(ω)| (Ом)',
                   line=dict(color='#1f77b4', width=2)),
        secondary_y=False,
    )

    # 2. Фаза — градусы
    fig.add_trace(
        go.Scatter(x=w, y=np.degrees(np.angle(Z)), mode='lines',
                   name='∠Z(ω) (град)',
                   line=dict(color='#d62728', width=2, dash='dot')),
        secondary_y=True,
    )

    # Настройка осей
    fig.update_xaxes(title_text="ω (рад/с)", type="log")

    fig.update_yaxes(title_text="|Z| (Ом)", type="log", secondary_y=False, color='#1f77b4')
    fig.update_yaxes(title_text="Фаза (градусы)", range=[-100, 100], secondary_y=True, color='#d62728')

    # Линия резонанса
    fig.add_vline(x=w0, line_dash="dash", line_color="green",
                  annotation_text=f"ω₀ = {w0:.0f}",
                  annotation_position="top right")

    fig.update_layout(
        title='Импеданс последовательного RLC-контура: провал |Z| до R на резонансе',
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Текстовая информация ---
    Z_at_res = abs(R + 1j*w0*L + 1/(1j*w0*C))
    info = (
        f"Параметры: R = {R:.1f} Ом, L = {L_mH:.1f} мГн, C = {C_uF:.1f} мкФ\n"
        f"Резонансная частота: ω0 = 1/√(LC) = {w0:.0f} рад/с  (f0 = {f0:.0f} Гц)\n"
        f"Добротность: Q = (1/R)·√(L/C) = {Q_factor:.2f}\n\n"
        f"На резонансе: |Z(ω0)| = {Z_at_res:.3f} Ом  (равно R — индуктивное и ёмкостное сопротивления взаимно компенсируются!)\n"
        f"Фаза на низких частотах: -90° (ведёт себя как конденсатор)\n"
        f"Фаза на высоких частотах: +90° (ведёт себя как катушка индуктивности)\n"
        f"Фаза на резонансе: 0° (чисто активное сопротивление)"
    )

    return fig, info

# =============================================================================
# CALLBACK 15: Разряд конденсатора
# =============================================================================
@app.callback(
    Output('plot-discharge', 'figure'),
    Output('plot-discharge-poles', 'figure'),
    Output('discharge-info', 'children'),
    Input('dis-r', 'value'),
    Input('dis-l', 'value'),
    Input('dis-c', 'value'),
)
def update_discharge(R, L, C):
    # Эталонные случаи для L=1, C=1
    ref_Rs = [0.5, 2.0, 8.0]
    ref_names = ['колебательный', 'критический', 'апериодический']

    fig_time = go.Figure()

    # 1. Рисуем эталонные графики (пунктиром, серым)
    for r_ref, name in zip(ref_Rs, ref_names):
        sys_ref = signal.TransferFunction([1, r_ref], [1, r_ref, 1])
        t_ref, q_ref = signal.impulse(sys_ref)
        # Ограничиваем время 20 секундами для наглядности
        t_ref_masked = t_ref[t_ref <= 20]
        q_ref_masked = q_ref[t_ref <= 20]
        fig_time.add_trace(go.Scatter(
            x=t_ref_masked, y=q_ref_masked, mode='lines',
            name=f'R={r_ref} ({name})',
            line=dict(width=1.5, dash='dot', color='gray')
        ))

    # 2. Рисуем текущий график (жирным, цветным)
    sys_curr = signal.TransferFunction([1, R/L], [1, R/L, 1/(L*C)])
    t_curr, q_curr = signal.impulse(sys_curr)
    t_curr_masked = t_curr[t_curr <= 20]
    q_curr_masked = q_curr[t_curr <= 20]

    fig_time.add_trace(go.Scatter(
        x=t_curr_masked, y=q_curr_masked, mode='lines',
        name=f'Текущий: R={R:.1f}, L={L:.1f}, C={C:.1f}',
        line=dict(width=3, color='#1f77b4')
    ))

    fig_time.update_layout(
        title='Разряд конденсатора: q(t) для разных характеров процесса',
        xaxis_title='t (сек)', yaxis_title='q(t)',
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- График 2: Полюса на комплексной плоскости ---
    poles, zeros, _ = signal.tf2zpk([1, R/L], [1, R/L, 1/(L*C)])

    fig_poles = go.Figure()
    fig_poles.add_hline(y=0, line_color='black', line_width=1)
    fig_poles.add_vline(x=0, line_color='black', line_width=1)

    fig_poles.add_trace(go.Scatter(
        x=poles.real, y=poles.imag, mode='markers',
        marker=dict(symbol='x', size=14, color='#d62728', line=dict(width=2)),
        name='Полюса (корни)'
    ))

    fig_poles.update_layout(
        title='Расположение полюсов: от мнимой оси к вещественной',
        xaxis_title='Re(s)', yaxis_title='Im(s)',
        template='plotly_white',
        xaxis=dict(zeroline=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(zeroline=False),
        height=350, margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- Текст с результатом ---
    alpha = R / (2*L)
    w0 = 1 / np.sqrt(L*C)
    crit_R = 2 * np.sqrt(L/C)

    if R < crit_R:
        regime = "КОЛЕБАТЕЛЬНЫЙ (недодемпфированный)"
        reason = "Полюса комплексно-сопряжённые (справа налево через Im)."
    elif np.isclose(R, crit_R):
        regime = "КРИТИЧЕСКИЙ"
        reason = "Полюса совпали на вещественной оси."
    else:
        regime = "АПЕРИОДИЧЕСКИЙ (передемпфированный)"
        reason = "Два разных вещественных полюса."

    info = (
        f"Параметры: R = {R:.1f} Ом, L = {L:.1f} Гн, C = {C:.1f} Ф\n"
        f"Характер разряда: {regime}\n"
        f"Критическое сопротивление: R_crit = 2·√(L/C) = {crit_R:.2f} Ом\n"
        f"{reason}\n\n"
        f"Полюса (корни s² + (R/L)s + 1/(LC) = 0):\n"
    )
    for p in poles:
        if np.isclose(p.imag, 0):
            info += f"  s = {p.real:.3f}\n"
        else:
            info += f"  s = {p.real:.3f} ± {abs(p.imag):.3f}j\n"

    return fig_time, fig_poles, info

# =============================================================================
# CALLBACK 16: RC-ФНЧ (первый порядок)
# =============================================================================
@app.callback(
    Output('plot-rc-bode', 'figure'),
    Output('plot-rc-step', 'figure'),
    Output('rc-info', 'children'),
    Input('rc-r', 'value'),
    Input('rc-c', 'value'),
)
def update_rc_lowpass(R_kOhm, C_uF):
    # Перевод в СИ
    R = R_kOhm * 1e3
    C = C_uF * 1e-6

    # Постоянная времени и частота среза
    tau = R * C
    f0 = 1 / (2 * np.pi * tau)

    # --- 1. Диаграмма Боде ---
    f = np.logspace(np.log10(f0) - 3, np.log10(f0) + 3, 900)
    H_jw = 1 / (1 + 1j * 2 * np.pi * f * tau)

    magnitude_db = 20 * np.log10(np.abs(H_jw))
    phase_deg = np.degrees(np.angle(H_jw))

    fig_bode = make_subplots(specs=[[{"secondary_y": True}]])

    fig_bode.add_trace(
        go.Scatter(x=f, y=magnitude_db, mode='lines',
                   name='|K(f)| (дБ)',
                   line=dict(color='#1f77b4', width=2)),
        secondary_y=False
    )
    fig_bode.add_trace(
        go.Scatter(x=f, y=phase_deg, mode='lines',
                   name='∠K(f) (град)',
                   line=dict(color='#d62728', width=2, dash='dot')),
        secondary_y=True
    )

    # Линии -3 дБ и f0
    fig_bode.add_hline(y=-3, line_dash="dash", line_color="green",
                       annotation_text="-3 дБ", annotation_position="bottom right",
                       secondary_y=False)
    fig_bode.add_vline(x=f0, line_dash="dot", line_color="gray",
                       annotation_text=f"f0 = {f0:.0f} Гц",
                       annotation_position="top right")

    fig_bode.update_xaxes(title_text="f (Гц)", type="log")
    fig_bode.update_yaxes(title_text="|K| (дБ)", type="linear",
                          range=[-40, 5], secondary_y=False, color='#1f77b4')
    fig_bode.update_yaxes(title_text="Фаза (градусы)", type="linear",
                          range=[-95, 5], secondary_y=True, color='#d62728')

    fig_bode.update_layout(
        title='RC-ФНЧ: АЧХ (граф Боде) и ФЧХ',
        template='plotly_white', height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- 2. Переходная характеристика ---
    t_step = np.linspace(0, 5 * tau, 400)
    y_step = 1 - np.exp(-t_step / tau)

    fig_step = go.Figure()

    fig_step.add_trace(go.Scatter(
        x=t_step / tau, y=y_step, mode='lines',
        name='h(t) = 1 - e^(−t/τ)',
        line=dict(color='#2ca02c', width=3)
    ))

    # Маркер 0.632 при t = τ
    fig_step.add_trace(go.Scatter(
        x=[1], y=[1 - 1/np.e], mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle'),
        name='0.63 при t = τ'
    ))

    # Горизонтальная линия 0.632
    fig_step.add_hline(y=0.632, line_dash="dash", line_color="red",
                       annotation_text="63.2%", annotation_position="right")

    fig_step.add_vline(x=1, line_dash="dot", line_color="gray")

    fig_step.update_layout(
        title='Переходная характеристика RC-ФНЧ (отклик на ступеньку)',
        xaxis_title='t / τ',
        yaxis_title='h(t)',
        template='plotly_white', height=350,
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=40, r=20, t=50, b=40)
    )

    # --- 3. Текстовая информация ---
    tau_ms = tau * 1e3
    H_at_f0 = abs(1 / (1 + 1j * 2 * np.pi * f0 * tau))

    info = (
        f"Параметры: R = {R_kOhm:.1f} кОм, C = {C_uF:.1f} мкФ\n"
        f"Постоянная времени: τ = R·C = {tau_ms:.2f} мс\n"
        f"Частота среза: f0 = 1/(2πτ) = {f0:.1f} Гц\n\n"
        f"На частоте среза: |K(f0)| = 1/√2 ≈ {H_at_f0:.4f}  (это соответствует −3 дБ)\n"
        f"На частоте среза: φ(f0) = -45°\n\n"
        f"Асимптотика АЧХ:\n"
        f"  f << f0  →  |K| ≈ 1 (0 дБ), сигнал проходит\n"
        f"  f >> f0  →  |K| ~ f0/f  (спад 20 дБ/декаду, -6 дБ/октаву)\n\n"
        f"Постоянная времени τ — это время, за которое при ступеньке выход достигает 63.2% от конечного значения.\n"
        f"За 5τ переходный процесс практически завершается (99.3%)."
    )

    return fig_bode, fig_step, info

# =============================================================================
# CALLBACK 17: Вычеты и обратное преобразование Лапласа
# =============================================================================
@app.callback(
    Output('plot-residue-poles', 'figure'),
    Output('plot-residue-signal', 'figure'),
    Output('residue-info', 'children'),
    Input('res-re1', 'value'),
    Input('res-im1', 'value'),
    Input('res-re2', 'value'),
    Input('res-im2', 'value'),
)
def update_residues(re1, im1, re2, im2):
    # Полюса
    p1 = complex(re1, im1)
    p2 = complex(re2, im2)

    # Защита от совпадения
    if abs(p1 - p2) < 1e-6:
        p2 = p2 + 0.1  # сдвигаем, чтобы избежать деления на ноль

    # Вычеты для X(s) = 1 / ((s-p1)(s-p2))
    # Res(s=p1) = lim (s-p1) X(s) = 1/(p1 - p2)
    Res1 = 1 / (p1 - p2)
    Res2 = 1 / (p2 - p1)

    # Сигнал x(t) = Res1 e^{p1 t} + Res2 e^{p2 t}
    t = np.linspace(0, 10, 800)
    x1 = Res1 * np.exp(p1 * t)
    x2 = Res2 * np.exp(p2 * t)
    x = x1 + x2

    # Берём вещественную часть (для сопряжённых полюсов это даёт реальный сигнал)
    x_real = x.real
    x1_real = x1.real
    x2_real = x2.real

    # --- График 1: Полюса на комплексной плоскости ---
    fig_poles = go.Figure()
    fig_poles.add_hline(y=0, line_color='black', line_width=1)
    fig_poles.add_vline(x=0, line_color='black', line_width=1)

    # Заливаем правую полуплоскость красным (неустойчивость)
    fig_poles.add_vrect(x0=0, x1=2, fillcolor="rgba(214, 39, 40, 0.1)",
                        line_width=0, layer="below")

    # Полюса
    fig_poles.add_trace(go.Scatter(
        x=[p1.real, p2.real], y=[p1.imag, p2.imag],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color='#d62728', line=dict(width=2)),
        text=['p₁', 'p₂'], textposition='top right',
        name='Полюса'
    ))

    fig_poles.update_layout(
        title='Карта полюсов (красная зона: Re(s) > 0 — неустойчивость)',
        xaxis_title='Re(s)', yaxis_title='Im(s)',
        template='plotly_white',
        xaxis=dict(zeroline=False, scaleanchor='y', scaleratio=1, range=[-2.5, 1.5]),
        yaxis=dict(zeroline=False, range=[-5.5, 5.5]),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- График 2: Сигнал и вклады каждого полюса ---
    fig_sig = go.Figure()

    fig_sig.add_trace(go.Scatter(
        x=t, y=x1_real, mode='lines',
        name='Re(Res₁ · e^{p₁t})',
        line=dict(color='#1f77b4', width=1.5, dash='dot')
    ))
    fig_sig.add_trace(go.Scatter(
        x=t, y=x2_real, mode='lines',
        name='Re(Res₂ · e^{p₂t})',
        line=dict(color='#2ca02c', width=1.5, dash='dot')
    ))
    fig_sig.add_trace(go.Scatter(
        x=t, y=x_real, mode='lines',
        name='x(t) = Σ вычетов',
        line=dict(color='#d62728', width=3)
    ))

    fig_sig.update_layout(
        title='Обратное преобразование: сумма вычетов',
        xaxis_title='t', yaxis_title='x(t)',
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Текстовая информация ---
    def fmt_complex(z):
        if abs(z.imag) < 1e-9:
            return f"{z.real:+.4f}"
        return f"{z.real:+.4f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.4f}j"

    info = (
        f"X(s) = 1 / ((s - p₁)(s - p₂))\n\n"
        f"p₁ = {fmt_complex(p1)}\n"
        f"p₂ = {fmt_complex(p2)}\n\n"
        f"Вычет в p₁: Res₁ = lim (s-p₁)·X(s)·e^(st) = 1/(p₁-p₂) = {fmt_complex(Res1)}\n"
        f"Вычет в p₂: Res₂ = 1/(p₂-p₁) = {fmt_complex(Res2)}\n\n"
        f"x(t) = Res₁·e^(p₁t) + Res₂·e^(p₂t)\n"
    )

    if abs(p1.imag) > 1e-6 or abs(p2.imag) > 1e-6:
        info += "\nПолюса комплексные — сигнал колеблется. Re(p) определяет скорость затухания, Im(p) — частоту.\n"
    else:
        info += "\nПолюса вещественные — сигнал без колебаний (чистые экспоненты).\n"

    if p1.real > 0 or p2.real > 0:
        info += "\n⚠ Re(p) > 0 → сигнал РАСТЁТ. Система неустойчива!\n"
    else:
        info += "\nRe(p) < 0 → сигнал затухает. Система устойчива.\n"

    return fig_poles, fig_sig, info

# =============================================================================
# CALLBACK 18: RC-делитель (рампа)
# =============================================================================
@app.callback(
    Output('plot-rc-divider', 'figure'),
    Output('plot-rc-divider-error', 'figure'),
    Output('rc-divider-info', 'children'),
    Input('rcd-r1', 'value'),
    Input('rcd-r2', 'value'),
    Input('rcd-c', 'value'),
    Input('rcd-k', 'value'),
)
def update_rc_divider(R1_kOhm, R2_kOhm, C_uF, k):
    # Перевод в СИ
    R1 = R1_kOhm * 1e3
    R2 = R2_kOhm * 1e3
    C = C_uF * 1e-6

    a = R2 * C
    tau = C * (R1 + R2)

    # --- Численное решение ОДУ ---
    def rhs(t, y):
        # x(t) = k*t, x'(t) = k
        # y' = (x + C*R2*x' - y) / (C*(R1+R2))
        return [(k*t + C*R2*k - y[0]) / tau]

    t_end = 8 * tau
    t_eval = np.linspace(0, t_end, 500)
    sol = solve_ivp(rhs, [0, t_end], [0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    y_ode = sol.y[0]

    # --- Аналитическое решение ---
    # y(t) = k * (t - R1*C*(1 - exp(-t/tau)))
    y_analytic = k * (t_eval - R1 * C * (1 - np.exp(-t_eval / tau)))

    # Входной сигнал (рампа)
    x_input = k * t_eval

    # Установившееся значение (асимптота): y_ss(t) = k*t - k*R1*C
    y_steady = k * t_eval - k * R1 * C

    # --- График 1: Сигналы ---
    fig_sig = go.Figure()

    fig_sig.add_trace(go.Scatter(
        x=t_eval / tau, y=x_input, mode='lines',
        name='Вход x(t) = k·t',
        line=dict(color='gray', width=1.5, dash='dot')
    ))

    fig_sig.add_trace(go.Scatter(
        x=t_eval / tau, y=y_steady, mode='lines',
        name='Асимптота: k(t − R₁C)',
        line=dict(color='#9467bd', width=1.5, dash='dash')
    ))

    # Аналитическое решение (толстая линия)
    fig_sig.add_trace(go.Scatter(
        x=t_eval / tau, y=y_analytic, mode='lines',
        name='Аналитика',
        line=dict(color='#d62728', width=3)
    ))

    # Численное решение (точки/тонкая линия поверх, чтобы показать совпадение)
    fig_sig.add_trace(go.Scatter(
        x=t_eval / tau, y=y_ode, mode='lines',
        name='ОДУ (scipy)',
        line=dict(color='#1f77b4', width=1.5, dash='dash')
    ))

    fig_sig.update_layout(
        title='Реакция RC-делителя на рампу',
        xaxis_title='t / τ',
        yaxis_title='Напряжение (В)',
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- График 2: Ошибка численного метода ---
    error = np.abs(y_ode - y_analytic)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=t_eval / tau, y=error, mode='lines',
        name='|Численно − Аналитика|',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy'
    ))

    fig_err.update_layout(
        title='Абсолютная ошибка между численным и аналитическим решением',
        xaxis_title='t / τ',
        yaxis_title='Ошибка (В)',
        yaxis=dict(type='log'),  # Логарифмический масштаб для наглядности
        template='plotly_white',
        height=300,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- Текст ---
    max_err = np.max(error)
    R1_over_R2 = R1 / R2

    info = (
        f"Параметры: R₁ = {R1_kOhm:.1f} кОм, R₂ = {R2_kOhm:.1f} кОм, C = {C_uF:.2f} мкФ, k = {k:.1f} В/с\n"
        f"a = R₂·C = {a*1e3:.2f} мс\n"
        f"τ = C·(R₁+R₂) = {tau*1e3:.2f} мс  (в {tau/a:.1f} раза больше a)\n\n"
        f"Аналитика: y(t) = k·(t − R₁C·(1 − e^(−t/τ)))\n"
        f"На бесконечности y_ss(t) = k·t − k·R₁C  (отставание на R₁C = {R1*C*1e3:.2f} мВ)\n\n"
        f"Максимальное расхождение численного решения с аналитикой: {max_err:.2e} В\n"
        f"(это ошибка численного метода solve_ivp, а не физики)"
    )

    return fig_sig, fig_err, info

# =============================================================================
# CALLBACK 19: Дельта-функция
# =============================================================================
@app.callback(
    Output('plot-delta-time', 'figure'),
    Output('plot-delta-spectrum', 'figure'),
    Output('delta-info', 'children'),
    Input('delta-eps', 'value'),
    Input('delta-probe-func', 'value'),
)
def update_delta(eps_log, probe_type):
    eps = 10 ** eps_log

    # Параметры сетки
    Tview = 1.0
    N = 8192
    t = (np.arange(N) - N//2) * Tview / N
    dt = t[1] - t[0]

    # Дельта-импульс
    x = np.where(np.abs(t) < eps/2, 1/eps, 0.0)

    # Спектр
    S = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))) * dt
    fk = np.fft.fftshift(np.fft.fftfreq(N, dt))

    # Ограничение для отображения спектра
    m = np.abs(fk) < 200

    # --- График 1: Временная область ---
    fig_time = go.Figure()

    # Рисуем несколько эталонных импульсов бледным фоном
    for eps_bg in [0.1, 0.02, 0.004]:
        x_bg = np.where(np.abs(t) < eps_bg/2, 1/eps_bg, 0.0)
        fig_time.add_trace(go.Scatter(
            x=t, y=x_bg, mode='lines',
            line=dict(color='lightgray', width=1),
            name=f'ε={eps_bg}', showlegend=False, hoverinfo='skip'
        ))

    # Текущий импульс
    fig_time.add_trace(go.Scatter(
        x=t, y=x, mode='lines',
        line=dict(color='#1f77b4', width=2),
        name=f'Текущий ε = {eps:.4f}'
    ))

    fig_time.update_layout(
        title='Временная область: δ(t) как предел прямоугольников (площадь = 1)',
        xaxis_title='t', yaxis_title='δ_ε(t)',
        xaxis=dict(range=[-0.1, 0.1]),
        yaxis=dict(range=[0, 1.1/eps]),
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- График 2: Спектр ---
    fig_spec = go.Figure()

    fig_spec.add_trace(go.Scatter(
        x=fk[m], y=S[m], mode='lines',
        line=dict(color='#d62728', width=2),
        name=f'|S(f)| при ε = {eps:.4f}'
    ))

    # Опорная линия 1.0 (спектр идеальной дельты)
    fig_spec.add_hline(y=1.0, line_dash="dash", line_color="green",
                       annotation_text="Идеальный спектр δ(t) = 1",
                       annotation_position="top right")

    fig_spec.update_layout(
        title='Частотная область: спектр стремится к константе 1',
        xaxis_title='f (Гц)', yaxis_title='|S(f)|',
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- Фильтрующее свойство ---
    if probe_type == 'const':
        f_t = np.ones_like(t)
        f_name = "f(t) = 1"
        f0_val = 1.0
    elif probe_type == 'gauss':
        f_t = np.exp(-t**2 / 0.01)  # Немного шире, чтобы было видно
        f_name = "f(t) = exp(-t²/0.01)"
        f0_val = 1.0
    else:
        f_t = np.cos(20 * t)
        f_name = "f(t) = cos(20t)"
        f0_val = 1.0

    integral = np.trapezoid(f_t * x, t)

    info = (
        f"Ширина импульса ε = {eps:.5f}\n"
        f"Высота импульса 1/ε = {1/eps:.1f}\n"
        f"Площадь (интеграл δ_ε dt): {np.trapezoid(x, t):.6f} (должна быть 1.0)\n\n"
        f"Спектр: S(f) = ∫ δ_ε(t) e^(-j2πft) dt = sinc(πfε)\n"
        f"На низких частотах (f << 1/ε) S(f) ≈ 1. При ε → 0 спектр становится идеально плоским.\n\n"
        f"Фильтрующее свойство (свёртка с пробной функцией):\n"
        f"  Выбрано: {f_name}\n"
        f"  f(0) = {f0_val:.6f}\n"
        f"  ∫ f(t)·δ_ε(t) dt = {integral:.6f}\n"
        f"  (при ε → 0 этот интеграл стремится к f(0), если f непрерывна в нуле)"
    )

    return fig_time, fig_spec, info

# =============================================================================
# CALLBACK 20: Интеграл Дюамеля
# =============================================================================
@app.callback(
    Output('plot-duhamel', 'figure'),
    Output('duhamel-info', 'children'),
    Input('duh-tau', 'value'),
    Input('duh-input', 'value'),
)
def update_duhamel(tau_ms, input_type):
    tau = tau_ms * 1e-3
    t_end = 10 * tau
    dt = tau / 200  # шаг численного интегрирования
    t = np.arange(0, t_end, dt)

    # 1. Входной сигнал
    if input_type == 'step':
        x = np.ones_like(t)
        x_name = "x(t) = σ(t) (ступенька)"
    elif input_type == 'pulse':
        x = np.where(t < 3*tau, 1.0, 0.0)
        x_name = "x(t) = прямоугольник (ширина 3τ)"
    else: # sine
        w = 2 * np.pi / (5 * tau)  # период = 5τ
        x = np.sin(w * t)
        x_name = "x(t) = sin(2πt / 5τ)"

    # 2. Импульсная характеристика h(t) = (1/τ) e^(-t/τ)
    h = (1/tau) * np.exp(-t/tau)

    # 3. Численная свёртка (Интеграл Дюамеля)
    y_num = np.convolve(x, h) * dt
    y_num = y_num[:len(t)]  # обрезаем до длины t

    # 4. Точное решение ОДУ через scipy.signal.lsim
    sys = signal.TransferFunction([1], [tau, 1])
    _, y_exact, _ = signal.lsim(sys, U=x, T=t)

    # --- График ---
    fig = go.Figure()

    # Входной сигнал (серый пунктир)
    fig.add_trace(go.Scatter(
        x=t/tau, y=x, mode='lines',
        name=f'Вход: {x_name}',
        line=dict(color='gray', width=1.5, dash='dot')
    ))

    # Импульсная характеристика h(t), умноженная на tau, чтобы наложить на график
    fig.add_trace(go.Scatter(
        x=t/tau, y=h*tau, mode='lines',
        name='Импульсная h(t) · τ (масштаб)',
        line=dict(color='#2ca02c', width=1.5, dash='dash')
    ))

    # Численная свёртка (Дюамель)
    fig.add_trace(go.Scatter(
        x=t/tau, y=y_num, mode='lines',
        name='Дюамель (свёртка численно)',
        line=dict(color='#1f77b4', width=2)
    ))

    # Точное решение ОДУ
    fig.add_trace(go.Scatter(
        x=t/tau, y=y_exact, mode='lines',
        name='Точное ОДУ (lsim)',
        line=dict(color='#d62728', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Свёртка x(t) с h(t) = интеграл Дюамеля',
        xaxis_title='t / τ',
        yaxis_title='Амплитуда',
        template='plotly_white', height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Текстовая информация ---
    max_err = np.max(np.abs(y_num - y_exact))
    info = (
        f"τ = {tau_ms:.1f} мс, шаг интегрирования dt = τ/200 = {dt*1e6:.1f} мкс\n"
        f"Вход: {x_name}\n\n"
        f"Интеграл Дюамеля: y(t) = ∫₀ᵗ x(τ) h(t−τ) dτ\n"
        f"Импульсная характеристика RC-ФНЧ: h(t) = (1/τ) e^(-t/τ)\n\n"
        f"Максимальное расхождение между численной свёрткой (np.convolve) и точным решением ОДУ (lsim):\n"
        f"  {max_err:.2e}\n"
        f"Это доказывает, что свёртка с импульсной характеристикой математически эквивалентна решению дифференциального уравнения."
    )

    return fig, info

# =============================================================================
# CALLBACK 21: Дельта в H(s) (прямая связь)
# =============================================================================
@app.callback(
    Output('plot-feedthrough', 'figure'),
    Output('feedthrough-info', 'children'),
    Input('ft-a', 'value'),
    Input('ft-input', 'value'),
)
def update_feedthrough(a_ms, input_type):
    a = a_ms * 1e-3
    tau = 4 * a
    t_end = 10 * tau
    dt = tau / 200
    t = np.arange(0, t_end, dt)

    # Входной сигнал
    if input_type == 'step':
        x = np.ones_like(t)
        x_name = "σ(t)"
        k = 1.0
    else:
        k = 1.0 / tau  # Наклон рампы, чтобы график был наглядным
        x = k * t
        x_name = f"{k:.1f}·t"

    # Аналитическое решение
    # H(s) = 1/4 + (3/4) / (1 + 4as)
    if input_type == 'step':
        # y(t) = 1/4 + 3/4 (1 - e^{-t/tau})
        y_exact = 0.25 + 0.75 * (1 - np.exp(-t/tau))
    else:
        # y(t) = 1/4 k t + 3/4 k (t - tau(1 - e^{-t/tau}))
        y_exact = 0.25 * k * t + 0.75 * k * (t - tau * (1 - np.exp(-t/tau)))

    # Численная свёртка (чтобы показать, что она тоже работает)
    # Создаем импульсную характеристику: дельта (очень узкий импульс) + экспонента
    delta_approx = np.zeros_like(t)
    delta_approx[0] = 1.0 / dt  # Площадь 1
    h = 0.25 * delta_approx + (3 / (4 * tau)) * np.exp(-t/tau)

    y_num = np.convolve(x, h) * dt
    y_num = y_num[:len(t)]

    # --- График ---
    fig = go.Figure()

    # Вход
    fig.add_trace(go.Scatter(
        x=t/tau, y=x, mode='lines',
        name=f'Вход: {x_name}',
        line=dict(color='gray', width=1.5, dash='dot')
    ))

    # Выход (аналитика)
    fig.add_trace(go.Scatter(
        x=t/tau, y=y_exact, mode='lines',
        name='Выход (аналитика)',
        line=dict(color='#d62728', width=3)
    ))

    # Выход (численная свёртка)
    fig.add_trace(go.Scatter(
        x=t/tau, y=y_num, mode='lines',
        name='Выход (свёртка)',
        line=dict(color='#1f77b4', width=1.5, dash='dash')
    ))

    fig.update_layout(
        title='Выход системы с прямой связью (дельта в H(s))',
        xaxis_title='t / τ',
        yaxis_title='Амплитуда',
        template='plotly_white', height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Текст ---
    info = (
        f"H(s) = (1 + as) / (1 + 4as) = 1/4 + (3/4) / (1 + 4as)\n"
        f"a = {a_ms:.1f} мс, τ = 4a = {4*a_ms:.1f} мс\n\n"
        f"Импульсная характеристика: h(t) = 1/4·δ(t) + 3/(16a)·e^(-t/4a)\n"
        f"Дельта-функция означает прямую связь (мгновенный проскок) 1/4 от входа.\n\n"
    )

    if input_type == 'step':
        info += (
            "При ступеньке x(t) = σ(t):\n"
            "  y(t) = 1/4 + 3/4·(1 - e^(-t/τ))\n"
            "В момент t=0 выход мгновенно подскакивает до 1/4 (из-за дельты),\n"
            "а затем плавно экспоненциально нарастает до 1."
        )
    else:
        info += (
            "При рампе x(t) = k·t:\n"
            "  y(t) = 1/4·k·t + 3/4·k·(t - τ(1 - e^(-t/τ)))\n"
            "Наклон в установившемся режиме равен k (как у входа),\n"
            "но выход отстаёт на 3/4·τ."
        )

    return fig, info

# =============================================================================
# CALLBACK 22: Три метода (Дюамель vs ОДУ vs Аналитика)
# =============================================================================
@app.callback(
    Output('plot-three-methods', 'figure'),
    Output('plot-three-methods-error', 'figure'),
    Output('three-methods-info', 'children'),
    Input('tm-a', 'value'),
    Input('tm-k', 'value'),
)
def update_three_methods(a_ms, k):
    # Параметры системы
    a = a_ms * 1e-3
    tau = 4 * a
    t_end = 10 * tau
    dt = tau / 200
    t = np.arange(0, t_end, dt)

    # 1. Аналитика
    y_analytic = k * (t - 3 * a * (1 - np.exp(-t / tau)))

    # 2. ОДУ (solve_ivp)
    def rhs(t, y):
        return [(k*t + a*k - y[0]) / tau]

    sol = solve_ivp(rhs, [0, t_end], [0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    y_ode = sol.y[0]

    # 3. Интеграл Дюамеля (свёртка)
    # h_reg(t) = (3/(16a)) * exp(-t/(4a))
    h_reg = (3 / (16 * a)) * np.exp(-t / tau)

    # Численная свёртка
    y_duhamel = np.convolve(k*t, h_reg) * dt
    y_duhamel = y_duhamel[:len(t)] + k*t/4  # добавляем вклад от дельты

    # --- График 1: Все три метода ---
    fig_methods = go.Figure()

    # Аналитика (толстая линия)
    fig_methods.add_trace(go.Scatter(
        x=t/tau, y=y_analytic, mode='lines',
        name='Аналитика',
        line=dict(color='#d62728', width=4)
    ))

    # ОДУ (пунктир)
    fig_methods.add_trace(go.Scatter(
        x=t/tau, y=y_ode, mode='lines',
        name='ОДУ (scipy)',
        line=dict(color='#1f77b4', width=2, dash='dot')
    ))

    # Дюамель (штрих-пунктир)
    fig_methods.add_trace(go.Scatter(
        x=t/tau, y=y_duhamel, mode='lines',
        name='Дюамель (свёртка)',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))

    fig_methods.update_layout(
        title='Реакция на рампу: три метода, один ответ',
        xaxis_title='t / τ',
        yaxis_title='y(t)',
        template='plotly_white', height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- График 2: Ошибки ---
    err_ode = np.abs(y_ode - y_analytic)
    err_duh = np.abs(y_duhamel - y_analytic)

    fig_error = go.Figure()

    fig_error.add_trace(go.Scatter(
        x=t/tau, y=err_ode, mode='lines',
        name='|ОДУ − Аналитика|',
        line=dict(color='#1f77b4', width=2)
    ))

    fig_error.add_trace(go.Scatter(
        x=t/tau, y=err_duh, mode='lines',
        name='|Дюамель − Аналитика|',
        line=dict(color='#2ca02c', width=2)
    ))

    fig_error.update_layout(
        title='Абсолютная ошибка методов относительно аналитики',
        xaxis_title='t / τ',
        yaxis_title='Ошибка',
        yaxis=dict(type='log'), # Логарифмическая шкала
        template='plotly_white', height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Текстовая информация ---
    max_err_ode = np.max(err_ode)
    max_err_duh = np.max(err_duh)

    info = (
        f"Параметры: a = {a_ms:.2f} мс, τ = 4a = {4*a_ms:.2f} мс, k = {k:.2f} В/с\n\n"
        f"Максимальная ошибка (относительно аналитики):\n"
        f"  Метод ОДУ (solve_ivp):       {max_err_ode:.2e}\n"
        f"  Метод Дюамеля (свёртка):     {max_err_duh:.2e}\n\n"
        f"Обратите внимание, что ошибка Дюамеля в основном сосредоточена в самом начале (t ≈ 0),\n"
        f"где происходит резкий скачок от дельта-функции, и численное интегрирование\n"
        f"не может мгновенно отследить этот скачок с идеальной точностью."
    )

    return fig_methods, fig_error, info

# =============================================================================
# CALLBACK 23: БПФ от h(t) (Связь времени и частоты)
# =============================================================================
@app.callback(
    Output('plot-fft-impulse', 'figure'),
    Output('fft-impulse-info', 'children'),
    Input('fft-imp-a', 'value'),
)
def update_fft_impulse(a_ms):
    a = a_ms * 1e-3
    tau = 4 * a

    # Параметры временной сетки (такие же, как в Дюамеле)
    t_end = 10 * tau
    dt = tau / 200
    t = np.arange(0, t_end, dt)

    # 1. Регулярная часть импульсной характеристики
    h_reg = (3 / (16 * a)) * np.exp(-t / tau)

    # 2. Численное БПФ
    # Добавляем 0.25 (спектр дельта-функции) к результату БПФ
    H_num = np.fft.fft(h_reg) * dt + 0.25
    w_ax = 2 * np.pi * np.fft.fftfreq(len(h_reg), dt)

    # Берем только положительные частоты
    m = w_ax > 0
    w_pos = w_ax[m]
    H_num_pos = H_num[m]

    # 3. Аналитическая частотная характеристика
    H_exact = (1 + 1j * w_pos * a) / (1 + 1j * w_pos * 4 * a)

    # --- График (АЧХ и ФЧХ) ---
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('АЧХ: |H(jω)| (в дБ)', 'ФЧХ: ∠H(jω) (в градусах)'))

    # АЧХ (в дБ)
    mag_num_db = 20 * np.log10(np.abs(H_num_pos))
    mag_exact_db = 20 * np.log10(np.abs(H_exact))

    fig.add_trace(go.Scatter(x=w_pos, y=mag_num_db, mode='lines',
                             name='БПФ h(t) (численно)',
                             line=dict(color='#1f77b4', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w_pos, y=mag_exact_db, mode='lines',
                             name='H(jω) (теория)',
                             line=dict(color='#d62728', width=2, dash='dash')), row=1, col=1)

    # ФЧХ (в градусах)
    # Используем np.unwrap для устранения скачков фазы на 360 градусов
    phase_num = np.degrees(np.unwrap(np.angle(H_num_pos)))
    phase_exact = np.degrees(np.unwrap(np.angle(H_exact)))

    fig.add_trace(go.Scatter(x=w_pos, y=phase_num, mode='lines',
                             name='БПФ h(t) (численно)',
                             line=dict(color='#2ca02c', width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=w_pos, y=phase_exact, mode='lines',
                             name='H(jω) (теория)',
                             line=dict(color='#9467bd', width=2, dash='dash'), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="ω (рад/с)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="дБ", row=1, col=1)

    fig.update_xaxes(title_text="ω (рад/с)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Градусы", row=1, col=2)

    fig.update_layout(
        title='Спектр импульсной характеристики = Частотная характеристика',
        template='plotly_white', height=450,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    # --- Текстовая информация ---
    info = (
        f"Параметр a = {a_ms:.1f} мс, τ = 4a = {4*a_ms:.1f} мс\n\n"
        f"h(t) = 1/4·δ(t) + 3/(16a)·e^(-t/4a)\n\n"
        f"БПФ регулярной части (экспоненты) даёт дробь (3/4)/(1+j4aω).\n"
        f"Дельта-функция δ(t) даёт константу 1/4 на всех частотах.\n"
        f"Сумма: H(jω) = 1/4 + (3/4)/(1+j4aω) = (1 + jaω)/(1 + j4aω)\n\n"
        f"Обратите внимание: численное БПФ (толстые линии) идеально\n"
        f"накладывается на аналитическую формулу (пунктир) во всём\n"
        f"диапазоне частот. Это доказывает, что преобразование Фурье\n"
        f"от импульсной характеристики есть передаточная функция."
    )

    return fig, info

# =============================================================================
# CALLBACK 24: FFT Blur (Скользящее среднее)
# =============================================================================
@app.callback(
    Output('plot-fft-blur', 'figure'),
    Output('blur-info', 'children'),
    Input('blur-window', 'value'),
    Input('blur-noise', 'value'),
    Input('blur-period', 'value'),
)
def update_fft_blur(window, noise_amp, period):
    rng = np.random.default_rng(7)
    N = 99

    # 1. Генерация сигнала
    # Формула из исходного кода, но с динамическим периодом и амплитудой шума
    f_sig = (np.mod(np.arange(1, N+1), period) - period/4) / 8 + noise_amp * rng.standard_normal(N)
    f_sig[:N//10] = 0
    f_sig[-N//10:] = 0

    # 2. Создание импульсной характеристики (окно)
    h = np.zeros(N)
    start_idx = N//2 - window//2
    end_idx = start_idx + window
    h[start_idx:end_idx] = 1.0 / window

    # 3. Фильтрация через БПФ
    F = np.fft.fft(f_sig)
    H = np.fft.fft(h)
    f_blur = np.fft.ifftshift(np.fft.ifft(F * H)).real

    # Частотные оси для отображения
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    F_shift = np.fft.fftshift(F)
    H_shift = np.fft.fftshift(H)

    # --- Сборка фигуры (2x2 сетка) ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Временная область: Сигнал до и после фильтрации',
            'Спектр мощности: |F(jω)| и |F(jω)·H(jω)|',
            f'Импульсная характеристика фильтра h[n] (окно {window})',
            'АЧХ фильтра |H(jω)|'
        )
    )

    # 1. Временная область
    fig.add_trace(go.Scatter(x=np.arange(N), y=f_sig, mode='lines', name='Исходный (пила+шум)',
                             line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(N), y=f_blur, mode='lines', name='Отфильтрованный',
                             line=dict(color='#d62728', width=2)), row=1, col=1)

    # 2. Спектр
    fig.add_trace(go.Scatter(x=freqs, y=np.abs(F_shift), mode='lines', name='|F| (до)',
                             line=dict(color='#1f77b4', width=1.5, dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(x=freqs, y=np.abs(F_shift * H_shift), mode='lines', name='|F·H| (после)',
                             line=dict(color='#d62728', width=2)), row=1, col=2)

    # 3. Импульсная характеристика (используем маркеры, чтобы показать дискретность)
    fig.add_trace(go.Scatter(x=np.arange(N), y=h, mode='lines+markers', name='h[n]',
                             line=dict(color='#2ca02c', width=2),
                             marker=dict(size=4)), row=2, col=1)

    # 4. АЧХ фильтра
    fig.add_trace(go.Scatter(x=freqs, y=np.abs(H_shift), mode='lines', name='|H|',
                             line=dict(color='#9467bd', width=2)), row=2, col=2)

    fig.update_layout(
        template='plotly_white',
        height=650,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    fig.update_xaxes(title_text="n (отсчет)", row=1, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=1)

    fig.update_xaxes(title_text="Нормированная частота", row=1, col=2)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=2)

    fig.update_xaxes(title_text="n (отсчет)", row=2, col=1)
    fig.update_yaxes(title_text="h[n]", row=2, col=1)

    fig.update_xaxes(title_text="Нормированная частота", row=2, col=2)
    fig.update_yaxes(title_text="|H|", row=2, col=2)

    # --- Текстовая сводка ---
    info = (
        f"Окно: {window} отсчетов, Амплитуда шума: {noise_amp:.2f}, Период пилы: {period}\n\n"
        f"Скользящее среднее — это простейший цифровой фильтр нижних частот.\n"
        f"Обратите внимание на АЧХ (правый нижний график): у неё есть характерные нули.\n"
        f"Эти нули приходятся на частоты, где период синусоиды кратен ширине окна.\n\n"
        f"Попробуйте увеличить окно (например, до 9 или 11). Вы увидите, как пила становится\n"
        f"более гладкой, а в спектре высокочастотный шум подавляется сильнее."
    )

    return fig, info

# =============================================================================
# CALLBACK 25: Свёртка vs Спектр
# =============================================================================
@app.callback(
    Output('plot-conv-spec', 'figure'),
    Output('conv-spec-info', 'children'),
    Input('conv-window', 'value'),
    Input('conv-noise', 'value'),
    Input('conv-period', 'value'),
    Input('conv-zero-edges', 'value'),
)
def update_conv_spec(w_sel, noise_amp, period, zero_edges):
    rng = np.random.default_rng(7)
    N = 99

    # 1. Генерация сигнала
    f_sig = (np.mod(np.arange(1, N+1), period) - period/4) / 8 + noise_amp * rng.standard_normal(N)
    if zero_edges == 'yes':
        f_sig[:N//10] = 0
        f_sig[-N//10:] = 0

    # --- Сборка фигуры ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Свёртка во времени: больше окно — сильнее blur',
            'Два пути для выбранного окна: линейная свёртка vs F⁻¹[X·H]'
        ),
        vertical_spacing=0.15
    )

    # Верхний график: разные окна
    fig.add_trace(go.Scatter(
        x=np.arange(N), y=f_sig, mode='lines', name='Исходный (пила+шум)',
        line=dict(color='gray', width=1, dash='dot'), opacity=0.6
    ), row=1, col=1)

    for w_ in (3, 7, 15):
        hw = np.zeros(N)
        hw[N//2 - w_//2 : N//2 - w_//2 + w_] = 1.0 / w_
        y_conv_w = np.convolve(f_sig, hw, mode='same')
        fig.add_trace(go.Scatter(
            x=np.arange(N), y=y_conv_w, mode='lines', name=f'Окно {w_}',
            line=dict(width=1.5)
        ), row=1, col=1)

    # Нижний график: выбранное окно (w_sel)
    h_sel = np.zeros(N)
    start = N//2 - w_sel//2
    h_sel[start : start + w_sel] = 1.0 / w_sel

    # Линейная свёртка
    y_conv_sel = np.convolve(f_sig, h_sel, mode='same')

    # Циклическая свёртка через БПФ
    # ifftshift компенсирует сдвиг центрированного окна h_sel
    y_spec = np.fft.ifftshift(np.fft.ifft(np.fft.fft(f_sig) * np.fft.fft(h_sel))).real

    fig.add_trace(go.Scatter(
        x=np.arange(N), y=y_conv_sel, mode='lines+markers', name='Свёртка (линейная)',
        line=dict(color='#1f77b4', width=2), marker=dict(size=3)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(N), y=y_spec, mode='lines', name='F⁻¹[X·H] (циклическая)',
        line=dict(color='#d62728', width=2, dash='dash')
    ), row=2, col=1)

    # Оформление
    fig.update_layout(
        template='plotly_white',
        height=550,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(title_text="n (отсчет)", row=1, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=1, col=1)
    fig.update_xaxes(title_text="n (отсчет)", row=2, col=1)
    fig.update_yaxes(title_text="Амплитуда", row=2, col=1)

    # --- Текст ---
    max_err = np.max(np.abs(y_conv_sel - y_spec))

    info = f"Размер окна (нижний график): {w_sel}\n"
    if zero_edges == 'yes':
        info += (
            "Края обнулены (как в исходном скрипте) — поэтому циклическая свёртка\n"
            "идеально совпадает с линейной. Краевые эффекты не успевают «завернуться».\n"
        )
    else:
        info += (
            "Края НЕ обнулены — циклическая свёртка «заворачивается» через границу\n"
            "кадра (свойство ДПФ). Вы увидите, что на краях графики расходятся.\n"
        )

    info += f"\nМакс. |линейная − циклическая| = {max_err:.2e}"

    return fig, info

def main():
    # Открываем браузер только если мы НЕ в Docker
    if not os.environ.get('DOCKER_ENV'):
        def open_browser():
            time.sleep(1.0)
            webbrowser.open_new('http://127.0.0.1:8050/')
        threading.Thread(target=open_browser, daemon=True).start()
    else:
        print("\n" + "="*60)
        print("🚀 Math Visual Lab запущен в Docker!")
        print("👉 Откройте в браузере: http://localhost:8050")
        print("="*60 + "\n")

    # host='0.0.0.0' нужен, чтобы к приложению можно было подключиться извне (в Docker)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()


