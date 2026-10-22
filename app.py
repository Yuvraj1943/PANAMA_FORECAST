import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""Panama load-forecast dashboard (Streamlit). Run: streamlit run app.py
Tabs: Forecast viewer | Upload & predict | Model evaluation | Explainability."""
import json, io
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from data import FEATURES, TARGET
from metrics import summary

st.set_page_config(page_title='Panama Load Forecast', layout='wide')
COLORS = {'CND': '#B4452F', 'XGBoost_paper': '#0F766E', 'XGBoost': '#5AA9A2', 'XGBoost_tuned': '#2DA39A', 'LSTM': '#B7791F', 'LinearRegression': '#6B7280', 'Persistence(wk-2)': '#A3A3A3'}
LABELS = {'CND': 'CND (official)', 'XGBoost_paper': 'XGBoost (published settings)', 'XGBoost': 'XGBoost (hand-set)', 'XGBoost_tuned': 'XGBoost (Optuna-tuned)',
          'LSTM': 'LSTM', 'LinearRegression': 'Linear Regression', 'Persistence(wk-2)': 'Persistence (2 wks ago)'}


@st.cache_data
def load():
    P = pd.read_csv('predictions_final.csv', parse_dates=['datetime'])
    F = pd.read_csv('shap_features.csv')
    assert len(P) == len(F)
    return P, F, json.load(open('summary.json')), pd.read_csv('shap_importance.csv', index_col=0).iloc[:, 0]


P, F, S, IMP = load()
st.title('Panama national electricity demand - weekly forecast')
st.caption('Walk-forward evaluation on 14 weekly test folds (Apr 2019 - Jun 2020), benchmarked against CND\'s official pre-dispatch forecast.')
tab1, tab2, tab3, tab4 = st.tabs(['Forecast viewer', 'Upload & predict', 'Model evaluation', 'Explainability'])

with tab1:
    c1, c2 = st.columns([1, 2])
    fold = c1.selectbox('Test week', list(P['fold'].unique()), index=12)
    chosen = c1.multiselect('Models to show', [k for k in LABELS if k != 'CND'], default=['XGBoost_paper', 'LSTM'], format_func=lambda k: LABELS[k])
    d = P[P['fold'] == fold]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=d['datetime'], y=d[TARGET], name='Actual', line=dict(color='#1B2430', width=3)))
    fig.add_trace(go.Scatter(x=d['datetime'], y=d['CND'], name=LABELS['CND'], line=dict(color=COLORS['CND'], width=2)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=d['datetime'], y=d[m], name=LABELS[m], line=dict(color=COLORS[m], width=2)))
    fig.update_layout(height=420, yaxis_title='Demand (MWh)', margin=dict(t=20), legend=dict(orientation='h'))
    c2.plotly_chart(fig, width='stretch')
    rows = {LABELS[m]: summary(d[TARGET], d[m]) for m in ['CND'] + chosen}
    c1.write('**Accuracy this week**'); c1.dataframe(pd.DataFrame(rows).T.round(2))
    c1.download_button('Download this week (CSV)', d.to_csv(index=False), file_name=f'{fold}.csv')

with tab2:
    st.write('**Data input -> prediction.** Upload hourly rows containing the model features and get demand forecasts '
             'from the XGBoost model with the published settings (trained on data up to Jun 2020).')
    st.write('Required columns: ', ', '.join(FEATURES))
    sample = pd.concat([P[['datetime']].iloc[-168:].reset_index(drop=True), F.iloc[-168:].reset_index(drop=True)], axis=1)
    st.download_button('Download a sample input file (last test week)', sample.to_csv(index=False), file_name='sample_input.csv')
    up = st.file_uploader('Upload CSV', type='csv')
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        elif df[FEATURES].isna().any().any():
            st.error('Input contains missing values - please fill or remove those rows.')
        else:
            model = joblib.load('xgb_published_settings_last_week.joblib')
            df['predicted_demand_MWh'] = model.predict(df[FEATURES]).round(2)
            x = df['datetime'] if 'datetime' in df.columns else df.index
            f2 = go.Figure(go.Scatter(x=x, y=df['predicted_demand_MWh'], name='Predicted', line=dict(color=COLORS['XGBoost_paper'])))
            f2.update_layout(height=380, yaxis_title='Predicted demand (MWh)', margin=dict(t=20)); st.plotly_chart(f2, width='stretch')
            st.dataframe(df.head(48)); st.download_button('Download predictions', df.to_csv(index=False), file_name='predictions.csv')

with tab3:
    ov = pd.DataFrame(S['overall']).T.rename(columns=lambda c: c)
    st.subheader('Overall (14 test weeks, 2,352 hours)'); st.dataframe(ov.round(2))
    st.caption(f"Weeks where each model beat CND: {S['folds_better_than_CND']}")
    seg = st.selectbox('Situation', list(S['segments'].keys()))
    st.write(f"{S['segment_hours'][seg]} hours"); st.dataframe(pd.DataFrame(S['segments'][seg]).T.round(2))
    c1, c2 = st.columns(2); c1.image('per_fold_mape.png'); c2.image('segments.png')

with tab4:
    st.subheader('What drives the XGBoost forecast (mean |SHAP|)')
    c1, c2 = st.columns(2); c1.image('shap_importance.png'); c2.image('shap_beeswarm.png')
    st.write(IMP.sort_values(ascending=False).round(2).to_frame('mean |SHAP| (MWh)'))
