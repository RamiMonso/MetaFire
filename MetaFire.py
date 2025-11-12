"""
Streamlit app: Firefighter Metabolism Calculator (multi-option)

This improved version lets you define multiple equipment options, each with its own weight and its own duty-cycle sequence.
Input format for options (one per line):
    Label:weight_kg:sequence
Example:
    Baseline:19.9:68,15,68
    Option A:15.1:45,15,45,15,45
    Option B:12.67:45,15,45,15,45

Sequence is a comma-separated list of minutes. The sequence starts with WORK by default (you can toggle).

Outputs:
- Table with per-option VO2 (L/min), total O2 (L), total kcal, kcal/min, percent savings vs baseline (heaviest option).
- Instantaneous VO2 plot and cumulative O2 plot.
- PDF export including table and plots.

Deployment:
- Save this file as streamlit_metabolism_calculator_multi.py in a GitHub repo.
- requirements.txt should contain:
    streamlit
    pandas
    numpy
    matplotlib
    fpdf2

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Metabolism Calculator — Multi-option", layout="wide")
st.title("מחשבון מטבוליזם לכבאים — גרסה מרובת אפשרויות")
st.write("הזן כל שורה בתיבת הקלט בצד שמאל כאופציה נפרדת בפורמט: `Label:weight_kg:sequence`\nלדוגמה: `Baseline:19.9:68,15,68`\nלחץ 'חשב' כדי לראות טבלאות, גרפים וייצוא PDF.")

with st.sidebar.form(key='inputs'):
    st.header("קלטים כלליים")
    body_mass = st.number_input("משקל הכבאי (kg)", min_value=40.0, max_value=200.0, value=80.0, step=1.0)
    vo2_ml_kg_min = st.number_input("VO₂ עבודה (ml/kg/min)", min_value=5.0, max_value=80.0, value=35.0)
    factor_ml_min_per_kg = st.number_input("פקטור שינוי VO₂ per kg (ml O₂·min⁻¹·kg⁻¹)", value=33.5)
    rest_vo2 = st.number_input("VO₂ בזמן מנוחה בתוך הציוד (L/min)", value=0.8)
    kcal_per_L = st.number_input("ק״ק ליטר חמצן (kcal/L)", value=5.0)
    starts_with_work = st.checkbox("הרצף בכל אופציה מתחיל בעבודה (אם לא, מתחיל במנוחה)", value=True)
    st.markdown('---')
    st.markdown("**הזן אופציות (שורה/אופציה)**")
    st.markdown("פורמט: Label:weight_kg:sequence (sequence = ví—comma separated minutes).\nדוגמה:\nBaseline:19.9:68,15,68")
    options_text = st.text_area("אפשרויות (שורה לכל אופציה)", value='Baseline:19.9:68,15,68\nOption A:15.1:45,15,45,15,45\nOption B:12.67:45,15,45,15,45', height=180)
    export_pdf_name = st.text_input("שם קובץ ה-PDF להורדה", value="metabolism_report_multi.pdf")
    compute_btn = st.form_submit_button("חשב")

# Helper functions

def parse_option_line(line, default_starts_with_work=True):
    # expected: Label:weight:seq
    parts = [p.strip() for p in line.split(':')]
    if len(parts) < 3:
        return None
    label = parts[0]
    try:
        weight = float(parts[1])
    except:
        return None
    seq_text = parts[2]
    # allow additional colons only in label by joining
    if len(parts) > 3:
        seq_text = ':'.join(parts[2:])
    seq = parse_sequence(seq_text, default_starts_with_work)
    return {'label': label, 'weight': weight, 'sequence': seq}


def parse_sequence(text, starts_with_work=True):
    parts = [p.strip() for p in text.split(',') if p.strip()!='']
    seq = []
    if starts_with_work:
        kind_order = ['work', 'rest']
    else:
        kind_order = ['rest', 'work']
    for i,p in enumerate(parts):
        try:
            m = int(float(p))
        except:
            m = 0
        kind = kind_order[i%2]
        seq.append((kind, m))
    return seq


def make_pattern_from_sequence(seq, vo2_work, total_T=None, rest_vo2=0.8):
    total_minutes = sum(duration for _,duration in seq)
    if total_T is None:
        total_T = total_minutes
    pattern = np.zeros(total_T+1)
    minute = 0
    for kind,dur in seq:
        for i in range(dur):
            if minute > total_T:
                break
            pattern[minute] = vo2_work if kind=='work' else rest_vo2
            minute += 1
    while minute <= total_T:
        pattern[minute] = rest_vo2
        minute+=1
    return pattern


def vo2_from_weights(base_vo2_L_min, ref_weight, target_weight, factor_L_per_min_per_kg):
    delta = ref_weight - target_weight
    return max(0.05, base_vo2_L_min - factor_L_per_min_per_kg * delta)

# Main compute
if compute_btn:
    # parse options
    lines = [l for l in options_text.splitlines() if l.strip()!='']
    parsed = [parse_option_line(l, default_starts_with_work=starts_with_work) for l in lines]
    parsed = [p for p in parsed if p is not None]
    if len(parsed) == 0:
        st.error('לא נמצאו אופציות תקינות — בדוק את קלט הטקסט לפי הפורמט.')
        st.stop()

    # convert base VO2 ml/kg/min -> L/min for a reference weight we'll choose as the heaviest weight
    weights = [p['weight'] for p in parsed]
    heaviest = max(weights)
    base_vo2_L_min = vo2_ml_kg_min * body_mass / 1000.0
    factor_L_per_min_per_kg = factor_ml_min_per_kg / 1000.0

    # determine max total minutes among sequences for plotting alignment
    seq_totals = [sum(d for _,d in p['sequence']) for p in parsed]
    max_total = max(seq_totals)

    results = []
    patterns = {}
    cumulative = {}

    for p in parsed:
        w = p['weight']
        label = p['label']
        seq = p['sequence']
        vo2_w = vo2_from_weights(base_vo2_L_min, heaviest, w, factor_L_per_min_per_kg)
        pat = make_pattern_from_sequence(seq, vo2_w, total_T=max_total, rest_vo2=rest_vo2)
        cum = np.cumsum(pat)
        total_O2 = float(cum[-1])
        total_kcal = total_O2 * kcal_per_L
        kcal_per_min = total_kcal / max_total
        results.append({
            'Label': label,
            'Weight_kg': w,
            'VO2_work_L_min': round(vo2_w,4),
            'Total_minutes': max_total,
            'Total_O2_L': round(total_O2,3),
            'Total_kcal': round(total_kcal,2),
            'kcal_per_min': round(kcal_per_min,3)
        })
        patterns[label] = pat
        cumulative[label] = cum

    df_res = pd.DataFrame(results).sort_values(by='Weight_kg', ascending=False)

    # percent saving vs heaviest (baseline)
    baseline_kcal = df_res['Total_kcal'].max()
    df_res['Pct_saving_vs_heaviest_%'] = df_res['Total_kcal'].apply(lambda x: round((baseline_kcal-x)/baseline_kcal*100.0,3))

    st.subheader('תוצאות — טבלה')
    st.dataframe(df_res.reset_index(drop=True))

    # Plots
    st.subheader('גרפים')
    fig1, ax1 = plt.subplots(figsize=(9,3))
    for label, pat in patterns.items():
        ax1.plot(np.arange(len(pat)), pat, label=f'{label} ({dict((p['label'],p['weight']) for p in parsed)[label]} kg)')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('VO2 (L/min)')
    ax1.set_title('Instantaneous VO2 pattern')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(9,3))
    for label, cum in cumulative.items():
        ax2.plot(np.arange(len(cum)), cum, label=f'{label} ({dict((p['label'],p['weight']) for p in parsed)[label]} kg)')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Cumulative O2 (L)')
    ax2.set_title('Cumulative O2 consumed')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # PDF export
    def create_pdf(df_table, fig1, fig2, params_text):
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, 'Metabolism Report - Multi-option', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', size=10)
        pdf.multi_cell(0, 6, params_text)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Summary table:', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', size=9)
        col_w = [35,30,30,30,30]
        hdrs = ['Label', 'Weight', 'VO2 L/min', 'Total kcal', 'kcal/min']
        for i,h in enumerate(hdrs):
            pdf.cell(col_w[i], 6, h, border=1)
        pdf.ln()
        for _,row in df_table.iterrows():
            pdf.cell(col_w[0],6,str(row['Label']),border=1)
            pdf.cell(col_w[1],6,str(row['Weight_kg']),border=1)
            pdf.cell(col_w[2],6,str(row['VO2_work_L_min']),border=1)
            pdf.cell(col_w[3],6,str(row['Total_kcal']),border=1)
            pdf.cell(col_w[4],6,str(row['kcal_per_min']),border=1)
            pdf.ln()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0,6,'Instantaneous VO2 pattern', ln=True)
        img_buf = BytesIO()
        fig1.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        pdf.image(img_buf, x=15, y=None, w=180)
        img_buf.close()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0,6,'Cumulative O2 consumed', ln=True)
        img_buf2 = BytesIO()
        fig2.savefig(img_buf2, format='png', dpi=150, bbox_inches='tight')
        img_buf2.seek(0)
        pdf.image(img_buf2, x=15, y=None, w=180)
        img_buf2.close()

        out = BytesIO()
        pdf.output(out)
        out.seek(0)
        return out

    params_text = f'Body mass={body_mass} kg | VO2_work={vo2_ml_kg_min} ml/kg/min | factor={factor_ml_min_per_kg} ml/min/kg | rest_vo2={rest_vo2} L/min'
    pdf_io = create_pdf(df_res, fig1, fig2, params_text)
    st.download_button('הורד PDF של הדוח', data=pdf_io.getvalue(), file_name=export_pdf_name, mime='application/pdf')

    st.success('החישוב הושלם — הורד את ה-PDF או שחק עם הערכים.')

else:
    st.info('הזן אופציות בצד שמאל ולחץ "חשב".')
