"""
Streamlit app: Firefighter Metabolism Calculator
File: streamlit_metabolism_calculator.py

Features:
- Input firefighter body mass, one or more equipment weights (comma-separated), VO2 baseline, factor per kg, rest VO2.
- Define a duty-cycle sequence as a list of minutes (work/rest/work/rest...) starting with work or rest.
  Example sequence: "68,15,68" with "Starts with work" = True
- Computes per-scenario: instantaneous VO2 pattern (L/min), cumulative O2 (L), total kcal, kcal/min, percent savings vs baseline.
- Shows results in an interactive table, plots (instantaneous VO2 & cumulative O2) and offers PDF export (download button).

Deployment:
- Put this file in a GitHub repo and create a requirements.txt with the packages shown below.
- Connect the repo to Streamlit Cloud (share.streamlit.io) and deploy.

Recommended requirements.txt:
streamlit
pandas
numpy
matplotlib
fpdf2

Notes:
- This is a self-contained single-file app intended for demonstration and small-scale use.
- You can easily adapt units and defaults to match local measured VO2 values.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64

st.set_page_config(page_title="Firefighter Metabolism Calculator", layout="wide")

st.title("מחושב מטבוליזם לכבאים — מחשבון צריכת אנרגיה")
st.write("הזן את הפרמטרים שלך בצד שמאל ולחץ 'חשב' כדי לראות טבלאות, גרפים וייצוא PDF.")

# --- Sidebar inputs ---
with st.sidebar.form(key='inputs'):
    st.header("קלטים")
    body_mass = st.number_input("משקל הכבאי (kg)", min_value=40.0, max_value=200.0, value=80.0, step=1.0)
    eq_weights_text = st.text_input("משקל ציוד (kg) - רשימת אופציות מופרדות בפסיקים","19.9,15.1,12.67")
    vo2_ml_kg_min = st.number_input("VO₂ עבודה (ml/kg/min) - ברירת מחדל לדוגמה", min_value=5.0, max_value=80.0, value=35.0)
    factor_ml_min_per_kg = st.number_input("פקטור שינוי VO₂ per kg (ml O₂·min⁻¹·kg⁻¹)", value=33.5)
    rest_vo2 = st.number_input("VO₂ בזמן מנוחה בתוך הציוד (L/min)", value=0.8)
    kcal_per_L = st.number_input("ק״ק ליטר חמצן (kcal/L)", value=5.0)

    st.markdown("---")
    st.markdown("**הגדרת דפוס עבודה/מנוחה**")
    seq_text = st.text_input("רשום רצף דקות מופרדות בפסיקים (לדוג': 68,15,68) ", value="68,15,68")
    starts_with_work = st.checkbox("הסדר מתחיל בעבודה (אם לא - מתחיל במנוחה)", value=True)

    st.markdown("---")
    st.markdown("**הגדרות מתקדמות (אופציונלי)**")
    # allow user to select whether multiple weights should be compared
    export_pdf_name = st.text_input("שם קובץ ה-PDF להורדה", value="metabolism_report.pdf")

    compute_btn = st.form_submit_button("חשב")

# Helper functions

def parse_weights(text):
    items = [s.strip() for s in text.split(',') if s.strip()!='']
    weights=[]
    for it in items:
        try:
            weights.append(float(it))
        except:
            pass
    return weights


def parse_sequence(text, starts_with_work=True):
    # returns list of tuples ('work'/'rest', minutes)
    parts = [p.strip() for p in text.split(',') if p.strip()!='']
    seq=[]
    kinds = []
    if starts_with_work:
        kind_order = ['work','rest']
    else:
        kind_order = ['rest','work']
    for i,p in enumerate(parts):
        try:
            m = int(float(p))
        except:
            m = 0
        kind = kind_order[i%2]
        seq.append((kind, m))
    return seq


def make_pattern_from_sequence(seq, vo2_work, total_T=None, rest_vo2=0.8):
    # build minute-by-minute pattern
    total_minutes = sum(duration for _,duration in seq)
    if total_T is None:
        total_T = total_minutes
    pattern = np.zeros(total_T+1)
    minute = 0
    for kind,dur in seq:
        for i in range(dur):
            if minute > total_T:
                break
            if kind=='work':
                pattern[minute] = vo2_work
            else:
                pattern[minute] = rest_vo2
            minute += 1
    # if anything remains fill with rest_vo2
    while minute <= total_T:
        pattern[minute] = rest_vo2
        minute+=1
    return pattern


def vo2_from_weights(base_vo2_L_min, base_weight, target_weight, factor_L_per_min_per_kg):
    delta = base_weight - target_weight
    return max(0.1, base_vo2_L_min - factor_L_per_min_per_kg * delta)


# Main compute block
if compute_btn:
    eq_weights = parse_weights(eq_weights_text)
    if len(eq_weights)==0:
        st.error("אנא הכנס לפחות משקל ציוד אחד תקין.")
        st.stop()

    seq = parse_sequence(seq_text, starts_with_work=starts_with_work)
    if sum(d for _,d in seq)==0:
        st.error("רצף הדקות שהזנת ריק/לא תקין.")
        st.stop()

    # convert VO2 baseline to L/min
    baseline_vo2_L_min = vo2_ml_kg_min * body_mass / 1000.0
    factor_L_per_min_per_kg = factor_ml_min_per_kg / 1000.0

    # We'll compute results for each equipment weight
    results = []
    patterns = {}
    cumulative = {}

    # Determine total length for plotting (max of sequences lengths)
    seq_total = sum(d for _,d in seq)
    max_total = seq_total

    for w in eq_weights:
        # assume baseline corresponds to first listed weight? We'll compute VO2 relative to baseline_vo2 as if baseline weight is the heaviest in the list
        # For simplicity, treat baseline_vo2 as VO2 for the first weight in the list if multiple given. Otherwise baseline_vo2 is the provided baseline.
        # To avoid ambiguity, we'll compute vo2 for each w by reducing from the provided baseline_vo2 proportional to delta from the heaviest weight present.
        # find heaviest weight to treat as reference
        heaviest = max(eq_weights)
        # compute per-weight VO2
        vo2_w = vo2_from_weights(baseline_vo2_L_min, heaviest, w, factor_L_per_min_per_kg)
        # build pattern
        pat = make_pattern_from_sequence(seq, vo2_w, total_T=max_total, rest_vo2=rest_vo2)
        cum = np.cumsum(pat)
        total_O2 = float(cum[-1])
        total_kcal = total_O2 * kcal_per_L
        kcal_per_min = total_kcal / max_total
        results.append({
            'equipment_weight_kg': w,
            'vo2_work_L_min': round(vo2_w,4),
            'total_minutes': max_total,
            'total_O2_L': round(total_O2,3),
            'total_kcal': round(total_kcal,2),
            'kcal_per_min': round(kcal_per_min,3)
        })
        patterns[w] = pat
        cumulative[w] = cum

    df_res = pd.DataFrame(results).sort_values(by='equipment_weight_kg', ascending=False)

    # compute percent savings relative to heaviest (baseline reference)
    baseline_total_kcal = df_res['total_kcal'].max()
    df_res['pct_saving_vs_heaviest_%'] = df_res['total_kcal'].apply(lambda x: round((baseline_total_kcal - x)/baseline_total_kcal*100.0,3))

    # Show table
    st.subheader("טבלת תוצאות")
    st.dataframe(df_res.reset_index(drop=True))

    # Plots: instantaneous VO2 and cumulative O2
    st.subheader("גרפים")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    for w, pat in patterns.items():
        ax1.plot(np.arange(len(pat)), pat, label=f'{w} kg')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('VO2 (L/min)')
    ax1.set_title('Instantaneous VO2 pattern')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    for w, cum in cumulative.items():
        ax2.plot(np.arange(len(cum)), cum, label=f'{w} kg')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Cumulative O2 (L)')
    ax2.set_title('Cumulative O2 consumed')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Prepare PDF export: render table and figures into a simple PDF
    def create_pdf(df, fig1, fig2, seq_text, starts_with_work, body_mass, eq_weights_text, vo2_ml_kg_min):
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, 'Metabolism Report - Firefighter', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', size=10)
        pdf.multi_cell(0, 6, f'Parameters: body_mass={body_mass} kg | equipment_weights={eq_weights_text} | VO2_work={vo2_ml_kg_min} ml/kg/min | sequence={seq_text} | starts_with_work={starts_with_work}')
        pdf.ln(4)

        # Table: we'll render as text
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Summary table:', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', size=9)
        # header
        col_w = [35,30,30,30,30]
        hdrs = ['Equipment (kg)', 'VO2 L/min', 'Total O2 (L)', 'Total kcal', 'kcal/min']
        for i,h in enumerate(hdrs):
            pdf.cell(col_w[i], 6, h, border=1)
        pdf.ln()
        for _,row in df.iterrows():
            pdf.cell(col_w[0],6,str(row['equipment_weight_kg']),border=1)
            pdf.cell(col_w[1],6,str(row['vo2_work_L_min']),border=1)
            pdf.cell(col_w[2],6,str(row['total_O2_L']),border=1)
            pdf.cell(col_w[3],6,str(row['total_kcal']),border=1)
            pdf.cell(col_w[4],6,str(row['kcal_per_min']),border=1)
            pdf.ln()

        # Add figures as images saved to memory
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0,6, 'Instantaneous VO2 pattern', ln=True)
        img_buf = BytesIO()
        fig1.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        pdf.image(img_buf, x=15, y=None, w=180)
        img_buf.close()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0,6, 'Cumulative O2 consumed', ln=True)
        img_buf2 = BytesIO()
        fig2.savefig(img_buf2, format='png', dpi=150, bbox_inches='tight')
        img_buf2.seek(0)
        pdf.image(img_buf2, x=15, y=None, w=180)
        img_buf2.close()

        out = BytesIO()
        pdf.output(out)
        out.seek(0)
        return out

    pdf_bytes_io = create_pdf(df_res, fig1, fig2, seq_text, starts_with_work, body_mass, eq_weights_text, vo2_ml_kg_min)

    st.download_button(label='הורד PDF של הדוח', data=pdf_bytes_io.getvalue(), file_name=export_pdf_name, mime='application/pdf')

    st.success('החישוב הושלם — תוכל להוריד את הדוח ב-PDF או לשנות פרמטרים ולהריץ שוב.')

else:
    st.info('הכנס קלטים בסיידבר ולחץ "חשב".')
