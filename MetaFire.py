"""
Streamlit app: Firefighter Metabolism Calculator — Multi-option + Help

How to use:
- Save as streamlit_metabolism_calculator_multi_with_help.py
- requirements.txt should include:
    streamlit
    pandas
    numpy
    matplotlib
    fpdf2
- Deploy on Streamlit Cloud (GitHub repo -> New app -> select file).

Features added in this version:
- Define multiple options (one per line) in format: Label:weight_kg:sequence
  Example lines:
      Baseline:19.9:68,15,68
      Option A:15.1:45,15,45,15,45
- Each option can have its own weight and sequence of work/rest minutes.
- Tooltips (help text) for input fields explaining what they represent.
- A button + expander that shows a detailed explanatory table for each default parameter.
- Outputs: results table, instantaneous VO2 plot, cumulative O2 plot, PDF export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Metabolism Calculator — Multi-option (with Help)", layout="wide")
st.title("מחשבון מטבוליזם לכבאים — גרסה מרובת אפשרויות עם הסברים")
st.write("הכנס אופציות בצד שמאל (שורה לכל אופציה) בפורמט `Label:weight_kg:sequence` ואז לחץ 'חשב'.\n"
         "לדוגמה: `Baseline:19.9:68,15,68` או `Option A:15.1:45,15,45,15,45`.")

# ---------------- Sidebar inputs ----------------
with st.sidebar.form(key='inputs'):
    st.header("קלטים כלליים")
    body_mass = st.number_input(
        "משקל הכבאי (kg)",
        min_value=40.0, max_value=200.0, value=80.0, step=1.0,
        help="משקל הגוף של הכבאי. משפיע על ההוצאה האנרגטית (ערך ברירת-מחדל: 80 kg)."
    )
    vo2_ml_kg_min = st.number_input(
        "VO₂ עבודה (ml/kg/min)",
        min_value=5.0, max_value=80.0, value=35.0,
        help="צריכת חמצן לדקה לכל קילוגרם בשל מאמץ עבודה. 35 מייצג מאמץ בינוני-גבוה שמתאים למטלות כבאות."
    )
    factor_ml_min_per_kg = st.number_input(
        "פקטור שינוי VO₂ per kg (ml O₂·min⁻¹·kg⁻¹)",
        value=33.5,
        help="בכמה מיליליטר חמצן לדקה עולה/יורד צריכת החמצן עבור כל ק\"ג ציוד שמתווסף/מוסר. "
             "מחקרים מצביעים על ~33–35 ml/min לק\"ג."
    )
    rest_vo2 = st.number_input(
        "VO₂ בזמן מנוחה בתוך הציוד (L/min)",
        value=0.8,
        help="צריכת חמצן בדקת מנוחה כאשר הכבאי עדיין בחליפת המגן/מסכה — גבוהה ממנוחה מוחלטת בגלל התנגדות נשימה / חום."
    )
    kcal_per_L = st.number_input(
        "ק״ק ליטר חמצן (kcal/L)",
        value=5.0,
        help="כמה קלוריות מפיקים בממוצע מכל ליטר חמצן שנצרך. קירוב סטנדרטי: 5 kcal/L."
    )

    starts_with_work = st.checkbox(
        "הרצף בכל אופציה מתחיל בעבודה (אם לא - מתחיל במנוחה)",
        value=True,
        help="אם מסומן — כל רצף מספרי שהזנת יתחיל בפעולת 'עבודה' ולא במנוחה."
    )

    st.markdown('---')
    st.markdown("**הזן אופציות (שורה/אופציה)**")
    st.markdown("פורמט: `Label:weight_kg:sequence`  — sequence = coma-separated minutes (e.g. 68,15,68).")
    options_text = st.text_area(
        "אפשרויות (שורה לכל אופציה)",
        value='Baseline:19.9:68,15,68\nOption A:15.1:45,15,45,15,45\nOption B:12.67:45,15,45,15,45',
        height=220
    )

    export_pdf_name = st.text_input("שם קובץ ה-PDF להורדה", value="metabolism_report_multi.pdf")
    compute_btn = st.form_submit_button("חשב")

# ---------------- Help / Explanations (button + expander) ----------------
# Detailed explanation table for default parameters
help_table = pd.DataFrame([
    {
        'Parameter (עברית)': 'משקל כבאי (Body mass)',
        'Default value': '80 kg',
        'Explanation': 'משקל הגוף משפיע על הוצאת האנרגיה — ערך ממוצע ריאלי עבור לוחמי אש.'
    },
    {
        'Parameter (עברית)': 'VO₂ עבודה (ml/kg/min)',
        'Default value': '35 ml/kg/min',
        'Explanation': 'צריכת חמצן לדקת עבודה לכל ק\"ג. מייצג מאמץ בינוני-גבוה של מטלות כבאות.'
    },
    {
        'Parameter (עברית)': 'פקטור שינוי VO₂ per kg',
        'Default value': '33.5 ml·min⁻¹·kg⁻¹',
        'Explanation': 'כמה צריכת החמצן משתנה לכל ק\"ג ציוד שמתווסף — מבוסס על ספרות (≈33–35 ml/min·kg).'
    },
    {
        'Parameter (עברית)': 'VO₂ במנוחה בתוך הציוד',
        'Default value': '0.8 L/min',
        'Explanation': 'צריכת חמצן בדקת מנוחה בזמן שהכבאי עדיין עם ציוד/מסכה — גבוה יותר ממנוחה מוחלטת.'
    },
    {
        'Parameter (עברית)': 'אנרגיה ל-1L O₂',
        'Default value': '5 kcal/L',
        'Explanation': 'יחס המרה סטנדרטי: ליטר חמצן מניב כ-5 קלוריות (תלוי בתערובת פחמימות/שומן).'
    },
    {
        'Parameter (עברית)': 'Sequence starts with work?',
        'Default value': 'True',
        'Explanation': 'רוב המשימות מתחילות בעבודה - ברירת מחדל זו מתאימה ללוחמי אש שנכנסים מיד לפעולה.'
    }
])

# show help controls under main area (not inside form) so user can click at any time
st.markdown("---")
st.write("הסברים — לחץ לכפתור לקבלת הסבר מפורט על כל פרמטר:")
if 'show_help' not in st.session_state:
    st.session_state['show_help'] = False

if st.button("הצג/הסתר הסבר מפורט"):
    st.session_state['show_help'] = not st.session_state['show_help']

if st.session_state['show_help']:
    with st.expander("טבלת הסברים מפורטת לכל פרמטר", expanded=True):
        st.table(help_table)

# ---------------- Helper functions ----------------
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
    if len(parts) > 3:
        seq_text = ':'.join(parts[2:])
    seq = parse_sequence(seq_text, starts_with_work=default_starts_with_work)
    return {'label': label, 'weight': weight, 'sequence': seq}

def parse_sequence(text, starts_with_work=True):
    parts = [p.strip() for p in text.split(',') if p.strip() != '']
    seq = []
    if starts_with_work:
        kind_order = ['work', 'rest']
    else:
        kind_order = ['rest', 'work']
    for i, p in enumerate(parts):
        try:
            m = int(float(p))
        except:
            m = 0
        kind = kind_order[i % 2]
        seq.append((kind, m))
    return seq

def make_pattern_from_sequence(seq, vo2_work, total_T=None, rest_vo2=0.8):
    total_minutes = sum(duration for _, duration in seq)
    if total_T is None:
        total_T = total_minutes
    pattern = np.zeros(total_T + 1)
    minute = 0
    for kind, dur in seq:
        for i in range(dur):
            if minute > total_T:
                break
            pattern[minute] = vo2_work if kind == 'work' else rest_vo2
            minute += 1
    while minute <= total_T:
        pattern[minute] = rest_vo2
        minute += 1
    return pattern

def vo2_from_weights(base_vo2_L_min, ref_weight, target_weight, factor_L_per_min_per_kg):
    delta = ref_weight - target_weight
    return max(0.05, base_vo2_L_min - factor_L_per_min_per_kg * delta)

# ---------------- Main compute ----------------
if compute_btn:
    # parse options lines
    lines = [l for l in options_text.splitlines() if l.strip() != '']
    parsed = [parse_option_line(l, default_starts_with_work=starts_with_work) for l in lines]
    parsed = [p for p in parsed if p is not None]
    if len(parsed) == 0:
        st.error('לא נמצאו אופציות תקינות — בדוק את קלט הטקסט לפי הפורמט: Label:weight:sequence')
        st.stop()

    # convert VO2 baseline ml/kg/min -> L/min using body mass (we will compute relative to heaviest weight)
    weights = [p['weight'] for p in parsed]
    heaviest = max(weights)
    base_vo2_L_min = vo2_ml_kg_min * body_mass / 1000.0
    factor_L_per_min_per_kg = factor_ml_min_per_kg / 1000.0

    # determine max total minutes for plotting alignment
    seq_totals = [sum(d for _, d in p['sequence']) for p in parsed]
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
            'VO2_work_L_min': round(vo2_w, 4),
            'Total_minutes': max_total,
            'Total_O2_L': round(total_O2, 3),
            'Total_kcal': round(total_kcal, 2),
            'kcal_per_min': round(kcal_per_min, 3)
        })
        patterns[label] = pat
        cumulative[label] = cum

    df_res = pd.DataFrame(results).sort_values(by='Weight_kg', ascending=False)
    baseline_kcal = df_res['Total_kcal'].max()
    df_res['Pct_saving_vs_heaviest_%'] = df_res['Total_kcal'].apply(lambda x: round((baseline_kcal - x) / baseline_kcal * 100.0, 3))

    st.subheader('תוצאות — טבלה')
    st.dataframe(df_res.reset_index(drop=True))

    # Plots
    st.subheader('גרפים')
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    label_to_weight = {p['label']: p['weight'] for p in parsed}
    for label, pat in patterns.items():
        ax1.plot(np.arange(len(pat)), pat, label=f'{label} ({label_to_weight[label]} kg)')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('VO2 (L/min)')
    ax1.set_title('Instantaneous VO2 pattern')
    ax1.legend(ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    for label, cum in cumulative.items():
        ax2.plot(np.arange(len(cum)), cum, label=f'{label} ({label_to_weight[label]} kg)')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Cumulative O2 (L)')
    ax2.set_title('Cumulative O2 consumed')
    ax2.legend(ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.grid(True)
    st.pyplot(fig2)

    # PDF export function
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
        col_w = [40, 30, 30, 35, 35]  # widths
        hdrs = ['Label', 'Weight', 'VO2 L/min', 'Total kcal', 'kcal/min']
        for i, h in enumerate(hdrs):
            pdf.cell(col_w[i], 6, h, border=1)
        pdf.ln()
        for _, row in df_table.iterrows():
            pdf.cell(col_w[0], 6, str(row['Label']), border=1)
            pdf.cell(col_w[1], 6, str(row['Weight_kg']), border=1)
            pdf.cell(col_w[2], 6, str(row['VO2_work_L_min']), border=1)
            pdf.cell(col_w[3], 6, str(row['Total_kcal']), border=1)
            pdf.cell(col_w[4], 6, str(row['kcal_per_min']), border=1)
            pdf.ln()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Instantaneous VO2 pattern', ln=True)
        img_buf = BytesIO()
        fig1.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        pdf.image(img_buf, x=15, y=None, w=180)
        img_buf.close()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Cumulative O2 consumed', ln=True)
        img_buf2 = BytesIO()
        fig2.savefig(img_buf2, format='png', dpi=150, bbox_inches='tight')
        img_buf2.seek(0)
        pdf.image(img_buf2, x=15, y=None, w=180)
        img_buf2.close()

        # Optionally include help table
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Parameter Explanations', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', size=9)
        for _, r in help_table.iterrows():
            pdf.multi_cell(0, 5, f"{r['Parameter (עברית)']} ({r['Default value']}): {r['Explanation']}")
            pdf.ln(1)

        out = BytesIO()
        pdf.output(out)
        out.seek(0)
        return out

    params_text = (f'Body mass={body_mass} kg | VO2_work={vo2_ml_kg_min} ml/kg/min | '
                   f'factor={factor_ml_min_per_kg} ml/min/kg | rest_vo2={rest_vo2} L/min '
                   f'| kcal_per_L={kcal_per_L}')
    pdf_io = create_pdf(df_res, fig1, fig2, params_text)
    st.download_button('הורד PDF של הדוח', data=pdf_io.getvalue(), file_name=export_pdf_name, mime='application/pdf')

    st.success('החישוב הושלם — הורד את ה-PDF או עדכן פרמטרים והריץ שוב.')

else:
    st.info('הזן אופציות בצד שמאל ולחץ "חשב". (ניתן ללחוץ על "הצג/הסתר הסבר מפורט" כדי לראות את פירוט הפרמטרים.)')
