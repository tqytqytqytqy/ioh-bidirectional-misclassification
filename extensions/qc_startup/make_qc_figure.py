"""Build aggregate-only Supplementary Figure S2 with editable SVG/vector PDF.

IOH_PROJECT_ROOT defaults to the reproducibility package; IOH_QC_OUTPUT_ROOT
defaults to its outputs directory. IOH_QC_FIGURE_DIR and IOH_QC_QA_DIR
optionally select publication and verification directories.
No source data or primary figures are modified. No model is fitted here.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', ROOT / 'outputs')).expanduser().resolve()
AGG = OUT / 'aggregate/qc'
FIGURES = Path(os.environ.get('IOH_QC_FIGURE_DIR', OUT / 'figures')).expanduser().resolve()
QA = Path(os.environ.get('IOH_QC_QA_DIR', OUT / 'qc/figurechecks')).expanduser().resolve()
STEM = 'Supplementary_Figure_S2_Startup_QC'
POLICIES = ('original', 'qc60', 'qc_all', 'startup10')
LABELS = ('Original', 'Sustained segments', 'All flagged bins', 'Exclude startup 10 min')
SHORT_LABELS = ('Original', 'Sustained segments', 'All flagged bins', 'Exclude 10 min*')
COLORS = ('#343A40', '#247A65', '#579881', '#C35537')
MARKERS = ('o', 's', 'D', '^')
CAPTION = 'All panels use MAP <65 mmHg and a 5 min emulated display. (A) Reference episodes (continuous low MAP >=60 s) with no low display or a fresh within-episode low sample. These are not complements: inherited-only low display forms a third category. (B) Normal held display at low-reference infusion adjustments. A-B show phase-averaged percentages and subject-cluster bootstrap 95% CIs (1,000/2,000 replicates). (C) RRs per 10 mmHg min h-1 hidden-deficit AUC, using the original anaesthesia-hour, adjusted for frozen clinical covariates and total-reference-burden spline [3-df quadratic B-spline of log(1 + reference AUC per anaesthesia-hour)]; 95% CIs use subject-cluster robust variance. Outcomes are creatinine-defined AKI through postoperative day 7 and ICU stay >=2 days, combining planned/unplanned admissions. Original remains primary. Startup begins at the first finite reference MAP. Excluding its first 600 s retains original cases and is a window sensitivity, not artifact correction; true early hypotension may be removed. Masking follows the specified signal-quality rules. See startup methods and Tables S8 and S14 to S16 for definitions. The startup-window AKI estimate is retained (RR 1.181; 95% CI 1.003 to 1.391). Associations are exploratory, not causal monitoring effects. AKI, acute kidney injury; AUC, area under the deficit curve; CI, confidence interval; ICU, intensive care unit; MAP, mean arterial pressure; RR, risk ratio.'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    paths = {name: AGG / name for name in ('events_summary.json', 'treatment_summary.json', 'qc_frequency_population.csv', 'outcomes_all_models.csv')}
    data = {name: json.loads(p.read_text()) if p.suffix == '.json' else pd.read_csv(p) for name, p in paths.items()}
    return (paths, data)

def build_source(data):
    rows = []
    for policy, label in zip(POLICIES, LABELS):
        ep = data['events_summary.json']['policies'][policy]['principal']['events']
        for panel, metric, estimate, low, high in (('A', 'reference_episode_miss', 100 * ep['complete_miss_probability'], 100 * ep['complete_miss_ci_low'], 100 * ep['complete_miss_ci_high']), ('A', 'fresh_low_sample_within_reference_episode', ep['fresh_percent'], ep['fresh_lower95'], ep['fresh_upper95'])):
            rows.append(dict(panel=panel, metric=metric, policy=policy, policy_label=label, estimate=estimate, ci_low=low, ci_high=high, unit='percent', n_cases=ep['n_cases_total'], n_subjects=ep['n_clusters_total'], reference_events=ep['reference_episodes'], outcome_events=np.nan, denominator='qualifying reference episodes, averaged over all 30 global phases', uncertainty='subject-cluster percentile bootstrap, 1000 replicates', source_file='events_summary.json', source_field=f'policies.{policy}.principal.events'))
        tr = data['treatment_summary.json']['policies'][policy]['principal_numbers']['display_by_interval_min']['5']['normal_display']
        rows.append(dict(panel='B', metric='normal_display_at_low_reference_infusion_adjustment', policy=policy, policy_label=label, estimate=tr['estimate_pct'], ci_low=tr['lower95_pct'], ci_high=tr['upper95_pct'], unit='percent', n_cases=tr['cases'], n_subjects=tr['subjects'], reference_events=tr['events'], outcome_events=np.nan, denominator='quality-eligible low-reference infusion adjustment blocks', uncertainty='subject-cluster percentile bootstrap, 2000 replicates', source_file='treatment_summary.json', source_field=f'policies.{policy}.principal_numbers.display_by_interval_min.5.normal_display'))
    models = data['outcomes_all_models.csv']
    selected = models[(models.adjustment_sequence == 'clinical_plus_reference') & models.model_family.isin(['aki_sequential', 'icu_sequential']) & (models.map_threshold_mm_hg.isna() | (models.map_threshold_mm_hg == 65)) & (models.display_interval_min.isna() | (models.display_interval_min == 5)) & (models.term == 'hidden_twa10')]
    if len(selected) != 8 or selected.duplicated(['policy', 'model_family']).any():
        raise ValueError('Expected exactly one specified burden-adjusted model per outcome and policy')
    for outcome, family in (('AKI', 'aki_sequential'), ('ICU stay >=2 days', 'icu_sequential')):
        for policy, label in zip(POLICIES, LABELS):
            r = selected[(selected.policy == policy) & (selected.model_family == family)].iloc[0]
            if r.exposure != 'hidden AUC per anaesthesia-hour at MAP <65 mm Hg':
                raise ValueError('Wrong MAP/exposure in selected outcome model')
            if r.reference_burden_adjustment != 'quadratic B-spline (3 df) of log(1 + reference AUC per anaesthesia-hour)':
                raise ValueError('Unexpected total reference-burden adjustment')
            rows.append(dict(panel='C', metric=outcome, policy=policy, policy_label=label, estimate=r.relative_risk, ci_low=r.ci_low, ci_high=r.ci_high, unit='risk_ratio', n_cases=r.n_cases, n_subjects=r.n_subjects, reference_events=np.nan, outcome_events=r.events, denominator=r.population, uncertainty=r.variance_estimator, source_file='outcomes_all_models.csv', source_field=family + '/clinical_plus_reference/hidden_twa10', exposure_scale=r.exposure_scale, covariate_adjustment=r.covariate_adjustment, reference_burden_adjustment=r.reference_burden_adjustment, p_value=r.p_value))
    source = pd.DataFrame(rows)
    if len(source) != 20 or not np.isfinite(source[['estimate', 'ci_low', 'ci_high']]).all().all():
        raise ValueError('Missing figure estimates')
    if not ((source.ci_low <= source.estimate) & (source.estimate <= source.ci_high)).all():
        raise ValueError('Confidence interval does not include estimate')
    startup = source[(source.panel == 'C') & (source.metric == 'AKI') & (source.policy == 'startup10')].iloc[0]
    np.testing.assert_allclose([startup.estimate, startup.ci_low, startup.ci_high], [1.181169, 1.002743, 1.391344], atol=5e-07, rtol=0)
    frequency = data['qc_frequency_population.csv']
    for policy in POLICIES:
        r = frequency[(frequency.policy == policy) & (frequency.threshold == 65) & (frequency.interval_min == 5)]
        if len(r) != 1:
            raise ValueError('Frequency source has ambiguous policy rows')
        ep = data['events_summary.json']['policies'][policy]['principal']
        np.testing.assert_allclose(r.episode_sensitivity.iloc[0], ep['events']['episode_detection_probability'], atol=1e-12)
        np.testing.assert_allclose(r[['HDR', 'ODR', 'NetBias']].to_numpy()[0], [ep['decomposition'][x] for x in ('HDR', 'ODR', 'NetBias')], atol=1e-12)
    return source

def configure_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = 'Arial' if 'Arial' in available else 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 12.4, 'axes.labelsize': 12.4, 'xtick.labelsize': 12.2, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.6, 'text.color': '#343A40', 'axes.labelcolor': '#343A40', 'xtick.color': '#62696C', 'savefig.facecolor': 'white'})
    return family

def panel_title(fig, letter, title, x):
    fig.text(x, 0.824, letter, size=16, weight='bold', va='baseline')
    fig.text(x + 0.022, 0.824, title, size=14.6, weight='bold', va='baseline')

def style_axis(ax, forest=False):
    ax.set_facecolor('white')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#AAB1AF')
    ax.set_yticks([])
    ax.tick_params(axis='x', length=3, width=0.6, pad=4)
    ax.grid(axis='x', color='#E5E9E7', linewidth=0.5, zorder=0)
    if forest:
        ax.set_xscale('log')
        ax.set_xlim(0.76, 1.55)
        ax.set_xticks([0.8, 1.0, 1.4], labels=['0.8', '1.0', '1.4'])
        ax.minorticks_off()
        ax.axvline(1, color='#737A78', lw=0.8, ls='--', zorder=1)
        ax.set_xlabel('Risk ratio (log scale)', labelpad=6)
    else:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Events (%)', labelpad=6)

def plot_panel(fig, source, panel, left):
    bottom, height = (0.223, 0.534)
    ax = fig.add_axes([left + 0.145, bottom, 0.12, height])
    style_axis(ax, forest=panel == 'C')
    ax.set_ylim(-0.6, 9.3)
    groups = {'A': [('reference_episode_miss', 'No low display', [8, 7, 6, 5]), ('fresh_low_sample_within_reference_episode', 'New within-event low sample', [3, 2, 1, 0])], 'B': [('normal_display_at_low_reference_infusion_adjustment', 'Normal display during low MAP', [7.7, 5.3, 2.9, 0.5])], 'C': [('AKI', 'AKI (104 events)', [8, 7, 6, 5]), ('ICU stay >=2 days', 'ICU stay >=2 days (153 events)', [3, 2, 1, 0])]}
    for metric, heading, positions in groups[panel]:
        sub = source[(source.panel == panel) & (source.metric == metric)].set_index('policy')
        header_y = bottom + (positions[0] + 0.6 + 0.75) / 9.9 * height
        if panel == 'C' and metric == 'ICU stay >=2 days':
            header_y = bottom + (positions[0] + 1.0 + 0.6) / 9.9 * height
            title = fig.text(left + 0.002, header_y, 'ICU stay >=2 days\n(153 events)',
                             size=12.2, weight='bold', va='center', linespacing=1.05)
            title.set_gid('C_ICU_group_heading')
        else:
            fig.text(left + 0.002, header_y, heading, size=12.2, weight='bold')
        for i, (policy, y) in enumerate(zip(POLICIES, positions)):
            r = sub.loc[policy]
            ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt=MARKERS[i], color=COLORS[i], ecolor=COLORS[i], markersize=4.7, capsize=3, capthick=1.0, elinewidth=1.15, zorder=3)
            ypos = bottom + (y + 0.6) / 9.9 * height
            label = fig.text(left + 0.002, ypos, SHORT_LABELS[i], size=12.1, va='center', color=COLORS[i])
            if panel == 'C' and metric == 'ICU stay >=2 days' and policy == 'original':
                label.set_gid('C_ICU_first_row')
            if panel == 'C' and metric == 'AKI' and policy == 'startup10':
                label.set_gid('C_AKI_last_row')
            fig.text(left + 0.306, ypos, f'{r.estimate:.3f}' if panel == 'C' else f'{r.estimate:.1f}', size=12.1, ha='right', va='center', color=COLORS[i])
            if panel == 'B':
                fig.text(left + 0.002, ypos - 0.043, f'{int(r.reference_events)} events / {int(r.n_subjects)} subjects', size=12.0, color='#62696C', va='center')
    return ax

def draw(source):
    family = configure_font()
    fig = plt.figure(figsize=(12, 5), dpi=150, facecolor='white')
    fig.text(0.025, 0.967, 'Supplementary Figure S2. Startup quality-control sensitivity analyses', size=16.3, weight='bold', va='top')
    fig.text(0.025, 0.893, 'MAP <65 mmHg  |  5-min display  |  Points and 95% CIs  |  Original remains primary', size=12.3, color='#60686A', va='top')
    for panel, title, left in [('A', 'Reference episodes', 0.025), ('B', 'Infusion adjustments', 0.345), ('C', 'Postoperative outcomes', 0.665)]:
        panel_title(fig, panel, title, left)
        plot_panel(fig, source, panel, left)
    fig.text(0.025, 0.088, '* Exclude 10 min: startup-window sensitivity, not artifact correction. Algorithmic sensitivity rules.', size=12.2, color=COLORS[3])
    fig.text(0.025, 0.034, 'Startup-window AKI: RR 1.181 (95% CI 1.003 to 1.391). Outcome models adjust for clinical covariates + total reference burden.', size=12.0, color='#343A40')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tagged = {a.get_gid(): a for a in fig.texts if a.get_gid()}
    heading_box = tagged['C_ICU_group_heading'].get_window_extent(renderer)
    lower_row_box = tagged['C_ICU_first_row'].get_window_extent(renderer)
    upper_row_box = tagged['C_AKI_last_row'].get_window_extent(renderer)
    forest_box = fig.axes[2].get_window_extent(renderer)
    if heading_box.x1 >= forest_box.x0 - 1:
        raise ValueError('ICU subgroup heading extends into forest plot')
    if heading_box.y0 <= lower_row_box.y1 + 1 or heading_box.y1 >= upper_row_box.y0 - 1:
        raise ValueError('ICU subgroup heading overlaps an adjacent result row')
    heading_layout = {'status': 'PASS', 'heading_wrapped': True,
                      'heading_bbox_px': [float(x) for x in heading_box.extents],
                      'forest_left_px': float(forest_box.x0),
                      'clearance_to_lower_row_px': float(heading_box.y0 - lower_row_box.y1),
                      'clearance_to_upper_row_px': float(upper_row_box.y0 - heading_box.y1)}
    clipped = []
    for artist in fig.texts:
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < 1 or bounds.y0 < 1 or bounds.x1 > fig.bbox.width - 1 or (bounds.y1 > fig.bbox.height - 1):
            clipped.append(artist.get_text()[:100])
    if clipped:
        raise ValueError(f'Figure text outside canvas: {clipped}')
    check = {'font_family': family, 'page_inches': list(fig.get_size_inches()), 'figure_text_clipping': clipped, 'caption_external_for_DOCX_embedding': True, 'caption_words': len(CAPTION.split()), 'at_9_1_inch_embed': {'height_inches': 5 * 9.1 / 12, 'minimum_text_points': 12.0 * 9.1 / 12}, 'axes_count': len(fig.axes), 'data_points': len(source)}
    check['ICU_group_heading_layout'] = heading_layout
    if len(fig.axes) != 3 or len(source) != 20:
        raise ValueError('A required panel or point is missing')
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(FIGURES / f'{STEM}.{extension}', dpi=300, bbox_inches=None, metadata={'Creator': 'Matplotlib; aggregate-only technical QC sensitivity'} if extension == 'pdf' else None)
    fig.savefig(QA / 'S2_preview.png', dpi=150)
    bounds = {'A': (0.018, 0.145, 0.335, 0.859), 'B': (0.338, 0.145, 0.655, 0.859), 'C': (0.658, 0.145, 0.985, 0.859)}
    plt.close(fig)
    im = Image.open(FIGURES / f'{STEM}.png').convert('RGB')
    pixels = np.asarray(im)
    check['png_dimensions'] = list(im.size)
    check['png_dpi'] = Image.open(FIGURES / f'{STEM}.png').info.get('dpi')
    check['panel_pixel_checks'] = {}
    for panel, (x0, y0, x1, y1) in bounds.items():
        region = pixels[int((1 - y1) * im.height):int((1 - y0) * im.height), int(x0 * im.width):int(x1 * im.width)]
        nonwhite = float(np.mean(np.min(region, axis=2) < 235))
        if nonwhite < 0.01:
            raise ValueError(f'Panel {panel} is blank')
        check['panel_pixel_checks'][panel] = {'nonwhite_fraction': nonwhite, 'nonblank': True}
        Image.fromarray(region).save(QA / f'S2_panel_{panel}.png')
    svg = ET.parse(FIGURES / f'{STEM}.svg')
    ns = {'s': 'http://www.w3.org/2000/svg'}
    check['svg_text_elements'] = len(svg.findall('.//s:text', ns))
    check['svg_embedded_images'] = len(svg.findall('.//s:image', ns))
    if check['svg_text_elements'] < 50 or check['svg_embedded_images']:
        raise ValueError('SVG is not editable vector text/artwork')
    return check

def verify_vector_and_page_fit():
    from pypdf import PdfReader
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    reader = PdfReader(FIGURES / f'{STEM}.pdf')
    if len(reader.pages) != 1 or len(reader.pages[0].images):
        raise ValueError('Final PDF must be a single all-vector figure')
    text = reader.pages[0].extract_text()
    for required in ('Reference episodes', 'Infusion adjustments', 'Postoperative outcomes', '1.003 to 1.391'):
        if required not in text:
            raise ValueError(f'Required PDF label missing: {required}')
    font = font_manager.findfont(plt.rcParams['font.family'][0])
    pdfmetrics.registerFont(TTFont('FigureCaptionFont', font))
    width, height = landscape(A4)
    margin = 3 * cm
    available_width, available_height = (width - 2 * margin, height - 2 * margin)
    image_width, image_height = (9.1 * inch, 9.1 / 12 * 5 * inch)
    caption_html = escape(CAPTION).replace('h-1', 'h<super>-1</super>')
    paragraph = Paragraph(caption_html, ParagraphStyle('caption', fontName='FigureCaptionFont', fontSize=9, leading=10.8, spaceBefore=0, spaceAfter=0))
    paragraph_width, paragraph_height = paragraph.wrap(available_width, available_height)
    total = image_height + 7 + paragraph_height
    if total > available_height:
        raise ValueError('Figure plus complete caption exceeds A4 landscape 3-cm-margin page')
    proof = QA / 'S2_A4_landscape_3cm_proof.pdf'
    doc = canvas.Canvas(str(proof), pagesize=(width, height))
    top = height - margin
    doc.drawImage(str(FIGURES / f'{STEM}.png'), (width - image_width) / 2, top - image_height, width=image_width, height=image_height)
    paragraph.drawOn(doc, margin, top - image_height - 7 - paragraph_height)
    doc.showPage()
    doc.save()
    return {'pdf_vector_images': len(reader.pages[0].images), 'pdf_pages': len(reader.pages), 'pdf_key_text_verified': True, 'proof_file': proof.name, 'landscape_A4_3cm_margins': {'usable_width_in': available_width / inch, 'usable_height_in': available_height / inch, 'embedded_figure_width_in': image_width / inch, 'embedded_figure_height_in': image_height / inch, 'caption_font_pt': 9, 'caption_height_in': paragraph_height / inch, 'total_used_height_in': total / inch, 'remaining_vertical_space_in': (available_height - total) / inch}}

def main():
    if len(CAPTION.split()) > 210:
        raise ValueError('Supplementary Figure S2 caption exceeds 210 words')
    for folder in (FIGURES, OUT / 'figures', OUT / 'tables', QA):
        folder.mkdir(parents=True, exist_ok=True)
    paths, data = sources()
    source_hashes = {name: sha(p) for name, p in paths.items()}
    primary_figures = sorted(set((OUT / 'figures').glob('Figure_*')) | set(FIGURES.glob('Figure_*')))
    frozen_hashes = {p: sha(p) for p in primary_figures}
    source = build_source(data)
    source_csv = AGG / 'qc_figure_source.csv'
    source_bytes = source.to_csv(index=False).encode('utf-8')
    if source_csv.exists():
        if source_csv.read_bytes() != source_bytes:
            raise ValueError('Existing figure source CSV differs; refusing a numeric/source rewrite')
    else:
        source_csv.write_bytes(source_bytes)
    (AGG / 'qc_figure_caption.txt').write_text(CAPTION + '\n', encoding='utf-8')
    (OUT / 'tables/Supplementary_Figure_S2_caption.txt').write_text(CAPTION + '\n', encoding='utf-8')
    check = draw(source)
    check.update(verify_vector_and_page_fit())
    if FIGURES != OUT / 'figures':
        for ext in ('png', 'pdf', 'svg'):
            shutil.copy2(FIGURES / f'{STEM}.{ext}', OUT / 'figures' / f'{STEM}.{ext}')
    if source_hashes != {name: sha(p) for name, p in paths.items()}:
        raise ValueError('Aggregate input changed during figure generation')
    if frozen_hashes != {p: sha(p) for p in primary_figures}:
        raise ValueError('A primary figure changed during figure generation')
    check.update({'status': 'GENERATED_PENDING_VISUAL_REVIEW', 'source_sha256': source_hashes, 'figure_sha256': {ext: sha(FIGURES / f'{STEM}.{ext}') for ext in ('png', 'pdf', 'svg')}, 'caption_sha256': sha(AGG / 'qc_figure_caption.txt'), 'source_csv_sha256': sha(AGG / 'qc_figure_source.csv'), 'runner_sha256': sha(Path(__file__)), 'proof_pdf_sha256': sha(QA / check['proof_file']), 'source_rows': len(source), 'clinical_signoff': False, 'primary_policy': 'original', 'primary_figures_unchanged': True, 'primary_figure_files_checked': len(frozen_hashes), 'contains_individual_case_data': False, 'model_selection': 'MAP65; 5 min; aki_sequential/icu_sequential; clinical_plus_reference; hidden_twa10', 'frequency_source_role': 'Independent cross-check of event sensitivity and HDR/ODR/NetBias; not plotted', 'recovery_role': 'Not plotted; supplementary tables retain recovery outcomes'})
    (QA / 'S2_checks.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps(check, indent=2))
if __name__ == '__main__':
    main()
