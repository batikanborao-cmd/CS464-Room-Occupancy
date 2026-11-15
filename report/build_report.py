"""Builds the CS464 progress report from one content source.

Outputs (in this folder):
  main.tex                              LaTeX source (compile on Overleaf: pdflatex main.tex, twice for the contents)
  CS464_Progress_Report_Group20.pdf     PDF rendered with ReportLab in the same layout

Inline markup in the text below: **bold**, *italic*, `code`.
"""
import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

TITLE = ["CS 464", "Progress Report: Robust Room Occupancy Estimation", "Group 20"]
AUTHORS = [
    ("Turan Emre Adıgüzel", "22203033"),
    ("Ali Osman Karahasanoğlu", "22201636"),
    ("Batıkan Bora Özüer", "22202849"),
    ("Merve Topuz", "22302937"),
    ("Öykü Kurt", "22202131"),
]
DATE = "November 29, 2026"

# ---------------------------------------------------------------- content
# ("h1", text) ("h2", text) ("p", text) ("ul", [items]) ("ol", [items])
# ("table", caption, header, rows, {"bold_rows": [...], "align": "lrr..."}) ("fig", file, caption, width_fraction)
C = []
h1 = lambda t: C.append(("h1", t))
h2 = lambda t: C.append(("h2", t))
p = lambda t: C.append(("p", t))
ul = lambda *items: C.append(("ul", list(items)))
ol = lambda *items: C.append(("ol", list(items)))
table = lambda cap, head, rows, **kw: C.append(("table", cap, head, rows, kw))
fig = lambda f, cap, w=0.85: C.append(("fig", f, cap, w))

h1("Introduction")
p("The goal of this project is to estimate the number of people in a room (0, 1, 2 or 3) from non-intrusive "
  "environmental sensors (temperature, light, sound, CO2 and passive-infrared motion) instead of cameras. "
  "Camera-free occupancy estimation is useful for energy-efficient heating, ventilation and lighting control "
  "while preserving privacy.")
p("We will also test how performance changes when sensors are removed, measurements become noisy, or readings "
  "are missing. The final study will compare Logistic Regression, Random Forest, XGBoost and a Multi-Layer "
  "Perceptron on the same day-based evaluation protocol.")
p("This progress report covers the first phase: dataset analysis, the fixed train/validation/test split that all "
  "later experiments will share, and initial results of a Random Forest baseline. The baseline performs well on "
  "the test day (macro-F1 0.906) but much worse on the validation days; for example, class-3 F1 falls from 0.901 "
  "on 23 December 2017 to 0.063 on 10 January 2018.")

h1("Dataset")
h2("Dataset Analysis")
p("We use the UCI Room Occupancy Estimation dataset [1, 2]: 10,129 tabular samples recorded every ~30 seconds "
  "(median interval 31 s) in a room between 22 December 2017 and 11 January 2018. There are no missing values "
  "and no images; each sample has a date, a time, 16 numeric sensor features and the label "
  "`Room_Occupancy_Count`. The sensors are summarised in Table 1.")
table("Sensor features in the dataset",
      ["Sensor type", "Features", "Unit / range"],
      [["Temperature", "S1–S4_Temp", "24.4–29.0 °C"],
       ["Light", "S1–S4_Light", "0–280 lux"],
       ["Sound", "S1–S4_Sound", "0.04–3.88 V (microphone amplitude)"],
       ["CO2", "S5_CO2, S5_CO2_Slope", "345–1270 ppm; slope over a sliding window"],
       ["Motion (PIR)", "S6_PIR, S7_PIR", "binary 0/1"]], align="lll", widths=[3.2, 4.6, 7.4])
p("The target has four classes (0, 1, 2, 3 occupants) and is highly imbalanced: an empty room accounts for "
  "81.2% of the samples (Table 2).")
table("Class distribution of the whole dataset",
      ["Occupants", "Samples", "Share"],
      [["0", "8,228", "81.2%"], ["1", "459", "4.5%"], ["2", "748", "7.4%"], ["3", "694", "6.9%"]], align="crr", widths=[3, 3, 3])
p("Two findings from the analysis shape the rest of the project:")
ul("**Occupancy is concentrated in three days.** Only 22 Dec 2017, 23 Dec 2017 and 10 Jan 2018 contain people; "
   "the other four days contain only empty-room samples. Class 1 appears only on 22 and 23 Dec 2017.",
   "**Light, sound and CO2 separate the classes most clearly.** Mean S1_Light is 2.7 lux for an empty room versus "
   "113–136 lux when occupied, and mean CO2 rises from 404 ppm (0 people) to 851 ppm (3 people). Mean "
   "temperature differs by less than 1 °C between classes. The four light sensors are strongly correlated with "
   "each other, as are the four temperature sensors.")
fig("timeseries_2017-12-22.png", "Light, sound, CO2 and temperature on 22 Dec 2017 with the true occupant count "
    "(bottom). Light and sound change with occupancy almost instantly; CO2 and temperature follow with a delay.", 0.8)
p("Figure 1 shows one occupied day. Figure 2 shows the distribution of six features per class: an empty room is "
  "clearly separated, while classes 2 and 3 overlap strongly.")
fig("feature_by_class.png", "Distribution of six features per class (box = interquartile range, outliers hidden).", 0.9)

h2("Dataset Preprocessing")
p("**Split.** Consecutive readings 30 s apart are nearly identical, so a random split puts near-duplicates of test "
  "samples into training and gives optimistic scores (Section 6.4). Following our TA's feedback, we use a "
  "**day-based split** in which validation days are separate from test days, and **the same split is used for "
  "every model and experiment** (defined once in the code). Because only three days are occupied, each split "
  "receives exactly one occupied day, plus empty days to balance sizes (Table 3, Figure 3).")
table("Fixed day-based split and samples per class",
      ["Split", "Days", "0", "1", "2", "3", "Total"],
      [["Train", "22, 24, 25, 26 Dec 2017", "4,480", "205", "257", "363", "5,305 (52%)"],
       ["Validation", "10, 11 Jan 2018", "1,751", "0", "93", "201", "2,045 (20%)"],
       ["Test", "23 Dec 2017", "1,997", "254", "398", "130", "2,779 (27%)"]], align="llrrrrr", widths=[2.3, 4.5, 1.4, 1.1, 1.1, 1.1, 2.6])
fig("split_distribution.png", "Class distribution per split.", 0.6)
p("Train and test contain all four classes. The validation days contain no class-1 samples: class 1 exists only "
  "on two days, so it cannot appear in three disjoint day groups. This is a structural limit of the dataset. "
  "Validation macro-F1 is therefore computed over a fixed label set {0, 2, 3} (`zero_division = 0`), the same for "
  "every model, and class-1 performance cannot be monitored on validation during tuning. We plan to confirm this "
  "choice with our TA.")
p("**Other steps.** Date and Time are dropped from the features so the model cannot memorise when a day was "
  "recorded. No imputation is needed. Tree-based models do not require feature scaling; for Logistic Regression "
  "and the MLP in the next phase, a standard scaler will be fitted on the training split only.")

h1("Coding Environment")
p("All code is written in Python 3.12 and runs on a standard laptop CPU; Random Forest training takes about 0.5 s, "
  "so no GPU is needed at this stage. The tools are listed in Table 4.")
table("Tools and libraries",
      ["Tool", "Use"],
      [["pandas, NumPy", "Loading the CSV, time stamps, split construction, statistics"],
       ["scikit-learn 1.9 [4]", "Random Forest, metrics (macro-F1, accuracy, confusion matrix)"],
       ["Matplotlib [5]", "All figures in this report"],
       ["Git + GitHub", "Version control and sharing within the team"],
       ["XGBoost, PyTorch (planned)", "XGBoost and MLP models in the next phase"]], align="ll", widths=[5, 10.2])
p("The code is public on GitHub [6]. `src/common.py` holds data loading and the fixed split, `src/eda.py` "
  "produces the dataset analysis, and `src/train_rf.py` trains and evaluates the baseline. All results can be "
  "reproduced by installing `requirements.txt` and running `python eda.py` and `python train_rf.py` inside `src/`.")

h1("Details of the Trained Model: Random Forest")
p("We chose **Random Forest** [3] as the first model because it needs no feature scaling, handles correlated "
  "features, trains in under a second on this data, and reports feature importances that feed directly into the "
  "planned sensor-ablation study.")
p("**How it works.** A Random Forest is an ensemble of decision trees. Each tree is trained on a bootstrap sample "
  "of the training set (sampling with replacement), and at every split it only considers a random subset of the "
  "features. Each tree greedily chooses the feature and threshold that most reduce Gini impurity, "
  "1 − Σ p_k², where p_k is the fraction of class k in the node. At prediction time the trees' class "
  "probabilities are averaged and the most likely class is returned. Bootstrapping and feature subsampling make "
  "the trees less correlated, so averaging them reduces variance compared with a single deep tree.")
p("**Hyperparameters and their expected effects.** The baseline uses scikit-learn defaults (Table 5); tuning on "
  "the validation split is planned for the final report. A fixed random seed (42) is used so that results are "
  "reproducible.")
table("Random Forest hyperparameters of the baseline and their expected effects",
      ["Hyperparameter", "Value", "Expected effect"],
      [["n_estimators", "100", "More trees give more stable predictions with diminishing returns; training and "
                               "inference time grow linearly."],
       ["max_depth", "None", "Fully grown trees fit the training data perfectly (train macro-F1 = 1.0); limiting "
                             "depth adds bias but can generalise better across days."],
       ["max_features", "sqrt(16) = 4", "Fewer features per split decorrelate the trees; more features make each "
                                        "tree stronger but the ensemble more similar."],
       ["min_samples_leaf", "1", "Larger leaves smooth the decision boundary and reduce overfitting to "
                                 "near-duplicate readings."],
       ["class_weight", "None", "\"balanced\" re-weights classes by inverse frequency, which should help the "
                                "minority classes 1–3."],
       ["criterion", "gini", "Entropy usually gives similar results; minor effect expected."]],
      align="llX", widths=[3.7, 2.6, 8.9])

h1("Experimental Setup")
p("The baseline Random Forest is trained on the training days with all 16 sensor features and evaluated on the "
  "training, validation and test splits. A second Random Forest with identical settings is trained on a "
  "stratified random 80/20 split, only as a reference. Because 81% of samples are empty-room readings, "
  "**macro-F1 is the main metric** (the unweighted mean of per-class F1, so each class counts equally); accuracy "
  "is reported as a secondary score. We also report per-class F1, a confusion matrix, feature importances and "
  "training/inference time.")
p("The goals of this first set of experiments are:")
ol("Establish a reproducible baseline on the fixed day-based split that every later model is compared against on "
   "the same test day.",
   "Show how optimistic a random split is compared with the day-based split (reference only, not used for model "
   "selection).",
   "Identify which sensors the model relies on, as input for the sensor-ablation experiment.")
p("The protocol for the rest of the project follows our TA's feedback. In this phase the test day was used only "
  "to report the untuned baseline. From now on, all model and hyperparameter choices will be made on the "
  "validation split, and the final comparison of all models will be done on the same test day with the same split.")
p("**Planned experiments for the final report:** Logistic Regression, XGBoost and an MLP on the same split; "
  "hyperparameter tuning; sensor ablation (subsets of sensor types); noise robustness (Gaussian noise of "
  "increasing strength); missing-sensor simulation with different imputation strategies; class-imbalance handling "
  "(class weighting, resampling); and an efficiency comparison (inference time versus macro-F1).")

h1("Results")
p("The Random Forest baseline reaches a test macro-F1 of **0.906** (accuracy 0.945) on the held-out test day. On "
  "the validation days it reaches 0.429; computed over the same three classes (0, 2, 3), the test score is 0.884, "
  "so the drop between days remains large. A random split gives 0.990 (Table 6). Training took 0.46 s and "
  "inference about 0.01 ms per sample.")
table("Random Forest baseline results",
      ["Evaluation", "Macro-F1", "Accuracy"],
      [["Train (22, 24–26 Dec 2017)", "1.000", "1.000"],
       ["Validation (10–11 Jan 2018), classes 0/2/3", "0.429", "0.868"],
       ["Test (23 Dec 2017), classes 0/2/3", "0.884", "–"],
       ["Test (23 Dec 2017), all classes", "0.906", "0.945"],
       ["Random 80/20 split (reference)", "0.990", "0.997"]], align="lrr", bold_rows=[3], widths=[9, 2.6, 2.6])

h2("Test Day")
fig("rf_confusion_matrix.png", "Random Forest confusion matrix on the test day (23 Dec 2017): counts (left) and "
    "row-normalised (right).", 0.95)
p("The weakest class is 2 occupants: 111 of 398 samples (28%) are predicted as an empty room, and 26 as three "
  "people (Figure 4). Classes 0, 1 and 3 are recognised with F1 at or above 0.90 (Table 7).")
table("Random Forest classification report on the test day",
      ["", "Precision", "Recall", "F1-Score", "Support"],
      [["0 occupants", "0.94", "1.00", "0.97", "1997"],
       ["1 occupant", "0.99", "0.96", "0.97", "254"],
       ["2 occupants", "0.97", "0.65", "0.78", "398"],
       ["3 occupants", "0.83", "0.98", "0.90", "130"],
       ["Accuracy", "", "", "0.94", "2779"],
       ["Macro Avg", "0.93", "0.90", "0.91", "2779"],
       ["Weighted Avg", "0.95", "0.94", "0.94", "2779"]], align="lrrrr", bold_rows=[4, 5, 6], midrule_before=4, widths=[3.6, 2.4, 2.4, 2.4, 2.4])

h2("Feature Importance")
p("The model relies mostly on light: S2_Light (0.208) and S1_Light (0.205) together account for 41% of the total "
  "Gini importance, followed by S5_CO2_Slope (0.115), S3_Light (0.094) and S5_CO2 (0.077) (Figure 5). The two "
  "PIR motion sensors have very low Gini importance (0.002 and 0.001). Gini importance is affected by correlated "
  "features, so this does not show that PIR is useless; the sensor-ablation experiment will test whether it adds "
  "predictive value.")
fig("rf_feature_importance.png", "Gini feature importance of the Random Forest baseline.", 0.6)

h2("Why Validation Is Much Lower: A Shift Between Days")
p("On 10 Jan 2018, with three people present, mean S1_Light is 18 lux and S2_Light 26 lux, compared with "
  "150–240 lux for three people in December. The model labels 184 of 201 three-person samples as empty and "
  "reaches only 0.063 F1 for class 3. Other sensors still indicate presence: mean CO2 is 554 ppm and the PIR "
  "sensors are active in 55% and 68% of these samples, compared with 0% for the empty room. Why light was low "
  "that day is not known from the data.")
p("The January results suggest that the baseline, which relies most on light, does not generalise well to "
  "occupied periods with low light levels. The sensor-ablation, noise and missing-sensor experiments will test "
  "whether reducing reliance on light improves validation performance; CO2 and motion are candidates for this "
  "analysis.")

h2("Random Versus Day-Based Split")
p("With a random split the same model scores 0.990 macro-F1. Near-duplicate readings 30 s apart fall on both "
  "sides of a random split; training size and day composition also differ, so the gap cannot be attributed to "
  "leakage alone. The day-based split keeps each day's readings in one partition and evaluates the model on a "
  "day excluded from training. The 74% day-based macro-F1 in our proposal came from a different preliminary "
  "split and is not directly comparable to these results.")
p("These results come from a single test day; they show how the baseline behaves on that day, not its general "
  "robustness.")

h1("Division of Work Among Teammates")
p("We divided the workload as follows:")
ul("**Batıkan Bora Özüer**: Repository setup, data pipeline, day-based split, Random Forest baseline and "
   "progress report preparation.",
   "**Turan Emre Adıgüzel**: [to be filled]",
   "**Ali Osman Karahasanoğlu**: [to be filled]",
   "**Merve Topuz**: [to be filled]",
   "**Öykü Kurt**: [to be filled]")

REFS = [
    "A. P. Singh, V. Jain, S. Chaudhari, F. A. Kraemer, S. Werner and V. Garg, \"Machine Learning-Based Occupancy "
    "Estimation Using Multivariate Sensor Nodes,\" in *2018 IEEE Globecom Workshops (GC Wkshps)*, Abu Dhabi, 2018.",
    "A. P. Singh and S. Chaudhari, *Room Occupancy Estimation* [Dataset], UCI Machine Learning Repository, 2018. "
    "DOI: 10.24432/C5P605. Available: `https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation`.",
    "L. Breiman, \"Random Forests,\" *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
    "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" *Journal of Machine Learning Research*, "
    "vol. 12, pp. 2825–2830, 2011.",
    "J. D. Hunter, \"Matplotlib: A 2D Graphics Environment,\" *Computing in Science & Engineering*, vol. 9, no. 3, "
    "pp. 90–95, 2007.",
    "Project repository, *CS464-Room-Occupancy*, GitHub. Available: "
    "`https://github.com/batikanborao-cmd/CS464-Room-Occupancy`.",
]


# ---------------------------------------------------------------- LaTeX output
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    s = "".join(rep.get(ch, ch) for ch in s)
    s = s.replace("p\\_k²", "$p_k^2$")
    s = s.replace("°C", r"\textdegree{}C").replace("Σ", r"$\sum$").replace("²", r"$^2$").replace("−", r"$-$")
    s = s.replace("p\\_k", r"$p_k$").replace("–", "--").replace("CO2", r"CO\textsubscript{2}")
    return s


def tex_inline(s):
    out, pos = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*", s):
        out.append(tex_escape(s[pos:m.start()]))
        if m.group(1) is not None:
            code = m.group(1)
            out.append(r"\url{%s}" % code if code.startswith("http") else r"\texttt{%s}" % tex_escape(code))
        elif m.group(2) is not None:
            out.append(r"\textbf{%s}" % tex_escape(m.group(2)))
        else:
            out.append(r"\textit{%s}" % tex_escape(m.group(3)))
        pos = m.end()
    out.append(tex_escape(s[pos:]))
    return "".join(out).replace('"', "''")


def build_tex():
    L = [r"\documentclass[12pt,a4paper]{article}",
         r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{textcomp}",
         r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{tabularx}", r"\usepackage{float}",
         r"\usepackage[margin=1in]{geometry}", r"\usepackage{url}", r"\usepackage{hyperref}",
         r"\hypersetup{hidelinks}", r"\graphicspath{{figures/}}", "",
         r"\title{" + r" \\ ".join(TITLE) + "}",
         r"\author{" + r" \\ ".join(f"{n} {i}" for n, i in AUTHORS) + "}",
         r"\date{" + DATE + "}", "", r"\begin{document}", r"\maketitle", r"\thispagestyle{empty}",
         r"\begin{center}\includegraphics[width=0.6\textwidth]{bilkent_logo.png}\end{center}",
         r"\newpage", r"\setcounter{page}{1}", r"\tableofcontents", r"\newpage", ""]
    for item in C:
        kind = item[0]
        if kind == "h1":
            L += [r"\section{%s}" % tex_inline(item[1])]
        elif kind == "h2":
            L += [r"\subsection{%s}" % tex_inline(item[1])]
        elif kind == "p":
            L += [tex_inline(item[1]), ""]
        elif kind in ("ul", "ol"):
            env = "itemize" if kind == "ul" else "enumerate"
            L += [r"\begin{%s}" % env] + [r"  \item " + tex_inline(x) for x in item[1]] + [r"\end{%s}" % env, ""]
        elif kind == "table":
            _, cap, head, rows, kw = item
            align = kw.get("align", "l" * len(head))
            spec = align.replace("X", "X")
            tabenv = "tabularx" if "X" in align else "tabular"
            widtharg = r"{\textwidth}" if "X" in align else ""
            bold = set(kw.get("bold_rows", []))
            L += [r"\begin{table}[H]", r"\centering", r"\begin{%s}%s{%s}" % (tabenv, widtharg, spec), r"\toprule",
                  " & ".join(r"\textbf{%s}" % tex_inline(h) for h in head) + r" \\", r"\midrule"]
            for r_i, row in enumerate(rows):
                if kw.get("midrule_before") == r_i:
                    L.append(r"\midrule")
                cells = [tex_inline(c) for c in row]
                if r_i in bold:
                    cells = [r"\textbf{%s}" % c if c else c for c in cells]
                L.append(" & ".join(cells) + r" \\")
            L += [r"\bottomrule", r"\end{%s}" % tabenv, r"\caption{%s}" % tex_inline(cap), r"\end{table}", ""]
        elif kind == "fig":
            _, f, cap, w = item
            L += [r"\begin{figure}[H]", r"\centering", r"\includegraphics[width=%.2f\textwidth]{%s}" % (w, f),
                  r"\caption{%s}" % tex_inline(cap), r"\end{figure}", ""]
    L += [r"\begin{thebibliography}{9}"]
    L += [r"\bibitem{r%d} %s" % (i + 1, tex_inline(r)) for i, r in enumerate(REFS)]
    L += [r"\end{thebibliography}", r"\end{document}", ""]
    (HERE / "main.tex").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- PDF output
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Serif", str(FONT_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("Serif-Bold", str(FONT_DIR / "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("Serif-Italic", str(FONT_DIR / "timesi.ttf")))
pdfmetrics.registerFont(TTFont("Serif-BoldItalic", str(FONT_DIR / "timesbi.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "consola.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                              boldItalic="Serif-BoldItalic")

BODY = ParagraphStyle("body", fontName="Serif", fontSize=12, leading=15, alignment=TA_JUSTIFY, spaceAfter=4)
BODY_IND = ParagraphStyle("bodyind", parent=BODY, firstLineIndent=18)
H1 = ParagraphStyle("h1", fontName="Serif-Bold", fontSize=17, leading=21, spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="Serif-Bold", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
CAP = ParagraphStyle("cap", parent=BODY, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12)
CELL = ParagraphStyle("cell", fontName="Serif", fontSize=11, leading=13)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Serif-Bold")
REF = ParagraphStyle("ref", parent=BODY, fontSize=12, leftIndent=26, firstLineIndent=-26, alignment=0, spaceAfter=6)


def rl_inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r'<font name="Mono" size="11">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    s = s.replace("CO2", "CO<sub>2</sub>").replace("p_k²", "p<sub>k</sub><super>2</super>")
    s = s.replace("p_k", "p<sub>k</sub>")
    return s


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2.6 * cm, rightMargin=2.6 * cm, topMargin=2.5 * cm,
                         bottomMargin=2.5 * cm, title="CS464 Progress Report - Group 20",
                         author="CS464 Group 20")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._footer)])

    @staticmethod
    def _footer(canvas, doc):
        n = canvas.getPageNumber() - 1
        if n >= 1:
            canvas.setFont("Serif", 12)
            canvas.drawCentredString(A4[0] / 2, 1.5 * cm, str(n))

    def afterFlowable(self, f):
        for g in (f._content if isinstance(f, KeepTogether) else [f]):
            if isinstance(g, Paragraph) and getattr(g, "_toc", None) is not None:
                level, text = g._toc
                key = f"h{id(g)}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page - 1, key))


def build_pdf():
    out = HERE / "CS464_Progress_Report_Group20.pdf"
    doc = Doc(out)
    s = []
    # title page
    s.append(Spacer(1, 2.2 * cm))
    for t in TITLE:
        s.append(Paragraph(t, ParagraphStyle("t", fontName="Serif", fontSize=20, leading=25, alignment=TA_CENTER)))
    s.append(Spacer(1, 14))
    for n, i in AUTHORS:
        s.append(Paragraph(f"{n} {i}", ParagraphStyle("a", fontName="Serif", fontSize=12, leading=15,
                                                      alignment=TA_CENTER)))
    s.append(Spacer(1, 10))
    s.append(Paragraph(DATE, ParagraphStyle("d", fontName="Serif", fontSize=12, alignment=TA_CENTER)))
    s.append(Spacer(1, 1.2 * cm))
    s.append(Image(str(FIG / "bilkent_logo.png"), width=11 * cm, height=11 * cm))
    s.append(PageBreak())
    # contents
    s.append(Paragraph("Contents", ParagraphStyle("ct", fontName="Serif-Bold", fontSize=20, leading=24,
                                                  spaceAfter=14)))
    toc = TableOfContents(dotsMinLevel=1)
    toc.levelStyles = [
        ParagraphStyle("t0", fontName="Serif-Bold", fontSize=12, leading=15, leftIndent=18, firstLineIndent=-18,
                       spaceBefore=8),
        ParagraphStyle("t1", fontName="Serif", fontSize=12, leading=15, leftIndent=46, firstLineIndent=-28),
    ]
    s.append(toc)
    s.append(PageBreak())

    sec = sub = tab = fg = 0
    after_heading = True
    pending = []
    for item in C:
        kind = item[0]
        start = len(s)
        if kind in ("h1", "h2"):
            if kind == "h1":
                sec, sub = sec + 1, 0
                num, style, lvl = f"{sec}", H1, 0
            else:
                sub += 1
                num, style, lvl = f"{sec}.{sub}", H2, 1
            text = rl_inline(item[1])
            para = Paragraph(f"{num}&nbsp;&nbsp;&nbsp;&nbsp;{text}", style)
            para._toc = (lvl, f"{num}&nbsp;&nbsp;&nbsp;{text}")
            pending.append(para)
            after_heading = True
            continue
        elif kind == "p":
            para = Paragraph(rl_inline(item[1]), BODY if after_heading else BODY_IND)
            after_heading = False
            if item[1].rstrip().endswith(":"):
                pending.append(para)
                continue
            s.append(para)
        elif kind in ("ul", "ol"):
            items = [ListItem(Paragraph(rl_inline(x), BODY), leftIndent=24) for x in item[1]]
            s.append(ListFlowable(items, bulletType="bullet" if kind == "ul" else "1",
                                  start="•" if kind == "ul" else None, bulletFontName="Serif", bulletFontSize=12,
                                  leftIndent=24, bulletDedent=14, spaceBefore=2, spaceAfter=6))
            after_heading = False
        elif kind == "table":
            _, cap, head, rows, kw = item
            tab += 1
            align = kw.get("align", "l" * len(head))
            bold = set(kw.get("bold_rows", []))
            data = [[Paragraph(f"<b>{rl_inline(h)}</b>", CELL) for h in head]]
            for r_i, row in enumerate(rows):
                st = CELL_B if r_i in bold else CELL
                data.append([Paragraph(rl_inline(c), st) for c in row])
            if kw.get("widths"):
                widths = [w * cm for w in kw["widths"]]
            elif "X" in align:
                fixed = 3.2 * cm
                widths = [fixed if a != "X" else None for a in align]
                rest = doc.width - fixed * align.count("l")
                widths = [w if w else rest for w in widths]
            else:
                widths = None
            t = Table(data, colWidths=widths, hAlign="CENTER")
            ts = [("LINEABOVE", (0, 0), (-1, 0), 1.1, "black"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, "black"),
                  ("LINEBELOW", (0, -1), (-1, -1), 1.1, "black"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                  ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                  ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
            if kw.get("midrule_before") is not None:
                ts.append(("LINEABOVE", (0, kw["midrule_before"] + 1), (-1, kw["midrule_before"] + 1), 0.6, "black"))
            for c_i, a in enumerate(align):
                if a == "r":
                    for r_i in range(len(data)):
                        data[r_i][c_i].style = ParagraphStyle("r", parent=data[r_i][c_i].style, alignment=2)
                elif a == "c":
                    for r_i in range(len(data)):
                        data[r_i][c_i].style = ParagraphStyle("c", parent=data[r_i][c_i].style, alignment=1)
            t.setStyle(TableStyle(ts))
            s.append(KeepTogether([Spacer(1, 6), t, Paragraph(f"Table {tab}: {rl_inline(cap)}", CAP)]))
            after_heading = False
        elif kind == "fig":
            _, f, cap, w = item
            fg += 1
            path = FIG / f
            iw, ih = ImageReader(str(path)).getSize()
            width = doc.width * w
            height = width * ih / iw
            if height > doc.height * 0.62:
                height = doc.height * 0.62
                width = height * iw / ih
            s.append(KeepTogether([Spacer(1, 6), Image(str(path), width=width, height=height),
                                   Paragraph(f"Figure {fg}: {rl_inline(cap)}", CAP)]))
            after_heading = False
        if pending and len(s) > start:
            first = s.pop(start)
            s.insert(start, KeepTogether(pending + [first]))
            pending = []

    s.append(PageBreak())
    refs_h = Paragraph("References", H1)
    refs_h._toc = (0, "References")
    s.append(refs_h)
    for i, r in enumerate(REFS):
        s.append(Paragraph(f"[{i + 1}]&nbsp;&nbsp;{rl_inline(r)}", REF))
    doc.multiBuild(s)
    return out


if __name__ == "__main__":
    build_tex()
    print("wrote", build_pdf())
