"""Build the Month 1 experiment report from saved experiment artifacts.

Run from the repository root:
    conda run -n mse-ai python reports/build_month1_report.py

The final PDF is written to output/pdf/month1_experiment_report.pdf.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "month1_experiment_report.pdf"
RESULTS_ROOT = PROJECT_ROOT / "results"

INK = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E74B5")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GRAY = colors.HexColor("#F2F4F7")
MID_GRAY = colors.HexColor("#667085")
GRID = colors.HexColor("#CBD5E1")
GREEN = colors.HexColor("#1B6B50")
GOLD = colors.HexColor("#8A6100")
WHITE = colors.white


def load_metrics(subdirectory: str) -> dict:
    metrics_path = RESULTS_ROOT / subdirectory / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def percentage(value: float, digits: int = 2) -> str:
    """Format a fractional metric as a percentage for the report."""
    return f"{value * 100:.{digits}f}%"


def percentage_point_difference(later: float, earlier: float, digits: int = 2) -> str:
    """Format a difference between two fractional metrics in percentage points."""
    return f"{(later - earlier) * 100:.{digits}f}"


def register_fonts() -> tuple[str, str]:
    """Use DejaVu when available and fall back to built-in Helvetica."""
    regular_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
    ]
    regular_path = next((path for path in regular_candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), None)
    if regular_path and bold_path:
        pdfmetrics.registerFont(TTFont("ReportBody", str(regular_path)))
        pdfmetrics.registerFont(TTFont("ReportBodyBold", str(bold_path)))
        return "ReportBody", "ReportBodyBold"
    return "Helvetica", "Helvetica-Bold"


BODY_FONT, BOLD_FONT = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.4,
            leading=12.6,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.2,
            leading=10.4,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName=BOLD_FONT,
            fontSize=27,
            leading=31,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=13,
            leading=17,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=9.5,
            leading=11,
            textColor=GOLD,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=sample["Heading1"],
            fontName=BOLD_FONT,
            fontSize=16,
            leading=19,
            textColor=BLUE,
            spaceBefore=0,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName=BOLD_FONT,
            fontSize=12.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.7,
            leading=9.5,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=6,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.2,
            leading=12,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.8,
            leading=9.6,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=7.8,
            leading=9.6,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.5,
            leading=12,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.2,
            leading=10.5,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


STYLES = make_styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return p(text, "h1" if level == 1 else "h2")


def bullet_list(items: list[str], compact: bool = False) -> ListFlowable:
    style = STYLES["body_small" if compact else "body"]
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontName=BODY_FONT,
        bulletFontSize=6,
        spaceAfter=6,
    )


def styled_table(
    rows: list[list],
    widths: list[float],
    header: bool = True,
    font_size: float = 7.8,
    row_backgrounds: bool = True,
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        converted_row = []
        for cell in row:
            if hasattr(cell, "wrap"):
                converted_row.append(cell)
            else:
                style = "table_header" if header and row_index == 0 else "table"
                converted_row.append(Paragraph(str(cell), STYLES[style]))
        converted.append(converted_row)

    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if row_backgrounds:
        first_body_row = 1 if header else 0
        for row_index in range(first_body_row, len(rows)):
            if (row_index - first_body_row) % 2 == 1:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def callout(text: str, color=LIGHT_BLUE) -> Table:
    table = Table([[p(text, "callout")]], colWidths=[6.45 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def scaled_image(path: Path, width_inches: float, max_height_inches: float) -> Image:
    with PillowImage.open(path) as source_image:
        pixel_width, pixel_height = source_image.size
    width = width_inches * inch
    height = width * pixel_height / pixel_width
    max_height = max_height_inches * inch
    if height > max_height:
        height = max_height
        width = height * pixel_width / pixel_height
    return Image(str(path), width=width, height=height)


def draw_page(canvas, document) -> None:
    page_number = canvas.getPageNumber()
    canvas.saveState()
    if page_number > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(0.65 * inch, 10.48 * inch, 7.85 * inch, 10.48 * inch)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(0.65 * inch, 10.57 * inch, "MONTH 1 EXPERIMENT REPORT")
        canvas.drawRightString(7.85 * inch, 10.57 * inch, "FEDERATED UNLEARNING THESIS")
    canvas.setFont(BODY_FONT, 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawCentredString(4.25 * inch, 0.38 * inch, f"Page {page_number} of 8")
    canvas.restoreState()


def build_report() -> None:
    week1 = load_metrics("month1_week1")
    week2 = load_metrics("month1_week2")
    mnist_cnn = load_metrics("month1_week3_mnist_cnn")
    cifar_cnn = load_metrics("month1_week3_cifar10_cnn")

    week1_accuracy = percentage(week1["accuracy"])
    week2_accuracy = percentage(week2["accuracy"])
    mnist_cnn_accuracy = percentage(mnist_cnn["accuracy"])
    cifar_cnn_accuracy = percentage(cifar_cnn["accuracy"])
    mlp_gain = percentage_point_difference(week2["accuracy"], week1["accuracy"])
    mnist_cnn_mlp_gain = percentage_point_difference(
        mnist_cnn["accuracy"], week2["accuracy"]
    )
    mnist_cnn_linear_gain = percentage_point_difference(
        mnist_cnn["accuracy"], week1["accuracy"]
    )
    week2_last_epoch = week2["epoch_history"][-1]
    week2_train_validation_gap = percentage_point_difference(
        week2_last_epoch["train_accuracy"], week2_last_epoch["validation_accuracy"]
    )

    per_class = cifar_cnn["per_class_accuracy"]
    sorted_classes = sorted(per_class.items(), key=lambda item: item[1])
    hardest_classes = sorted_classes[:3]
    easiest_classes = sorted(sorted_classes[-2:], key=lambda item: item[1], reverse=True)
    class_names = cifar_cnn["class_names"]
    confusion = cifar_cnn["confusion_matrix"]
    largest_error_count, largest_true_index, largest_predicted_index = max(
        (confusion[true_index][predicted_index], true_index, predicted_index)
        for true_index in range(len(class_names))
        for predicted_index in range(len(class_names))
        if true_index != predicted_index
    )
    reverse_error_count = confusion[largest_predicted_index][largest_true_index]
    largest_true_class = class_names[largest_true_index]
    largest_predicted_class = class_names[largest_predicted_index]
    largest_true_total = sum(confusion[largest_true_index])
    archive = cifar_cnn["dataset_archive"]
    runtime = mnist_cnn["runtime"]
    mnist_feature_example = mnist_cnn["feature_map_example"]
    cifar_feature_example = cifar_cnn["feature_map_example"]
    prepared_date = date.today().strftime("%d %B %Y")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="Month 1 Experiment Report - Federated Unlearning Thesis",
        author="Federated Unlearning Thesis Project",
        subject="Centralized ML and DL baselines on MNIST and CIFAR-10",
    )

    story = []

    # Page 1 - Editorial cover.
    story.extend(
        [
            Spacer(1, 0.72 * inch),
            p("FEDERATED UNLEARNING THESIS", "kicker"),
            p("Month 1 Experiment Report", "title"),
            p("Centralized ML and Deep Learning Foundations", "subtitle"),
            HRFlowable(width="72%", thickness=1.2, color=BLUE, spaceBefore=8, spaceAfter=22),
            styled_table(
                [
                    ["Period", "Month 1 - Weeks 1 to 4"],
                    ["Scope", "Python, experiment design, PyTorch, MLP, and CNN foundations"],
                    ["Datasets", "MNIST and CIFAR-10"],
                    [
                        "Environment",
                        f"Python {runtime['python_version']}, PyTorch "
                        f"{runtime['packages']['torch']} {runtime['device'].upper()}, "
                        f"fixed seed {mnist_cnn['random_seed']}",
                    ],
                    ["Prepared", prepared_date],
                ],
                [1.35 * inch, 5.1 * inch],
                header=False,
                row_backgrounds=True,
            ),
            Spacer(1, 0.35 * inch),
            callout(
                f"<b>Month 1 outcome.</b> The project progressed from a {week1_accuracy} "
                f"MNIST linear baseline to a {week2_accuracy} MLP and a "
                f"{mnist_cnn_accuracy} CNN, then established a {cifar_cnn_accuracy} "
                "centralized CNN reference on the harder CIFAR-10 task."
            ),
            Spacer(1, 0.28 * inch),
            p(
                "This report is written for a beginning AI student. It explains not only "
                "what ran, but why each experiment exists, how the measurements were kept "
                "reproducible, and what the results do and do not prove.",
                "body",
            ),
            PageBreak(),
        ]
    )

    # Page 2 - Executive overview.
    metric_cards = Table(
        [
            [
                p(f"<font size='18' color='#1B6B50'><b>{week1_accuracy}</b></font><br/>MNIST logistic regression", "metric"),
                p(f"<font size='18' color='#1B6B50'><b>{week2_accuracy}</b></font><br/>MNIST MLP", "metric"),
            ],
            [
                p(f"<font size='18' color='#1B6B50'><b>{mnist_cnn_accuracy}</b></font><br/>MNIST CNN", "metric"),
                p(f"<font size='18' color='#8A6100'><b>{cifar_cnn_accuracy}</b></font><br/>CIFAR-10 CNN", "metric"),
            ],
        ],
        colWidths=[3.15 * inch, 3.15 * inch],
        rowHeights=[0.75 * inch, 0.75 * inch],
    )
    metric_cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend(
        [
            heading("1. Executive Overview"),
            p(
                "The first month deliberately stays in centralized learning. Federated "
                "Learning and Federated Unlearning depend on the same training loop, loss, "
                "evaluation, and image-model concepts. Building those foundations first "
                "reduces the risk of mistaking a basic model bug for a federated-learning problem."
            ),
            metric_cards,
            Spacer(1, 8),
            heading("Learning progression", 2),
            styled_table(
                [
                    ["Week", "Model", "Dataset", "Accuracy", "Purpose"],
                    ["1", "Logistic regression", "MNIST", week1_accuracy, "Establish a simple linear floor."],
                    ["2", "128-unit MLP", "MNIST", week2_accuracy, "Learn forward pass, loss, backpropagation, optimizer, and learning rate."],
                    ["3", "Small CNN", "MNIST", mnist_cnn_accuracy, "Show why local image structure and learned filters help."],
                    ["3", "Small CNN", "CIFAR-10", cifar_cnn_accuracy, "Create a harder color-image baseline for later comparison."],
                ],
                [0.45 * inch, 1.25 * inch, 0.75 * inch, 0.72 * inch, 3.28 * inch],
            ),
            Spacer(1, 7),
            heading("Main interpretations", 2),
            bullet_list(
                [
                    f"The MLP's {mlp_gain} percentage-point gain over logistic regression shows that a hidden nonlinear layer learns useful structure beyond one linear decision boundary.",
                    f"The MNIST CNN adds another {mnist_cnn_mlp_gain} points, supporting the idea that preserving local pixel neighborhoods helps recognize handwritten shapes.",
                    "CIFAR-10 is not a failed MNIST result. It is a more difficult task with color, texture, backgrounds, and viewpoint variation, so its baseline must be interpreted separately.",
                    "All reported test numbers come from held-out official test sets after validation-based checkpoint selection.",
                ]
            ),
            callout(
                "<b>Scope discipline.</b> No FedAvg, FedProx, Non-IID partitioning, or "
                "unlearning algorithm was implemented in Month 1. Those remain gated behind "
                "verified centralized foundations."
            ),
            PageBreak(),
        ]
    )

    # Page 3 - Methodology.
    story.extend(
        [
            heading("2. Experimental Methodology"),
            p(
                "Every run follows one repeatable lifecycle. A saved JSON configuration is "
                "loaded first; the random seed and data split are fixed; the model trains only "
                "on training data; validation selects the checkpoint; the test set is evaluated "
                "once; and the script automatically writes metrics and visual evidence."
            ),
            styled_table(
                [
                    ["1. Configure", "2. Split data", "3. Train", "4. Validate", "5. Test and log"],
                    ["JSON settings and seed", "Train / validation / test", "Forward, loss, backward, update", "Choose best checkpoint", "Metrics, plots, timing, config"],
                ],
                [1.29 * inch] * 5,
            ),
            Spacer(1, 8),
            heading("Data protocol", 2),
            styled_table(
                [
                    ["Experiment", "Train", "Validation", "Test", "Augmentation"],
                    ["Week 1 MNIST", f"{week1['split']['train']:,}", f"{week1['split']['val']:,}", f"{week1['split']['test']:,}", "None"],
                    ["Week 2 MNIST", f"{week2['training_examples']:,}", f"{week2['validation_examples']:,}", f"{week2['test_examples']:,}", "None"],
                    ["Week 3 MNIST CNN", f"{mnist_cnn['training_examples']:,}", f"{mnist_cnn['validation_examples']:,}", f"{mnist_cnn['test_examples']:,}", "None"],
                    ["Week 3 CIFAR-10 CNN", f"{cifar_cnn['training_examples']:,}", f"{cifar_cnn['validation_examples']:,}", f"{cifar_cnn['test_examples']:,}", "Random crop and horizontal flip (training only)"],
                ],
                [1.55 * inch, 0.8 * inch, 0.85 * inch, 0.75 * inch, 2.5 * inch],
            ),
            Spacer(1, 7),
            heading("Metrics in plain language", 2),
            styled_table(
                [
                    ["Metric", "Meaning"],
                    ["Accuracy", "Fraction of all test examples assigned the correct class."],
                    ["Precision", "Of examples predicted as a class, how many actually belong to it."],
                    ["Recall", "Of examples that truly belong to a class, how many the model found."],
                    ["Macro F1", "Harmonic balance of precision and recall, computed per class and averaged equally across classes."],
                    ["Confusion matrix", "A class-by-class count showing exactly which labels are confused."],
                ],
                [1.35 * inch, 5.1 * inch],
            ),
            Spacer(1, 7),
            heading("Reproducibility controls", 2),
            bullet_list(
                [
                    "Seed 42 controls data splitting, initial weights, batch order, and augmentation randomness.",
                    "Every run reads a versioned JSON config and writes the required thesis fields automatically.",
                    "The best validation checkpoint is saved before final test evaluation.",
                    "Raw datasets and run outputs remain gitignored; source code and configs are versioned.",
                    "Centralized-only fields are explicit: one client, zero communication rounds, no target client, zero unlearning time, and zero communication cost.",
                ],
                compact=True,
            ),
            callout(
                "A fixed seed does not make one result universally true. It makes the run "
                "repeatable. Later benchmark stages must use multiple recorded seeds to estimate variability."
            ),
            PageBreak(),
        ]
    )

    # Page 4 - Week 1 and Week 2.
    week2_curve = scaled_image(
        RESULTS_ROOT / "month1_week2" / "learning_curve.png", 6.25, 2.45
    )
    story.extend(
        [
            heading("3. From a Linear Baseline to a Neural Network"),
            heading("Week 1 - Logistic regression", 2),
            p(
                "Logistic regression treats each 28 x 28 image as 784 input values and learns "
                "linear class boundaries. It is intentionally simple: a later model should beat "
                "it for a clear reason, not merely report an isolated percentage. The config-driven "
                f"rerun achieved <b>{week1['accuracy'] * 100:.2f}% accuracy</b> and "
                f"<b>{week1['f1'] * 100:.2f}% macro F1</b>."
            ),
            heading("Week 2 - Multilayer perceptron", 2),
            p(
                "The MLP adds a 128-unit hidden layer and ReLU activation. For each batch, forward "
                "propagation produces logits, cross-entropy measures the error, backpropagation "
                f"calculates gradients, and Adam updates {week2['trainable_parameters']:,} "
                f"trainable parameters using learning rate {week2['learning_rate']}."
            ),
            styled_table(
                [
                    ["Model", "Validation accuracy", "Test accuracy", "Macro F1", "CPU train time"],
                    ["Logistic regression", f"{week1['val_accuracy'] * 100:.2f}%", f"{week1['accuracy'] * 100:.2f}%", f"{week1['f1'] * 100:.2f}%", f"{week1['train_time_seconds']:.1f} s"],
                    ["MLP (best epoch)", f"{week2['best_validation_accuracy'] * 100:.2f}%", f"{week2['accuracy'] * 100:.2f}%", f"{week2['f1'] * 100:.2f}%", f"{week2['training_time_seconds']:.1f} s"],
                ],
                [1.45 * inch, 1.35 * inch, 1.15 * inch, 1.05 * inch, 1.45 * inch],
            ),
            Spacer(1, 6),
            week2_curve,
            p(f"Figure 1. Week 2 MLP learning curves. Both validation accuracy and validation loss improve across {week2['local_epochs']} epochs.", "caption"),
            callout(
                "<b>How to read the curves.</b> Falling loss means the predicted probability "
                "distribution is moving closer to the correct labels. Accuracy can stay unchanged "
                "for several updates even while loss improves, because loss also reflects prediction confidence."
            ),
            Spacer(1, 7),
            bullet_list(
                [
                    "Train data changes the weights; validation data selects among checkpoints; test data estimates final performance.",
                    f"A {week2_train_validation_gap}-point train-validation gap at epoch {week2_last_epoch['epoch']} does not indicate severe overfitting in this short run.",
                    f"The MLP's {week2_accuracy} test accuracy is {mlp_gain} points above the Week 1 baseline.",
                ],
                compact=True,
            ),
            PageBreak(),
        ]
    )

    # Page 5 - MNIST CNN.
    mnist_feature = scaled_image(
        RESULTS_ROOT / "month1_week3_mnist_cnn" / "feature_maps.png", 3.35, 3.25
    )
    story.extend(
        [
            heading("4. MNIST CNN Baseline"),
            p(
                "Unlike the MLP, a convolutional neural network preserves nearby-pixel structure. "
                "A learned 3 x 3 filter slides over the image; its responses form a feature map. "
                "Pooling then reduces spatial size while retaining strong responses."
            ),
            styled_table(
                [
                    ["Stage", "Output concept", "Beginner interpretation"],
                    ["Conv 1 + ReLU", "32 feature maps", "Learn simple local edges and strokes."],
                    ["Max pool", "Half width and height", "Keep strong responses with less computation."],
                    ["Conv 2 + ReLU", "64 feature maps", "Combine early patterns into richer shapes."],
                    ["Adaptive pool", "64 x 4 x 4", "Create a fixed-size representation."],
                    ["Classifier", "128 hidden units -> 10 logits", "Turn learned image features into digit scores."],
                ],
                [1.25 * inch, 1.5 * inch, 3.7 * inch],
            ),
            Spacer(1, 7),
            styled_table(
                [
                    ["Parameters", "Best epoch", "Validation accuracy", "Test accuracy", "Macro F1", "CPU time"],
                    [
                        f"{mnist_cnn['trainable_parameters']:,}",
                        str(mnist_cnn["best_epoch"]),
                        f"{mnist_cnn['best_validation_accuracy'] * 100:.2f}%",
                        f"{mnist_cnn['accuracy'] * 100:.2f}%",
                        f"{mnist_cnn['f1'] * 100:.2f}%",
                        f"{mnist_cnn['training_time_seconds'] / 60:.1f} min",
                    ],
                ],
                [1.05 * inch, 0.8 * inch, 1.35 * inch, 1.05 * inch, 1.0 * inch, 1.2 * inch],
            ),
            Spacer(1, 7),
            Table(
                [
                    [
                        mnist_feature,
                        p(
                            "<b>Feature-map interpretation</b><br/><br/>The input's true label is "
                            f"{mnist_feature_example['true_label']} and the prediction is "
                            f"{mnist_feature_example['predicted_label']}. Some filters respond to "
                            "strong strokes or contours, while others suppress most background pixels. "
                            "These maps are learned automatically from labels; they were not manually "
                            "programmed as digit detectors.<br/><br/><b>Measured improvement</b><br/><br/>"
                            f"The CNN reaches {mnist_cnn_accuracy} test accuracy: "
                            f"{mnist_cnn_mlp_gain} points above the MLP and "
                            f"{mnist_cnn_linear_gain} points above logistic regression.",
                            "body_small",
                        ),
                    ]
                ],
                colWidths=[3.45 * inch, 3.0 * inch],
            ),
            p("Figure 2. Eight first-layer feature maps for one MNIST test image.", "caption"),
            callout(
                "The CNN result supports a limited conclusion: local spatial processing helps on "
                "this MNIST configuration. It does not prove that every CNN architecture or every "
                "dataset will improve by the same amount."
            ),
            PageBreak(),
        ]
    )

    # Page 6 - CIFAR-10 result.
    cifar_curve = scaled_image(
        RESULTS_ROOT / "month1_week3_cifar10_cnn" / "learning_curve.png", 6.25, 2.7
    )
    story.extend(
        [
            heading("5. CIFAR-10 CNN Baseline"),
            p(
                "CIFAR-10 contains 32 x 32 color objects under varied backgrounds, positions, "
                "textures, and viewpoints. The same two-convolution architecture is used, except "
                "the first layer accepts three color channels. Training-only crop and horizontal "
                "flip augmentation make the task harder during optimization and improve invariance."
            ),
            styled_table(
                [
                    ["Parameters", "Epochs", "Best epoch", "Validation accuracy", "Test accuracy", "Macro F1", "CPU time"],
                    [
                        f"{cifar_cnn['trainable_parameters']:,}",
                        str(cifar_cnn["local_epochs"]),
                        str(cifar_cnn["best_epoch"]),
                        f"{cifar_cnn['best_validation_accuracy'] * 100:.2f}%",
                        f"{cifar_cnn['accuracy'] * 100:.2f}%",
                        f"{cifar_cnn['f1'] * 100:.2f}%",
                        f"{cifar_cnn['training_time_seconds'] / 60:.1f} min",
                    ],
                ],
                [0.9 * inch, 0.55 * inch, 0.7 * inch, 1.25 * inch, 1.0 * inch, 0.9 * inch, 1.15 * inch],
            ),
            Spacer(1, 6),
            cifar_curve,
            p(f"Figure 3. CIFAR-10 learning curves. Validation loss falls and validation accuracy rises throughout all {cifar_cnn['local_epochs']} epochs.", "caption"),
            callout(
                "<b>Why validation is above training.</b> Training accuracy is measured while random "
                "augmentation makes images harder and dropout disables part of the classifier. Both "
                "are off during validation. The difference is therefore expected and is not evidence "
                "of a data leak."
            ),
            Spacer(1, 7),
            styled_table(
                [
                    ["Class", "Accuracy", "Class", "Accuracy"],
                    ["Airplane", f"{per_class['airplane'] * 100:.1f}%", "Automobile", f"{per_class['automobile'] * 100:.1f}%"],
                    ["Bird", f"{per_class['bird'] * 100:.1f}%", "Cat", f"{per_class['cat'] * 100:.1f}%"],
                    ["Deer", f"{per_class['deer'] * 100:.1f}%", "Dog", f"{per_class['dog'] * 100:.1f}%"],
                    ["Frog", f"{per_class['frog'] * 100:.1f}%", "Horse", f"{per_class['horse'] * 100:.1f}%"],
                    ["Ship", f"{per_class['ship'] * 100:.1f}%", "Truck", f"{per_class['truck'] * 100:.1f}%"],
                ],
                [1.65 * inch, 1.0 * inch, 1.65 * inch, 1.0 * inch],
            ),
            PageBreak(),
        ]
    )

    # Page 7 - Error analysis.
    cifar_confusion = scaled_image(
        RESULTS_ROOT / "month1_week3_cifar10_cnn" / "confusion_matrix.png", 4.55, 4.55
    )
    cifar_features = scaled_image(
        RESULTS_ROOT / "month1_week3_cifar10_cnn" / "feature_maps.png", 2.05, 2.05
    )
    story.extend(
        [
            heading("6. Error Analysis and Compute Constraints"),
            p(
                "One overall percentage can hide systematic weaknesses. The CIFAR-10 confusion "
                "matrix shows where the model is reliable and where its learned representation is "
                "not yet discriminative enough."
            ),
            Table(
                [
                    [
                        cifar_confusion,
                        [
                            cifar_features,
                            p(
                                "Figure 5. First-layer responses for a test image with true label "
                                f"{cifar_feature_example['true_label']} and prediction "
                                f"{cifar_feature_example['predicted_label']}.",
                                "caption",
                            ),
                            p(
                                f"<b>Largest visible error:</b> {largest_error_count} of "
                                f"{largest_true_total:,} {largest_true_class}s are predicted as "
                                f"{largest_predicted_class}s, while {reverse_error_count} "
                                f"{largest_predicted_class}s are predicted as {largest_true_class}s.",
                                "body_small",
                            ),
                            p(
                                f"{hardest_classes[0][0].title()}, {hardest_classes[1][0]}, and "
                                f"{hardest_classes[2][0]} are the hardest classes "
                                f"({percentage(hardest_classes[0][1], 1)}, "
                                f"{percentage(hardest_classes[1][1], 1)}, and "
                                f"{percentage(hardest_classes[2][1], 1)}). "
                                f"{easiest_classes[0][0].title()} and {easiest_classes[1][0]} "
                                f"are the easiest ({percentage(easiest_classes[0][1], 1)} and "
                                f"{percentage(easiest_classes[1][1], 1)}).",
                                "body_small",
                            ),
                        ],
                    ]
                ],
                colWidths=[4.65 * inch, 1.8 * inch],
            ),
            p("Figure 4. CIFAR-10 test confusion matrix. Rows are true labels and columns are predictions.", "caption"),
            heading("What this means for later FL experiments", 2),
            bullet_list(
                [
                    "Per-class metrics must remain visible under IID and Non-IID partitions; overall accuracy alone could hide a client dominated by difficult animal classes.",
                    f"The {cifar_cnn['training_time_seconds'] / 60:.1f}-minute centralized CIFAR run warns against multiplying clients, rounds, and local epochs before MNIST FL is stable.",
                    "The current model is a baseline, not an optimized CIFAR architecture. Honest limitations are more useful than silently increasing model size or compute.",
                ],
                compact=True,
            ),
            callout(
                "<b>Integrity note.</b> The Toronto download stalled before training. The replacement "
                "Zenodo archive was accepted only after matching TorchVision's exact size "
                f"({archive['size_bytes']:,} bytes) and MD5 ({archive['md5']})."
            ),
            PageBreak(),
        ]
    )

    # Page 8 - SISA, conclusions, and Gate 1 readiness.
    story.extend(
        [
            heading("7. SISA Concept, Conclusions, and Gate Readiness"),
            p(
                "Week 4 reads only the motivation and concept of Bourtoule et al.'s SISA framework. "
                "No unlearning method is implemented at this stage. The detailed literature note "
                "uses the plan's fixed six-question format."
            ),
            styled_table(
                [
                    ["Question", "Concise answer"],
                    ["Problem", "Remove a training point's influence without paying full retraining cost."],
                    ["Gap", "Full retraining is strong but expensive; earlier approaches were limited for adaptive/stateful deep learning."],
                    ["Idea", "Shard data, train isolated models, slice incremental training with checkpoints, and aggregate predictions."],
                    ["Assumption", "SISA is designed before training; shards are disjoint; checkpoints and aggregation are available."],
                    ["Evaluation", "Purchase, MNIST, SVHN, ImageNet/Mini-ImageNet, and CIFAR-100 transfer; accuracy and unlearning speed."],
                    ["Limitation", "More shards can reduce accuracy, checkpoints cost storage, and SISA is centralized point-level unlearning rather than client-level FU."],
                ],
                [1.1 * inch, 5.35 * inch],
            ),
            Spacer(1, 7),
            heading("Month 1 conclusions", 2),
            bullet_list(
                [
                    "The project can train and evaluate PyTorch models with explicit forward, loss, backward, and optimizer steps.",
                    f"A small CNN is verified on both required centralized datasets: {mnist_cnn_accuracy} MNIST and {cifar_cnn_accuracy} CIFAR-10 accuracy.",
                    "Accuracy, precision, recall, macro F1, confusion matrices, seeds, checkpoints, configs, and timings are captured automatically.",
                    "The Month 1 code and results support progression to FL concepts only after the beginner Gate 1 explanation check is passed.",
                ],
                compact=True,
            ),
            heading("Known limitations", 2),
            p(
                "All neural-network results use one random seed and one CPU machine; they establish "
                "working baselines, not confidence intervals. CIFAR-10 uses a deliberately small model "
                "and ten epochs. The report contains no claim about FL convergence, Non-IID behavior, "
                "or forgetting because none of those stages has begun."
            ),
            callout(
                "<b>Gate 1 status:</b> technical artifacts are ready for review, but the README gate "
                "must remain unchecked until the student can explain the training loop, loss behavior, "
                "CNN feature maps, validation/test separation, and the observed accuracy changes in "
                "their own words."
            ),
            heading("References and project evidence", 2),
            p(
                "[1] Thesis roadmap: Ke_hoach_6_thang_Thac_si_Federated_Unlearning.md.<br/>"
                "[2] Bourtoule et al. (2021), <i>Machine Unlearning</i>, IEEE Symposium on Security "
                "and Privacy, DOI: 10.1109/SP40001.2021.00019; arXiv:1912.03817.<br/>"
                "[3] Saved configs: configs/month1_week1_mnist_logistic_regression.json, "
                "configs/month1_week2_mnist_mlp.json, and both configs/month1_week3_*_cnn.json files.<br/>"
                "[4] Measured outputs: results/month1_week1, month1_week2, "
                "month1_week3_mnist_cnn, and month1_week3_cifar10_cnn.",
                "body_small",
            ),
        ]
    )

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
