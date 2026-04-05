from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os


def generate_pdf_report(cluster_df, text_summaries, overall_summary):
    # Create PDF document
    doc = SimpleDocTemplate("cluster_report.pdf", pagesize=A4)

    # Get default styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = styles["Heading1"]
    section_style = styles["Heading2"]

    # Body text style with spacing
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        spaceAfter=10,
        leading=14
    )

    # Table text style (smaller so it fits properly)
    table_style_text = ParagraphStyle(
        "TableStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10
    )

    elements = []

    # =========================
    # TITLE
    # =========================
    elements.append(Paragraph("Cluster Analysis Report", title_style))
    elements.append(Spacer(1, 15))

    # =========================
    # OVERALL SUMMARY
    # =========================
    elements.append(Paragraph("Overall Summary", section_style))
    elements.append(Spacer(1, 10))

    for line in overall_summary:
        elements.append(Paragraph(line, body_style))

    elements.append(Spacer(1, 15))

    # =========================
    # CLUSTER TABLE
    # =========================
    elements.append(Paragraph("Cluster Overview", section_style))
    elements.append(Spacer(1, 10))

    # Table headers
    data = [[
        Paragraph("<b>Cluster</b>", table_style_text),
        Paragraph("<b>Size</b>", table_style_text),
        Paragraph("<b>Avg Distance</b>", table_style_text),
        Paragraph("<b>Standout Features</b>", table_style_text),
        Paragraph("<b>Nearest Cluster</b>", table_style_text)
    ]]

    # Add rows
    for _, row in cluster_df.iterrows():
        data.append([
            Paragraph(str(row["cluster"]), table_style_text),
            Paragraph(str(row["size"]), table_style_text),
            Paragraph(f"{row['average_distance_to_centroid']:.4f}", table_style_text),
            Paragraph(row["feature_description"], table_style_text),
            Paragraph(str(row["nearest_cluster"]), table_style_text),
        ])

    # Create table
    table = Table(data, colWidths=[50, 60, 80, 200, 80])

    # Style table
    table.setStyle(TableStyle([
        # Header styling
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        # Grid
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),

        # Alignment
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        # Alternating row colours
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),

        # Padding
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # =========================
    # DETAILED TEXT SUMMARIES
    # =========================
    elements.append(Paragraph("Detailed Cluster Insights", section_style))
    elements.append(Spacer(1, 10))

    for summary in text_summaries:
        elements.append(Paragraph(summary, body_style))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(elements)

    # =========================
    # AUTO-OPEN PDF
    # =========================
    try:
        os.startfile("cluster_report.pdf")  # Windows
    except:
        try:
            os.system("open cluster_report.pdf")  # Mac
        except:
            os.system("xdg-open cluster_report.pdf")  # Linux