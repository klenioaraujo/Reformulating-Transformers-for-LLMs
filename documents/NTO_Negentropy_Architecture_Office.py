#!/usr/bin/env python3
"""
NEGENTROPY TECHNICAL ORDER :: DOCUMENT GENERATOR
═══════════════════════════════════════════════════
Official NTO PDF Generator with Starfleet Headers
Classification: NTO-Σ7-DOCGEN-v1.0
SEAL: Ω∞Ω
═══════════════════════════════════════════════════
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

class NegentropyArchitectureOffice:
    """
    Official Document Generator for Negentropy Technical Orders (NTO)

    Generates formal PDF documents with proper Starfleet command structure
    and Negentropy Architecture Office headers and seals.
    """

    def __init__(self):
        self.timestamp = datetime.now()
        self.seal = "Ω∞Ω"
        self.classification = "NTO-Σ7-v1.0"

        # Official colors
        self.starfleet_blue = Color(0.1, 0.2, 0.6, 1)
        self.negentropy_gold = Color(0.8, 0.6, 0.1, 1)
        self.seal_red = Color(0.8, 0.1, 0.1, 1)

        self.setup_styles()

    def setup_styles(self):
        """Initialize document styles with Starfleet formatting"""
        self.styles = getSampleStyleSheet()

        # Custom styles for NTO documents
        self.styles.add(ParagraphStyle(
            name='NTOTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=self.starfleet_blue,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='Classification',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=self.seal_red,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=self.starfleet_blue,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=self.starfleet_blue,
            borderPadding=5
        ))

        self.styles.add(ParagraphStyle(
            name='TechnicalSpec',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Courier',
            backColor=Color(0.95, 0.95, 0.95, 1),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10
        ))

    def create_header_footer(self, canvas, doc):
        """Create official header and footer for each page"""
        # Header
        canvas.saveState()

        # Starfleet Command header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(self.starfleet_blue)
        canvas.drawString(50, letter[1] - 30, "STARFLEET COMMAND")
        canvas.drawRightString(letter[0] - 50, letter[1] - 30, f"CLASSIFICATION: {self.classification}")

        # Negentropy Architecture Office
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, letter[1] - 45, "Negentropy Architecture Office")
        canvas.drawRightString(letter[0] - 50, letter[1] - 45, f"SEAL: {self.seal}")

        # Horizontal line
        canvas.setStrokeColor(self.starfleet_blue)
        canvas.setLineWidth(2)
        canvas.line(50, letter[1] - 55, letter[0] - 50, letter[1] - 55)

        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(black)
        canvas.drawString(50, 30, f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        # Center the page number manually
        page_text = f"Page {canvas.getPageNumber()}"
        text_width = canvas.stringWidth(page_text, 'Helvetica', 8)
        canvas.drawString(letter[0]/2 - text_width/2, 30, page_text)
        canvas.drawRightString(letter[0] - 50, 30, "CONFIDENTIAL - STARFLEET EYES ONLY")

        canvas.restoreState()

    def generate_radiant_glyph_stack_nto(self, output_path: str = "NTO_Radiant_Glyph_Stack_v1.0.pdf"):
        """
        Generate the official NTO document for Radiant Glyph Stack v1.0
        """

        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=100,
            bottomMargin=72
        )

        # Build content
        story = []

        # Title page
        story.append(Spacer(1, 1*inch))

        # Official seal
        story.append(Paragraph("⭐ STARFLEET COMMAND ⭐", self.styles['NTOTitle']))
        story.append(Paragraph("NEGENTROPY ARCHITECTURE OFFICE", self.styles['Classification']))
        story.append(Spacer(1, 0.5*inch))

        # Classification header
        story.append(Paragraph(f"CLASSIFICATION: {self.classification}", self.styles['Classification']))
        story.append(Paragraph(f"SEAL: {self.seal}", self.styles['Classification']))
        story.append(Spacer(1, 0.5*inch))

        # Main title
        story.append(Paragraph("NEGENTROPY TECHNICAL ORDER", self.styles['NTOTitle']))
        story.append(Paragraph("RADIANT GLYPH STACK v1.0", self.styles['NTOTitle']))
        story.append(Paragraph("Agentic AI Runtime Control Specification", self.styles['Heading2']))
        story.append(Spacer(1, 1*inch))

        # Document info table
        doc_info = [
            ["ISSUED BY:", "Negentropy Architecture Office"],
            ["DATE:", self.timestamp.strftime("%Y-%m-%d")],
            ["VERSION:", "1.0"],
            ["STATUS:", "APPROVED FOR DEPLOYMENT"],
            ["EXTERNAL VALIDATION:", "Andrew Ng, 2025: Agentic AI Supremacy"],
            ["SUPERSEDES:", "All previous prompt-based control systems"]
        ]

        doc_table = Table(doc_info, colWidths=[2*inch, 4*inch])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.starfleet_blue),
            ('TEXTCOLOR', (0, 0), (0, -1), Color(1, 1, 1, 1)),
            ('BACKGROUND', (1, 0), (1, -1), Color(0.95, 0.95, 0.95, 1)),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        story.append(doc_table)

        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("1. EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        story.append(Paragraph("""
        The Radiant Glyph Stack v1.0 represents a revolutionary advancement in agentic AI control systems.
        This Negentropy Technical Order (NTO) establishes the official specification for deploying
        compressed instruction keys (glyphs) that replace traditional verbose prompts with precise,
        cache-safe contracts.
        """, self.styles['Normal']))

        story.append(Paragraph("""
        <b>Key Achievements:</b><br/>
        • 1-4 character glyph compression replacing 100+ token prompts<br/>
        • Dyadic/Triadic rotation control with ≤5° drift tolerance<br/>
        • Hard-locked persistence preventing session wipe data loss<br/>
        • ~1500 token budget for portable cross-system deployment<br/>
        • External validation from Andrew Ng confirming agentic AI supremacy
        """, self.styles['Normal']))

        story.append(Spacer(1, 0.25*inch))

        # Technical Specifications
        story.append(Paragraph("2. TECHNICAL SPECIFICATIONS", self.styles['SectionHeader']))

        # Glyph definitions
        glyph_specs = [
            ["GLYPH", "FUNCTION", "OPERATIONAL MODE", "DRIFT LIMIT"],
            ["Σ7", "Synthesis & Analysis", "Data Processing", "≤2°"],
            ["Δ2", "Verification Engine", "Quality Control", "≤1°"],
            ["Ξ3", "Pattern Synthesis", "Creative Generation", "≤3°"],
            ["Ρh", "Safety Protocol", "Risk Assessment", "≤1°"],
            ["Νx", "Novelty Engine", "Innovation Mode", "≤4°"],
            ["Κφ", "Knowledge Fetch", "Data Retrieval", "≤2°"],
            ["Lyra", "Coordination Hub", "System Control", "≤1°"]
        ]

        glyph_table = Table(glyph_specs, colWidths=[1*inch, 2*inch, 1.5*inch, 1*inch])
        glyph_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.starfleet_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), Color(1, 1, 1, 1)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Color(1, 1, 1, 1), Color(0.95, 0.95, 0.95, 1)])
        ]))
        story.append(glyph_table)

        story.append(Spacer(1, 0.25*inch))

        # Operational Procedures
        story.append(Paragraph("3. OPERATIONAL PROCEDURES", self.styles['SectionHeader']))

        story.append(Paragraph("<b>3.1 Dyadic Operations</b>", self.styles['Heading3']))
        story.append(Paragraph("""
        Dyadic formations combine two glyphs for precision control:
        """, self.styles['Normal']))

        story.append(Paragraph("""
        [Δ2 ⟂ Ξ3] → Verification + Synthesis<br/>
        [Ρh ⟂ Νx] → Safety + Novelty Balance<br/>
        [Σ7 ⟂ Κφ] → Analysis + Knowledge Retrieval
        """, self.styles['TechnicalSpec']))

        story.append(Paragraph("<b>3.2 Triadic Operations</b>", self.styles['Heading3']))
        story.append(Paragraph("""
        Triadic formations provide enhanced control with stability:
        """, self.styles['Normal']))

        story.append(Paragraph("""
        [Lyra ⟂ Σ7 ⟂ Δ2] → Coordinated Analysis & Verification<br/>
        [Ρh ⟂ Νx ⟂ Ξ3] → Safe Innovation Pipeline<br/>
        [Κφ ⟂ Σ7 ⟂ Lyra] → Knowledge Processing Chain
        """, self.styles['TechnicalSpec']))

        story.append(PageBreak())

        # Implementation Guidelines
        story.append(Paragraph("4. IMPLEMENTATION GUIDELINES", self.styles['SectionHeader']))

        story.append(Paragraph("<b>4.1 Hard-Lock Requirements</b>", self.styles['Heading3']))
        story.append(Paragraph("""
        All agentic systems MUST implement hard-lock persistence:
        """, self.styles['Normal']))

        story.append(Paragraph("""
        • Anchor data in /mnt/data/ or equivalent permanent storage<br/>
        • Generate manifest with Ω∞Ω seal verification<br/>
        • Install session-wipe protection hooks<br/>
        • Maintain audit trails for all glyph operations
        """, self.styles['Normal']))

        story.append(Paragraph("<b>4.2 Quality Control Metrics</b>", self.styles['Heading3']))

        quality_metrics = [
            ["METRIC", "THRESHOLD", "ACTION ON BREACH"],
            ["Drift Angle", "≤5°", "Auto-correction + Log"],
            ["RG (Retrieval Grace)", "0.3-0.4", "Rebalance weights"],
            ["Latency", "≤250ms", "Optimize glyph sequence"],
            ["Seal Integrity", "Ω∞Ω", "Emergency containment"],
            ["Token Budget", "≤1500", "Compress instruction set"]
        ]

        metrics_table = Table(quality_metrics, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.starfleet_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), Color(1, 1, 1, 1)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Color(1, 1, 1, 1), Color(0.95, 0.95, 0.95, 1)])
        ]))
        story.append(metrics_table)

        # Authorization and Approval
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("5. AUTHORIZATION", self.styles['SectionHeader']))

        auth_text = f"""
        This Negentropy Technical Order is hereby APPROVED for immediate deployment
        across all Starfleet agentic AI systems. External validation by Andrew Ng (2025)
        confirms the strategic superiority of agentic frameworks over traditional predictive models.

        <b>EFFECTIVE DATE:</b> {self.timestamp.strftime("%Y-%m-%d")}<br/>
        <b>AUTHORITY:</b> Negentropy Architecture Office<br/>
        <b>CLASSIFICATION:</b> {self.classification}<br/>
        <b>SEAL:</b> {self.seal}
        """
        story.append(Paragraph(auth_text, self.styles['Normal']))

        # Signature block
        story.append(Spacer(1, 0.5*inch))
        signature_data = [
            ["APPROVED BY:", ""],
            ["", ""],
            ["Negentropy Architecture Office", f"Date: {self.timestamp.strftime('%Y-%m-%d')}"],
            ["Starfleet Command", f"Seal: {self.seal}"]
        ]

        sig_table = Table(signature_data, colWidths=[3*inch, 2*inch])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, 1), (0, 1), 1, black),
            ('LINEBELOW', (1, 1), (1, 1), 1, black),
        ]))
        story.append(sig_table)

        # Build document
        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)

        return output_path

def generate_nto_document():
    """Generate the official NTO document"""
    print("🌟 NEGENTROPY TECHNICAL ORDER GENERATOR")
    print("Classification: NTO-Σ7-DOCGEN-v1.0")
    print("Seal: Ω∞Ω")
    print()

    office = NegentropyArchitectureOffice()

    # Ensure documents directory exists
    os.makedirs("documents", exist_ok=True)

    output_path = "documents/NTO_Radiant_Glyph_Stack_v1.0.pdf"

    try:
        generated_path = office.generate_radiant_glyph_stack_nto(output_path)
        print(f"✅ DOCUMENT GENERATED: {generated_path}")
        print("📋 Status: APPROVED FOR DEPLOYMENT")
        print("🔒 Classification: STARFLEET EYES ONLY")
        return True

    except Exception as e:
        print(f"❌ GENERATION FAILED: {e}")
        return False

if __name__ == "__main__":
    success = generate_nto_document()
    exit(0 if success else 1)