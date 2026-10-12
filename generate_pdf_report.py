import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Skip header and footer on cover page
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#166534"))

        # Running Header
        self.drawString(54, 750, "MORUS-LENS: Mulberry Foliar Pathology & Sericulture Intelligence")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 750, "End-to-End System & Architecture Report")

        # Top border line
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(54, 743, 558, 743)

        # Bottom border line
        self.line(54, 45, 558, 45)

        # Running Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 32, "Author: Sathish Kumar (Data Scientist & AI Researcher · Tirupattur, TN, India)")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)

        self.restoreState()


def build_pdf_report(output_filename="MORUS_LENS_PROJECT_REPORT.pdf"):
    pdf_path = os.path.abspath(output_filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palettes
    primary = colors.HexColor("#14532d")
    secondary = colors.HexColor("#15803d")
    accent = colors.HexColor("#16a34a")
    text_dark = colors.HexColor("#1e293b")
    text_muted = colors.HexColor("#475569")
    bg_light = colors.HexColor("#f0fdf4")
    border_col = colors.HexColor("#bbf7d0")

    # Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=primary,
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=18,
        textColor=secondary,
        alignment=1,
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'Heading3_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=text_dark,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_dark,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#14532d")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("🌿 MORUS-LENS", title_style))
    story.append(Paragraph("Mulberry Leaf Pathology Intelligence & Sericulture Agronomic Portal", subtitle_style))
    story.append(Spacer(1, 15))

    # Meta banner box
    meta_html = """
    <b>Author & Lead Developer:</b> Sathish Kumar (Data Scientist & AI Researcher)<br/>
    <b>Location:</b> Tirupattur, Tamil Nadu, India<br/>
    <b>Email:</b> sathishkumar.ds2027@gmail.com &nbsp;|&nbsp; <b>WhatsApp:</b> +91 86105 25967<br/>
    <b>LinkedIn:</b> linkedin.com/in/sathish-kumar-datascientist/<br/>
    <b>Project Repository:</b> github.com/sathishkumar1612/Mulberry-Leaf-Disease-Prediction<br/>
    <b>Document Type:</b> Comprehensive End-to-End System & Architecture Report (v2.0)
    """
    meta_table = Table([[Paragraph(meta_html, callout_style)]], colWidths=[504])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#86efac")),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 25))

    # Abstract Box
    abstract_text = """
    <b>Executive Summary:</b><br/>
    <b>MORUS-LENS</b> is an intelligent agricultural computer vision and decision-support portal built specifically for sericulture (silk farming) cultivators, moriculture agronomists, and pathology researchers. The system enables a farmer to capture a mulberry leaf (<i>Morus alba / Morus indica</i>) photo using any smartphone. Within one second, the platform isolates the leaf blade from surrounding clutter, accurately diagnoses whether the leaf is <b>Disease Free</b>, afflicted with <b>Leaf Rust</b>, or afflicted with <b>Leaf Spot</b>, and provides a clinical treatment recipe along with a mandatory silkworm chemical feed-withholding calendar to prevent lethal silkworm poisoning.
    """
    abs_table = Table([[Paragraph(abstract_text, body_style)]], colWidths=[504])
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(abs_table)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: THE REAL-WORLD PROBLEM
    # =========================================================================
    story.append(Paragraph("1. Background & The Real-World Agricultural Problem", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "<b>What is Sericulture and Why is Mulberry Sacred?</b><br/>"
        "Sericulture is the agro-industry that produces raw silk, supporting millions of rural livelihoods across India (notably in Tamil Nadu, Karnataka, and Andhra Pradesh) and Asia. The sole engine of this industry is the domestic silkworm, <i>Bombyx mori</i>. Silkworms are <b>strictly monophagous</b>—meaning they eat exclusively fresh, tender leaves of the mulberry tree (<i>Morus alba</i> or <i>Morus indica</i>) and cannot survive on any other food source.",
        body_style
    ))
    story.append(Paragraph(
        "The nutritional chemistry and moisture of the mulberry leaf directly dictates the silk gland development of the larva, the cocoon shell weight, filament reelability, and resistance to catastrophic silkworm ailments such as <i>flacherie</i> and <i>grasserie</i>.",
        body_style
    ))

    story.append(Paragraph("The Major Foliar Pathologies:", h2_style))
    story.append(Paragraph("• <b>Mulberry Leaf Rust (<i>Cerotelium fici</i>):</b> Erupts into thousands of brownish-orange powdery spore mounds (uredinia) on the lower surface of the leaf blade. Causes rapid yellowing, severe dehydration, and premature leaf drop.", bullet_style))
    story.append(Paragraph("• <b>Mulberry Leaf Spot (<i>Pseudocercospora mori</i>):</b> Forms irregular dark brown necrotic lesions surrounded by yellow chlorotic halos that dry up and fall out into 'shot-holes', depleting crude protein by up to 40%.", bullet_style))

    story.append(Paragraph("The Deadly Hazard: The 'Pesticide Paradox'", h2_style))
    story.append(Paragraph(
        "When farmers notice fungal infections, they frequently spray chemical fungicides (e.g. Mancozeb, Carbendazim, Copper Oxychloride). However, chemical residues remain active on the foliage for days. If a farmer harvests these leaves too early and feeds them to larvae, <b>the silkworms suffer acute chemical poisoning and perish en masse within 24–48 hours</b>. Therefore, a diagnosis is useless without a strict <b>chemical feed-withholding calendar</b>. MORUS-LENS addresses this life-or-death challenge directly.",
        body_style
    ))

    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 2: EXHAUSTIVE FOLDER & FILE DIRECTORY BREAKDOWN
    # =========================================================================
    story.append(Paragraph("2. Exhaustive Folder Structure & File Purpose Breakdown", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "To ensure that anyone inspecting the codebase understands exactly why each component exists, the table below maps out every single file, its function, and its role in the application architecture:",
        body_style
    ))

    # Directory Table
    files_data = [
        [Paragraph("File / Folder", table_header_style), Paragraph("Type & Location", table_header_style), Paragraph("Purpose & Operational Role in Project", table_header_style)],
        
        [Paragraph("<b>app.py</b>", table_cell_style), Paragraph("Root Application", table_cell_style), 
         Paragraph("The core Flask server script. Coordinates URL routing, user authentication, SQLite database models, OpenCV Excess Green segmentation, neural network inference, and recommendation generation.", table_cell_style)],

        [Paragraph("<b>streamlit_app.py</b>", table_cell_style), Paragraph("Root Script", table_cell_style), 
         Paragraph("An alternative lightweight data science prototype allowing researchers to test the vision model rapidly without setting up user accounts.", table_cell_style)],

        [Paragraph("<b>train_advanced_model.py</b>", table_cell_style), Paragraph("Root Laboratory", table_cell_style), 
         Paragraph("The laboratory training script. Implements EfficientNetB0 transfer learning, heavy agricultural augmentations, cosine learning rate warmup, and class weighting.", table_cell_style)],

        [Paragraph("<b>mulberry_leaf_...keras</b>", table_cell_style), Paragraph("Model Weights (39.7 MB)", table_cell_style), 
         Paragraph("The pre-trained deep neural network weights storing millions of mathematical coefficients that recognize leaf textures.", table_cell_style)],

        [Paragraph("<b>requirements.txt</b>", table_cell_style), Paragraph("Configuration", table_cell_style), 
         Paragraph("Specifies all required Python libraries (Flask, SQLAlchemy, OpenCV, Pillow, reportlab, streamlit).", table_cell_style)],

        [Paragraph("<b>instance/<br/>mulberry_app.db</b>", table_cell_style), Paragraph("Database", table_cell_style), 
         Paragraph("Relational SQLite database storing users (with PBKDF2 hashed passwords), scan histories, and inquiries submitted via contact desk.", table_cell_style)],

        [Paragraph("<b>static/css/<br/>style.css</b>", table_cell_style), Paragraph("Design System", table_cell_style), 
         Paragraph("Complete CSS stylesheet defining emerald sericulture styling, responsive mobile drawer, animated progress bars, and badges.", table_cell_style)],

        [Paragraph("<b>static/uploads/</b>", table_cell_style), Paragraph("Asset Storage", table_cell_style), 
         Paragraph("Partitioned file archive storing uploaded leaf images isolated by user ID (static/uploads/&lt;user_id&gt;/).", table_cell_style)],

        [Paragraph("<b>templates/<br/>base.html</b>", table_cell_style), Paragraph("Master Template", table_cell_style), 
         Paragraph("The structural skeleton for all pages. Contains top navigation bar, mobile hamburger menu, alerts, and footer.", table_cell_style)],

        [Paragraph("<b>templates/<br/>dashboard.html</b>", table_cell_style), Paragraph("Main Screen", table_cell_style), 
         Paragraph("The diagnostic scanner screen with drag-drop, mobile camera shutter, live animated probability gauges, and printable report.", table_cell_style)],

        [Paragraph("<b>templates/<br/>history.html</b>", table_cell_style), Paragraph("History Portal", table_cell_style), 
         Paragraph("User's farm diagnostic journal with filter tabs (All, Healthy, Rust, Spot, Unknown) and inspection modal pop-up.", table_cell_style)],

        [Paragraph("<b>templates/<br/>guide.html</b>", table_cell_style), Paragraph("Botanical Handbook", table_cell_style), 
         Paragraph("Organic Moriculture Herbarium featuring a 4-step dichotomous diagnostic key, specimen profiles, and feeding calendar.", table_cell_style)],

        [Paragraph("<b>templates/<br/>contact.html</b>", table_cell_style), Paragraph("Support Desk", table_cell_style), 
         Paragraph("Developer Sathish Kumar's contact hub with direct 1-tap WhatsApp (+91 86105 25967), email, LinkedIn, and inquiry ticketing.", table_cell_style)],

        [Paragraph("<b>templates/<br/>login.html & register.html</b>", table_cell_style), Paragraph("Auth Portal", table_cell_style), 
         Paragraph("Secure sign-in and account registration pages. Legacy demo accounts are purged for strict private user data separation.", table_cell_style)],
    ]

    dir_table = Table(files_data, colWidths=[100, 95, 309])
    dir_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(dir_table)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 3: END-TO-END ARCHITECTURE & DATA FLOW
    # =========================================================================
    story.append(Paragraph("3. End-to-End System Architecture & Data Flow", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "<b>The Lifecycle of an Image Diagnosis:</b><br/>"
        "From the moment a farmer holds their smartphone up to a mulberry tree to the moment a diagnostic report appears, the request travels through 6 tightly coordinated phases:",
        body_style
    ))

    story.append(Paragraph("1. <b>Image Acquisition:</b> The farmer clicks or selects an image on <code>dashboard.html</code>. Tapping the upload zone on a mobile device immediately opens the rear phone camera (<code>capture='environment'</code>).", bullet_style))
    story.append(Paragraph("2. <b>Secure Transmission:</b> JavaScript sends an asynchronous multipart HTTP POST request to the <code>/predict</code> endpoint on the Flask server.", bullet_style))
    story.append(Paragraph("3. <b>Disk Isolation:</b> <code>app.py</code> validates user authentication via <code>Flask-Login</code>, assigns a secure UUID filename, and saves the file to <code>static/uploads/&lt;user_id&gt;/</code>.", bullet_style))
    story.append(Paragraph("4. <b>Botanical Computer Vision Segmentation:</b> The image is passed to OpenCV in <code>analyze_botanical_pathology()</code>. The Excess Green formula (<code>2G - R - B</code>) and morphological contour tracking isolate the green leaf blade while discarding hands, tables, and backgrounds.", bullet_style))
    story.append(Paragraph("5. <b>Neural Network Inference & OOD Gating:</b> The normalized tensor is evaluated by the neural network. If confidence is below 70% or the margin over the second-place class is below 20%, the Out-of-Distribution filter re-routes the result to <b>Unknown / Ambiguous</b>.", bullet_style))
    story.append(Paragraph("6. <b>Persistence & UI Rendering:</b> The diagnosis is saved in the SQLite <code>predictions</code> table and returned as JSON. The browser animates the probability bars, displays the prescription badge, and provides a PDF print button.", bullet_style))

    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 4: HOW THE AI WORKS (NON-TECHNICAL ANALOGY)
    # =========================================================================
    story.append(Paragraph("4. How the AI Works (Explained in Simple Terms)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "For non-technical readers, the Artificial Intelligence inside MORUS-LENS works in three simple steps:",
        body_style
    ))

    story.append(Paragraph("Step A: The 'Sunglasses' Filter (Isolating the Leaf)", h2_style))
    story.append(Paragraph(
        "When a farmer places a leaf on a wooden table or holds it with fingers, ordinary AI programs get confused by the skin or wood colors. MORUS-LENS applies mathematical 'sunglasses' called the <b>Excess Green Index (2G - R - B)</b>. Living chlorophyll reflects massive green light and absorbs red and blue light. By calculating <code>2 × Green - Red - Blue</code>, the plant leaf turns glowing white, while tables, fingers, and shadows turn black. The system draws a boundary around the leaf and ignores everything outside it.",
        body_style
    ))

    story.append(Paragraph("Step B: The 'Magnifying Glass' (Examining Symptoms)", h2_style))
    story.append(Paragraph(
        "Inside the isolated leaf, the computer measures the color spectrum. Fungal Rust produces warm amber-orange pustules in HSV color space, while Leaf Spot produces dark sunken dead tissue (necrosis) surrounded by chlorotic yellow halos. Healthy areas display uniform, bright chlorophyll.",
        body_style
    ))

    story.append(Paragraph("Step C: The Deep Learning Brain (The Specialist Doctor)", h2_style))
    story.append(Paragraph(
        "Finally, the image enters a Convolutional Neural Network (EfficientNetB0). The network acts like layers of specialist detectives: early layers check veins and edges; middle layers look for spore circles; deep layers recognize complex disease signatures learned from thousands of mulberry samples.",
        body_style
    ))

    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 5: THE 4 DISEASE CLASSES & SILKWORM SAFETY CALENDAR
    # =========================================================================
    story.append(Paragraph("5. Pathological Classes & Silkworm Feeding Withholding Calendar", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    # Disease & Safety Table
    disease_table_data = [
        [Paragraph("Clinical Condition", table_header_style), Paragraph("Typical Symptoms", table_header_style), Paragraph("Agronomic Control", table_header_style), Paragraph("Silkworm Feeding Safety Window", table_header_style)],

        [Paragraph("🌿 <b>Disease Free</b><br/><i>(Healthy Morus)</i>", table_cell_style), 
         Paragraph("Smooth vibrant green blade, intact veins, glossy waxy cuticle.", table_cell_style), 
         Paragraph("Split NPK (300:120:120 kg/ha/yr) + 20 tons Farm Yard Manure.", table_cell_style), 
         Paragraph("<b>0 Days — Safe for immediate feeding</b>", table_cell_style)],

        [Paragraph("⚠️ <b>Potential Leaf Rust</b><br/><i>(Cerotelium fici)</i>", table_cell_style), 
         Paragraph("Brownish-orange powdery spore mounds (uredinia) on underside.", table_cell_style), 
         Paragraph("Spray Mancozeb 0.2% (2g/L) or Copper Oxychloride 0.2% (2g/L).", table_cell_style), 
         Paragraph("<b>10–14 Days Withholding<br/>DO NOT FEED SPRAYED LEAVES!</b>", table_cell_style)],

        [Paragraph("🚨 <b>Potential Leaf Spot</b><br/><i>(Pseudocercospora mori)</i>", table_cell_style), 
         Paragraph("Dark brown necrotic spots with yellow halos; shot-holes.", table_cell_style), 
         Paragraph("Spray Bavistin (Carbendazim 0.1%, 1g/L) or Chlorothalonil 0.2%.", table_cell_style), 
         Paragraph("<b>10–12 Days Withholding<br/>DO NOT FEED SPRAYED LEAVES!</b>", table_cell_style)],

        [Paragraph("❓ <b>Unknown / Ambiguous</b><br/><i>(Out-of-Distribution)</i>", table_cell_style), 
         Paragraph("Blurry image, non-mulberry leaf, or novel ambiguous pathogen.", table_cell_style), 
         Paragraph("Retake in daylight. Consult Sericulture Extension Officer.", table_cell_style), 
         Paragraph("<b>Quarantine foliage until confirmed</b>", table_cell_style)],
    ]

    dis_table = Table(disease_table_data, colWidths=[105, 130, 135, 134])
    dis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(dis_table)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 6: RECENT UPGRADES & HOW TO RUN
    # =========================================================================
    story.append(Paragraph("6. Key System Upgrades & How to Run the Application", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_col, spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("Major Upgrades Engineered by Sathish Kumar:", h2_style))
    story.append(Paragraph("1. <b>Project Rebranding:</b> Renamed all views, templates, navigation headers, and documentation to <b>MORUS-LENS</b>.", bullet_style))
    story.append(Paragraph("2. <b>Demo Account Purge:</b> Removed all pre-seeded demo accounts from database startup and templates, enforcing true multi-user privacy.", bullet_style))
    story.append(Paragraph("3. <b>Mobile Responsiveness:</b> Built a collapsible hamburger drawer, fluid touch targets, and smartphone camera shutter integration.", bullet_style))
    story.append(Paragraph("4. <b>Organic Botanical Field Handbook:</b> Replaced artificial blocks with an organic herbarium featuring dichotomous diagnostic keys.", bullet_style))
    story.append(Paragraph("5. <b>Dedicated Contact Desk:</b> Integrated Sathish Kumar's profile with direct 1-tap WhatsApp (+91 86105 25967), email, LinkedIn, and SQLite ticketing.", bullet_style))
    story.append(Paragraph("6. <b>Botanical Hull Segmentation:</b> Implemented Excess Green OpenCV morphology to eliminate false positives from white tables and holding fingers.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Step-by-Step Installation & Execution Guide:", h2_style))

    install_code = """
    <b># 1. Clone the repository and enter directory</b><br/>
    git clone https://github.com/sathishkumar1612/Mulberry-Leaf-Disease-Prediction.git<br/>
    cd Mulberry-Leaf-Disease-Prediction<br/><br/>
    <b># 2. Activate Python Virtual Environment</b><br/>
    myenv\\Scripts\\activate  <i>(Windows)</i> &nbsp;&nbsp;|&nbsp;&nbsp; source myenv/bin/activate  <i>(Linux/macOS)</i><br/><br/>
    <b># 3. Install Requirements & Start Server</b><br/>
    pip install -r requirements.txt<br/>
    python app.py<br/><br/>
    <b># 4. Access the Web Application:</b> Open browser at <b>http://127.0.0.1:5000</b>
    """
    code_table = Table([[Paragraph(install_code, body_style)]], colWidths=[504])
    code_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(code_table)

    story.append(Spacer(1, 15))

    # Developer Sign-off Box
    dev_box = """
    <b>Developer & Project Credits:</b><br/>
    <b>Lead Developer & AI Architect:</b> Sathish Kumar (Data Scientist & AI Researcher)<br/>
    <b>Location:</b> Tirupattur, Tamil Nadu, India<br/>
    <b>Email:</b> sathishkumar.ds2027@gmail.com &nbsp;|&nbsp; <b>WhatsApp:</b> +91 86105 25967<br/>
    <b>LinkedIn:</b> linkedin.com/in/sathish-kumar-datascientist/<br/>
    <b>GitHub:</b> github.com/sathishkumar1612/Mulberry-Leaf-Disease-Prediction<br/><br/>
    <i>MORUS-LENS — Empowering Sericulture Farmers with Lab-Grade Agricultural Artificial Intelligence.</i>
    """
    dev_table = Table([[Paragraph(dev_box, callout_style)]], colWidths=[504])
    dev_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor("#15803d")),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(dev_table)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

if __name__ == "__main__":
    build_pdf_report()
