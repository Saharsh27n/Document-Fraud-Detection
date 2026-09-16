"""
Stage 0 — Synthetic medical invoice dataset generator.

Usage:
    python generator/generate_dataset.py --n-invoices 400 --n-pairs 200 --seed 42 --noise light
"""

import argparse
import atexit
import base64
import csv
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

# Allow running from repo root or from generator/
REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = GENERATOR_DIR / "templates"
LOGOS_DIR = GENERATOR_DIR / "logos"
DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"

INVOICES_DIR.mkdir(parents=True, exist_ok=True)
LOGOS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Faker / content data
# ---------------------------------------------------------------------------
try:
    from faker import Faker
    _faker_available = True
except ImportError:
    _faker_available = False
    print("[WARN] faker not installed — using deterministic fallback names", file=sys.stderr)

MEDICAL_LINE_ITEMS = [
    ("Consultation Fee", 500, 2000),
    ("CBC (Complete Blood Count)", 300, 600),
    ("X-Ray Chest PA", 400, 900),
    ("ECG", 250, 500),
    ("Room Rent - General Ward", 1000, 3000),
    ("Room Rent - Private Ward", 3000, 8000),
    ("IV Fluids (per bottle)", 150, 350),
    ("Nursing Charges", 500, 1500),
    ("MRI Brain", 5000, 12000),
    ("CT Scan Abdomen", 4000, 9000),
    ("Physiotherapy Session", 300, 800),
    ("Blood Sugar Fasting", 80, 200),
    ("Urine Routine Examination", 100, 250),
    ("Liver Function Test", 600, 1200),
    ("Kidney Function Test", 700, 1400),
    ("Thyroid Profile (T3 T4 TSH)", 800, 1600),
    ("Lipid Profile", 600, 1200),
    ("Ultrasound Abdomen", 1500, 3500),
    ("2D Echocardiography", 2500, 5000),
    ("Chest Physiotherapy", 400, 900),
    ("Dietitian Consultation", 300, 700),
    ("Medication Charges", 200, 5000),
    ("Surgical Procedure", 5000, 50000),
    ("Anaesthesia Charges", 3000, 15000),
    ("Operation Theatre Charges", 5000, 30000),
    ("ICU Charges (per day)", 4000, 12000),
    ("Oxygen Charges", 200, 800),
    ("Blood Transfusion", 1500, 4000),
    ("Ambulance Charges", 500, 2000),
    ("Dressing / Wound Care", 200, 600),
    ("Pathology Report Fee", 150, 400),
    ("Radiology Reading Fee", 300, 700),
    ("Specialist Consultation", 800, 2500),
    ("Dental Extraction", 500, 2000),
    ("Eye Examination", 400, 1200),
    ("Dental X-Ray", 300, 600),
    ("Nebulisation Therapy", 150, 400),
    ("Audiometry Test", 500, 1000),
    ("Pregnancy Scan", 800, 2000),
    ("Vaccination", 500, 3000),
]

DIAGNOSES = [
    "Acute Gastroenteritis", "Type 2 Diabetes Mellitus", "Hypertension",
    "Community Acquired Pneumonia", "Urinary Tract Infection",
    "Dengue Fever", "Typhoid Fever", "Malaria", "Acute Appendicitis",
    "Inguinal Hernia", "Gallstones (Cholelithiasis)", "Kidney Stones (Nephrolithiasis)",
    "Peptic Ulcer Disease", "Coronary Artery Disease", "Asthma Exacerbation",
    "COPD Exacerbation", "Acute Myocardial Infarction", "Cerebrovascular Accident",
    "Fracture Right Femur", "Fracture Left Radius", "Lumbar Disc Prolapse",
    "Cervical Spondylosis", "Rheumatoid Arthritis", "Osteoarthritis Knee",
    "Anaemia — Iron Deficiency",
]

PAYMENT_MODES = ["Cash", "Credit Card", "Debit Card", "UPI", "Insurance", "NEFT/RTGS", "Cheque"]

TEMPLATE_IDS = ["t1_classic", "t2_modern", "t3_compact", "t4_gov", "t5_clinic", "t6_corporate"]

# Hospital name components for generating fake hospital names
HOSP_PREFIXES = ["City", "Metro", "Apollo", "Sunrise", "Green", "Life", "Prime", "National",
                 "Central", "Modern", "Pioneer", "Excel", "Apex", "Unity", "Global"]
HOSP_SUFFIXES = ["Hospital", "Medical Centre", "Healthcare", "Clinic", "Nursing Home",
                 "Multi-Specialty Hospital", "Institute of Medicine", "Super Speciality Hospital"]
INDIAN_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune",
                 "Ahmedabad", "Jaipur", "Lucknow", "Bhopal", "Nagpur", "Surat", "Vadodara"]

# ---------------------------------------------------------------------------
# Logo generation
# ---------------------------------------------------------------------------

def generate_logos(logos_dir: Path, n_logos: int = 10, rng: random.Random = None) -> list[Path]:
    """Generate simple placeholder logos with geometric shapes + monogram."""
    if rng is None:
        rng = random.Random(0)

    logo_paths = []
    colors = [
        (41, 98, 255), (220, 50, 50), (0, 150, 100), (180, 80, 200),
        (255, 140, 0), (0, 120, 180), (150, 30, 30), (30, 140, 30),
        (100, 60, 180), (200, 100, 50),
    ]
    monograms = ["AH", "BC", "CM", "DH", "EL", "FM", "GH", "HI", "JK", "KL"]

    for i in range(n_logos):
        logo_path = logos_dir / f"logo_{i:02d}.png"
        if logo_path.exists():
            logo_paths.append(logo_path)
            continue

        size = 128
        img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        color = colors[i % len(colors)]
        shape = i % 4

        if shape == 0:  # circle + cross (medical style)
            draw.ellipse([10, 10, 118, 118], fill=color, outline=color)
            draw.rectangle([54, 25, 74, 103], fill=(255, 255, 255))
            draw.rectangle([25, 54, 103, 74], fill=(255, 255, 255))
        elif shape == 1:  # square with rounded feel
            draw.rounded_rectangle([8, 8, 120, 120], radius=18, fill=color)
            draw.rounded_rectangle([28, 28, 100, 100], radius=10, fill=(255, 255, 255, 200))
        elif shape == 2:  # hexagon-ish (octagon)
            pts = []
            cx, cy, r = 64, 64, 54
            import math
            for k in range(8):
                angle = math.pi / 4 * k - math.pi / 8
                pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            draw.polygon(pts, fill=color)
        else:  # triangle on white
            draw.polygon([(64, 10), (118, 110), (10, 110)], fill=color)

        # Monogram
        mono = monograms[i % len(monograms)]
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), mono, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((size // 2 - tw // 2, size // 2 - th // 2), mono,
                  fill=(255, 255, 255), font=font)

        img.save(logo_path, "PNG")
        logo_paths.append(logo_path)

    return logo_paths


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------

def _num2words_inr(amount: int) -> str:
    """Simple rupees-to-words converter for amounts up to ~10 lakh."""
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
            "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
            "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def below_100(n):
        if n < 20:
            return ones[n]
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")

    def below_1000(n):
        if n < 100:
            return below_100(n)
        return ones[n // 100] + " Hundred" + (" " + below_100(n % 100) if n % 100 else "")

    if amount == 0:
        return "Zero Rupees Only"
    parts = []
    if amount >= 100000:
        parts.append(below_1000(amount // 100000) + " Lakh")
        amount %= 100000
    if amount >= 1000:
        parts.append(below_1000(amount // 1000) + " Thousand")
        amount %= 1000
    if amount:
        parts.append(below_1000(amount))
    return "Rupees " + " ".join(parts) + " Only"


_logo_data_uri_cache: dict = {}


def _logo_data_uri(logo_path: Path) -> str:
    """Embed the logo as a base64 data URI.

    Playwright's page.set_content() renders pages at the about:blank origin,
    and Chromium refuses to load file:// resources from a non-file:// page,
    so a filesystem path or file:// URI silently fails to load. A data URI
    sidesteps that cross-origin restriction entirely.
    """
    key = str(logo_path)
    if key not in _logo_data_uri_cache:
        b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        _logo_data_uri_cache[key] = f"data:image/png;base64,{b64}"
    return _logo_data_uri_cache[key]


def make_context(rng: random.Random, template_id: str, logo_path: Path,
                 hospital_name: str, hospital_address: str,
                 hospital_phone: str, hospital_email: str,
                 faker_obj=None) -> dict:
    """Build a Jinja2 template context dict with consistent financials."""

    def fake_name():
        if faker_obj:
            return faker_obj.name()
        first = rng.choice(["Amit", "Priya", "Ravi", "Sunita", "Vikram", "Anjali",
                             "Deepak", "Meera", "Suresh", "Kavitha"])
        last = rng.choice(["Sharma", "Patel", "Singh", "Kumar", "Verma", "Gupta",
                           "Joshi", "Nair", "Reddy", "Das"])
        return f"{first} {last}"

    def fake_doctor():
        prefix = rng.choice(["Dr.", "Dr."])
        name = fake_name()
        specialty = rng.choice(["MD", "MS", "MBBS", "DM", "MCh"])
        dept = rng.choice(["General Medicine", "Surgery", "Cardiology", "Orthopaedics",
                           "Gynaecology", "Paediatrics", "Neurology", "Urology"])
        return f"{prefix} {name} ({specialty}, {dept})"

    # Line items
    n_items = rng.randint(3, 12)
    chosen = rng.sample(MEDICAL_LINE_ITEMS, n_items)
    line_items = []
    for desc, low, high in chosen:
        rate = rng.randint(low, high)
        qty = rng.randint(1, 3)
        line_items.append({
            "description": desc,
            "qty": qty,
            "rate": f"{rate:,}",
            "amount": f"{rate * qty:,}",
            "_amount_int": rate * qty,
        })
    subtotal = sum(i["_amount_int"] for i in line_items)
    discount_pct = rng.choice([0, 5, 10, 0, 0])
    discount_amt = int(subtotal * discount_pct / 100)
    after_disc = subtotal - discount_amt
    tax_pct = rng.choice([5, 12, 18])
    tax_amt = int(after_disc * tax_pct / 100)
    total = after_disc + tax_amt

    # Clean up internal key
    for item in line_items:
        del item["_amount_int"]

    # Dates
    from datetime import date, timedelta
    today = date(2024, rng.randint(1, 12), rng.randint(1, 28))
    stay = rng.randint(1, 14)
    admission = today - timedelta(days=stay)
    discharge = today

    age = rng.randint(5, 85)
    sex = rng.choice(["M", "F"])

    patient_name = fake_name()
    patient_id = "PID" + "".join([str(rng.randint(0, 9)) for _ in range(6)])
    bill_number = f"BILL/{today.year}/{rng.randint(1000, 9999)}"

    return {
        "logo_path": _logo_data_uri(logo_path),
        "hospital_name": hospital_name,
        "hospital_address": hospital_address,
        "hospital_phone": hospital_phone,
        "hospital_email": hospital_email,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "age_sex": f"{age} / {sex}",
        "doctor_name": fake_doctor(),
        "diagnosis": rng.choice(DIAGNOSES),
        "bill_number": bill_number,
        "bill_date": today.strftime("%d-%b-%Y"),
        "admission_date": admission.strftime("%d-%b-%Y"),
        "discharge_date": discharge.strftime("%d-%b-%Y"),
        "payment_mode": rng.choice(PAYMENT_MODES),
        "line_items": line_items,
        "subtotal": f"{subtotal:,}",
        "discount_pct": discount_pct,
        "discount_amt": f"{discount_amt:,}",
        "tax_pct": tax_pct,
        "tax_amt": f"{tax_amt:,}",
        "total": f"{total:,}",
        "total_words": _num2words_inr(total),
    }


def make_hospital_identity(rng: random.Random, faker_obj=None) -> dict:
    """Return stable hospital identity (name, address, phone, email)."""
    prefix = rng.choice(HOSP_PREFIXES)
    suffix = rng.choice(HOSP_SUFFIXES)
    name = f"{prefix} {suffix}"
    city = rng.choice(INDIAN_CITIES)
    street_num = rng.randint(1, 200)
    road = rng.choice(["Main Road", "Hospital Road", "MG Road", "Ring Road", "NH-44"])
    address = f"{street_num}, {road}, {city} - {rng.randint(100000, 999999)}"
    phone = f"+91 {rng.randint(70000, 99999)}{rng.randint(10000, 99999)}"
    email = name.lower().replace(" ", "").replace("-", "")[:12] + f"@hospital.in"
    return {"name": name, "address": address, "phone": phone, "email": email}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_pw_ctx = None
_pw_browser = None


def _get_pw_browser():
    """Launch Chromium once and reuse it for every render() call in this process.

    Relaunching a fresh browser process per invoice (the original approach) is
    slow and occasionally races on shutdown, producing intermittent
    'Target page, context or browser has been closed' / capture-screenshot
    errors that silently mix ReportLab-rendered pages into the dataset.
    """
    global _pw_ctx, _pw_browser
    if _pw_browser is None:
        from playwright.sync_api import sync_playwright
        _pw_ctx = sync_playwright().start()
        _pw_browser = _pw_ctx.chromium.launch(headless=True)
        atexit.register(_close_pw_browser)
    return _pw_browser


def _close_pw_browser():
    global _pw_ctx, _pw_browser
    if _pw_browser is not None:
        try:
            _pw_browser.close()
        except Exception:
            pass
        _pw_browser = None
    if _pw_ctx is not None:
        try:
            _pw_ctx.stop()
        except Exception:
            pass
        _pw_ctx = None


def _render_playwright(html: str) -> Image.Image:
    """Render HTML string to PIL Image using a shared headless Chromium instance."""
    browser = _get_pw_browser()
    page = browser.new_page(
        viewport={"width": 794, "height": 1123},
        device_scale_factor=2,
    )
    try:
        page.set_content(html, wait_until="networkidle")
        # Allow fonts/images to load
        page.wait_for_timeout(200)
        content_height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": 794, "height": max(content_height, 1123)})
        png_bytes = page.screenshot(full_page=True)
    finally:
        page.close()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _render_reportlab(template_id: str, context: dict) -> Image.Image:
    """Fallback: render with ReportLab → PDF → PyMuPDF → PNG."""
    try:
        import fitz  # PyMuPDF
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError as e:
        raise RuntimeError(f"ReportLab/PyMuPDF not available: {e}")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    story = []

    title_style = ParagraphStyle("title", fontSize=16, fontName="Helvetica-Bold",
                                  spaceAfter=4, textColor=colors.HexColor("#1a3a5c"))
    small = ParagraphStyle("small", fontSize=8, fontName="Helvetica", leading=10)

    story.append(Paragraph(context["hospital_name"], title_style))
    story.append(Paragraph(context["hospital_address"], small))
    story.append(Paragraph(f"Tel: {context['hospital_phone']}  |  {context['hospital_email']}", small))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("MEDICAL INVOICE", ParagraphStyle("inv", fontSize=14,
                             fontName="Helvetica-Bold", alignment=1)))
    story.append(Spacer(1, 4*mm))

    meta = [
        ["Invoice No:", context["bill_number"], "Date:", context["bill_date"]],
        ["Patient:", context["patient_name"], "ID:", context["patient_id"]],
        ["Doctor:", context["doctor_name"], "Age/Sex:", context["age_sex"]],
        ["Diagnosis:", context["diagnosis"], "Payment:", context["payment_mode"]],
    ]
    mt = Table(meta, colWidths=[35*mm, 55*mm, 25*mm, 55*mm])
    mt.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
    ]))
    story.append(mt)
    story.append(Spacer(1, 6*mm))

    # Line items table
    header = ["Description", "Qty", "Rate (₹)", "Amount (₹)"]
    rows = [header] + [[i["description"], i["qty"], i["rate"], i["amount"]]
                        for i in context["line_items"]]
    rows += [
        ["", "", "Subtotal", context["subtotal"]],
        ["", "", f"Discount {context['discount_pct']}%", context["discount_amt"]],
        ["", "", f"GST {context['tax_pct']}%", context["tax_amt"]],
        ["", "", "TOTAL", context["total"]],
    ]
    col_widths = [90*mm, 15*mm, 30*mm, 35*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, len(context["line_items"])), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, len(context["line_items"])),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (-2, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (-2, -1), (-1, -1), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (-2, -1), (-1, -1), colors.white),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f"Amount in words: {context['total_words']}", small))

    doc.build(story)

    buf.seek(0)
    pdf_doc = fitz.open(stream=buf.read(), filetype="pdf")
    page = pdf_doc[0]
    mat = fitz.Matrix(2, 2)  # 2× scale
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()
    return img


def render(template_id: str, context: dict) -> Image.Image:
    """Render an invoice. Primary: Playwright. Fallback: ReportLab."""
    try:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        tmpl = env.get_template(f"{template_id}.html")
        html = tmpl.render(**context)
    except Exception as e:
        raise RuntimeError(f"Jinja2 rendering failed: {e}")

    try:
        return _render_playwright(html)
    except Exception as e:
        print(f"[WARN] Playwright failed ({e}), falling back to ReportLab", file=sys.stderr)
        return _render_reportlab(template_id, context)


# ---------------------------------------------------------------------------
# Scan-noise augmentation
# ---------------------------------------------------------------------------

def augment(img: Image.Image, rng: random.Random, noise_level: str) -> Image.Image:
    """Apply realistic scan noise. noise_level: none | light | heavy."""
    if noise_level == "none":
        return img

    w, h = img.size

    # Rotation ±1.5° (light) or ±3° (heavy)
    max_rot = 1.5 if noise_level == "light" else 3.0
    angle = rng.uniform(-max_rot, max_rot)
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # Perspective warp
    warp_strength = 0.008 if noise_level == "light" else 0.02
    dx = int(w * warp_strength)
    dy = int(h * warp_strength)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, dx), rng.randint(0, dy)),
        (w - rng.randint(0, dx), rng.randint(0, dy)),
        (w - rng.randint(0, dx), h - rng.randint(0, dy)),
        (rng.randint(0, dx), h - rng.randint(0, dy)),
    ]
    # Coefficients for PIL perspective
    coeffs = _find_perspective_coeffs(src, dst)
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Gaussian noise
    arr = np.array(img, dtype=np.float32)
    sigma = 3 if noise_level == "light" else 8
    noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Brightness / contrast jitter
    brightness_f = rng.uniform(0.88, 1.05) if noise_level == "light" else rng.uniform(0.75, 1.1)
    img = ImageEnhance.Brightness(img).enhance(brightness_f)
    contrast_f = rng.uniform(0.92, 1.05) if noise_level == "light" else rng.uniform(0.8, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast_f)

    # Occasional horizontal scan lines
    if rng.random() < 0.4:
        arr = np.array(img)
        step = rng.randint(30, 60)
        intensity = rng.randint(3, 12)
        for y in range(0, h, step):
            arr[y, :] = np.clip(arr[y, :].astype(int) - intensity, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # Light paper texture overlay
    if rng.random() < 0.3:
        tex = Image.new("RGB", (w, h), (255, 255, 255))
        tex_arr = np.array(tex, dtype=np.float32)
        tex_noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, 4, tex_arr.shape).astype(np.float32)
        tex_arr = np.clip(tex_arr + tex_noise, 230, 255).astype(np.uint8)
        tex_img = Image.fromarray(tex_arr).convert("RGB")
        img = Image.blend(img, tex_img, alpha=0.07)

    # JPEG recompression
    quality = rng.randint(70, 90)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


def _find_perspective_coeffs(src_points, dst_points):
    """Compute 8 perspective transform coefficients (PIL PERSPECTIVE format)."""
    matrix = []
    for p1, p2 in zip(dst_points, src_points):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array([p[0] for p in src_points] + [p[1] for p in src_points]).reshape(8)
    # Interleave B correctly
    B = np.array([c for p in src_points for c in p], dtype=float)
    res = np.linalg.solve(A, B)
    return np.array(res).flatten().tolist()


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _invoice_hash(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:8]


def generate_dataset(n_invoices: int, n_pairs: int, seed: int, noise: str):
    """Main entry point for dataset generation."""
    rng = random.Random(seed)
    np.random.seed(seed)

    # ---- Faker setup ----
    faker_obj = None
    if _faker_available:
        from faker import Faker
        faker_obj = Faker("en_IN")
        faker_obj.seed_instance(seed)

    # ---- Generate logos ----
    logo_paths = generate_logos(LOGOS_DIR, n_logos=10, rng=rng)
    print(f"[OK] Generated {len(logo_paths)} logos in {LOGOS_DIR}")

    # ---- Generate invoices ----
    # Each invoice: (template_id, logo_id, hospital_identity, context)
    invoices = []
    # Pregenerate hospital identities (one per template+logo combo, reused for pairs)
    n_hospitals = max(20, n_invoices // 10)
    hospital_pool = [make_hospital_identity(rng, faker_obj) for _ in range(n_hospitals)]

    print(f"[OK] Generating {n_invoices} invoices ...")
    manifest_rows = []

    for i in range(n_invoices):
        tmpl_id = TEMPLATE_IDS[i % len(TEMPLATE_IDS)]
        logo_idx = rng.randint(0, len(logo_paths) - 1)
        hosp = hospital_pool[rng.randint(0, len(hospital_pool) - 1)]
        logo_path = logo_paths[logo_idx]

        noise_seed = rng.randint(0, 2**31)
        item_rng = random.Random(noise_seed)

        ctx = make_context(
            rng=item_rng,
            template_id=tmpl_id,
            logo_path=logo_path,
            hospital_name=hosp["name"],
            hospital_address=hosp["address"],
            hospital_phone=hosp["phone"],
            hospital_email=hosp["email"],
            faker_obj=faker_obj,
        )

        img = render(tmpl_id, ctx)
        aug_rng = random.Random(noise_seed ^ 0xABCD)
        img = augment(img, aug_rng, noise)

        filename = f"invoice_{i:04d}_{tmpl_id}.png"
        out_path = INVOICES_DIR / filename
        img.save(str(out_path), "PNG")

        invoices.append({
            "idx": i,
            "filename": filename,
            "template_id": tmpl_id,
            "logo_id": logo_idx,
            "hospital_name": hosp["name"],
            "noise_seed": noise_seed,
            "rng_seed": seed,
        })
        manifest_rows.append(invoices[-1])

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{n_invoices}")

    # ---- Write manifest.csv ----
    manifest_path = DATA_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "filename", "template_id", "logo_id",
                                               "hospital_name", "noise_seed", "rng_seed"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[OK] Wrote {manifest_path}")

    # ---- Build pairs ----
    print(f"[OK] Building {n_pairs} pairs ...")
    pairs = []
    per_type = n_pairs // 4

    # Index invoices by template
    by_template = {t: [] for t in TEMPLATE_IDS}
    for inv in invoices:
        by_template[inv["template_id"]].append(inv)

    # 1. same_template_diff_content: same template, different invoices
    for _ in range(per_type):
        tmpl = rng.choice(TEMPLATE_IDS)
        pool = by_template[tmpl]
        if len(pool) < 2:
            continue
        a, b = rng.sample(pool, 2)
        pairs.append((a["filename"], b["filename"], "same_template_diff_content", 1))

    # 2. same_template_diff_logo: same template, but pick two from different hospital identities
    for _ in range(per_type):
        tmpl = rng.choice(TEMPLATE_IDS)
        pool = by_template[tmpl]
        if len(pool) < 2:
            continue
        # prefer different logos
        candidates = [(a, b) for a in pool for b in pool
                       if a["idx"] != b["idx"] and a["logo_id"] != b["logo_id"]]
        if not candidates:
            a, b = rng.sample(pool, 2)
        else:
            a, b = rng.choice(candidates)
        pairs.append((a["filename"], b["filename"], "same_template_diff_logo", 1))

    # 3. different_template: two different templates
    for _ in range(per_type):
        t1, t2 = rng.sample(TEMPLATE_IDS, 2)
        if not by_template[t1] or not by_template[t2]:
            continue
        a = rng.choice(by_template[t1])
        b = rng.choice(by_template[t2])
        pairs.append((a["filename"], b["filename"], "different_template", 0))

    # 4. same_issuer_updated_template: same hospital, different template — hard negative
    for _ in range(per_type):
        t1, t2 = rng.sample(TEMPLATE_IDS, 2)
        if not by_template[t1] or not by_template[t2]:
            continue
        a = rng.choice(by_template[t1])
        b = rng.choice(by_template[t2])
        pairs.append((a["filename"], b["filename"], "same_issuer_updated_template", 0))

    # Shuffle pairs for good measure
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    labels_path = DATA_DIR / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_a", "doc_b", "pair_type", "is_fraud"])
        for a, b, pt, fraud in pairs:
            writer.writerow([a, b, pt, fraud])
    print(f"[OK] Wrote {labels_path} ({len(pairs)} pairs)")
    print(f"\nDone! {n_invoices} invoices in {INVOICES_DIR}")
    return invoices, pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic medical invoice dataset")
    parser.add_argument("--n-invoices", type=int, default=400)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", choices=["none", "light", "heavy"], default="light")
    args = parser.parse_args()

    generate_dataset(args.n_invoices, args.n_pairs, args.seed, args.noise)
