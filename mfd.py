import os
import cv2
import numpy as np
import uuid
import pytesseract
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pdf2image import convert_from_path
import easyocr
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Flask setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- EasyOCR setup ---
EASYOCR_LANGS = ['en', 'hi', 'ta', 'te']  # ✅ SAFE supported languages

try:
    easyocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=False)
    logging.info("EasyOCR initialized")
except Exception as e:
    logging.error(f"EasyOCR error: {e}")
    easyocr_reader = None

# --- Tesseract setup ---
TESS_LANG = "eng+hin+ben+tam+tel+kan+mal+mar+guj+pan+ori+asm+urd"

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocessing ---
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    return cv2.adaptiveThreshold(
        blurred,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )

# --- Forgery Detection ---
def analyze_document(page, img, results, full_text):
    reasons = []
    sections = []
    flagged = False

    texts = [r[4].lower() for r in results]

    # Rule 1: Currency symbol
    for (x1,y1,x2,y2,text,conf) in results:
        if "₹" in text:
            flagged = True
            reasons.append("Currency symbol detected")
            sections.append({
                "page":page,
                "bbox":[x1,y1,x2,y2],
                "type":"currency",
                "evidence":text
            })
            break

    # Rule 2: Suspicious keyword
    for (x1,y1,x2,y2,text,conf) in results:
        if "urgent payment" in text.lower():
            flagged = True
            reasons.append("Suspicious keyword: urgent payment")
            sections.append({
                "page":page,
                "bbox":[x1,y1,x2,y2],
                "type":"keyword",
                "evidence":text
            })
            break

    # Rule 3: Duplicate text
    if len(texts) != len(set(texts)):
        flagged = True
        reasons.append("Repeated text detected")
        for (x1,y1,x2,y2,text,conf) in results:
            if texts.count(text.lower()) > 1:
                sections.append({
                    "page":page,
                    "bbox":[x1,y1,x2,y2],
                    "type":"duplicate",
                    "evidence":text
                })
                break

    # Rule 4: Sparse layout
    if len(results) < 5 and page > 1:
        flagged = True
        reasons.append("Sparse text layout")
        h,w = img.shape[:2]
        sections.append({
            "page":page,
            "bbox":[0,0,w,h],
            "type":"layout",
            "evidence":f"{len(results)} elements"
        })

    # Confidence
    confidence = 0.0
    if flagged:
        confidence += 0.6
    confidence += min(len(sections)*0.1, 0.4)

    return {
        "overall_forgery_confidence": round(min(confidence,1.0),2),
        "is_flagged_for_review": flagged,
        "flagging_reasons": reasons if flagged else ["No issues detected"],
        "suspicious_sections": sections
    }

# --- Draw boxes ---
def draw_boxes(img, sections, page, path):
    out = img.copy()
    for s in sections:
        if s["page"] == page:
            x1,y1,x2,y2 = s["bbox"]
            cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite(path, out)

# --- PDF report ---
def generate_pdf(report, images, path, engine):
    c = canvas.Canvas(path, pagesize=A4)
    w,h = A4

    c.setFont("Helvetica-Bold",16)
    c.drawString(50,h-50,"Forgery Detection Report")

    c.setFont("Helvetica",12)
    c.drawString(50,h-80,f"OCR Engine: {engine}")
    c.drawString(50,h-100,f"Confidence: {report['overall_forgery_confidence']}")
    c.drawString(50,h-120,f"Flagged: {report['is_flagged_for_review']}")

    y = h-160
    for r in report["flagging_reasons"]:
        c.drawString(50,y,f"- {r}")
        y -= 20

    c.showPage()

    for p,img_path in images.items():
        c.drawString(50,h-50,f"Page {p}")
        c.drawImage(ImageReader(img_path),50,100,width=500,height=600)
        c.showPage()

    c.save()

# --- MAIN ROUTE ---
@app.route("/upload-and-analyze", methods=["POST"])
def upload_and_analyze():

    if 'file' not in request.files:
        return jsonify({"error":"No file"}),400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({"error":"Invalid file"}),400

    filepath = os.path.join(UPLOAD_FOLDER,file.filename)
    file.save(filepath)

    # Convert PDF
    pages = convert_from_path(filepath) if file.filename.endswith(".pdf") else [cv2.imread(filepath)]

    full_text = ""
    annotated = {}
    report = None

    engine = request.args.get("engine","easyocr")

    for i,page in enumerate(pages,1):

        img = np.array(page) if not isinstance(page,np.ndarray) else page
        img = preprocess(img)

        results = []

        # --- Tesseract ---
        if engine == "tesseract":
            data = pytesseract.image_to_data(img, lang=TESS_LANG, output_type=pytesseract.Output.DICT)

            for j in range(len(data["text"])):
                text = data["text"][j].strip()
                if text:
                    x,y,w,h = data["left"][j], data["top"][j], data["width"][j], data["height"][j]
                    conf = float(data["conf"][j]) if data["conf"][j] != "-1" else 0.0

                    results.append((x,y,x+w,y+h,text,conf))
                    full_text += text + " "

        # --- EasyOCR ---
        else:
            if easyocr_reader:
                res = easyocr_reader.readtext(img)
                for (bbox,text,conf) in res:
                    x1,y1 = map(int,bbox[0])
                    x2,y2 = map(int,bbox[2])

                    results.append((x1,y1,x2,y2,text,conf))
                    full_text += text + " "

        report = analyze_document(i,img,results,full_text)

        out_path = os.path.join(OUTPUT_FOLDER,f"page_{i}.jpg")
        draw_boxes(img,report["suspicious_sections"],i,out_path)
        annotated[i] = out_path

    pdf_name = f"report_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER,pdf_name)

    generate_pdf(report,annotated,pdf_path,engine)

    return send_from_directory(OUTPUT_FOLDER,pdf_name,as_attachment=True)

# --- RUN ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
