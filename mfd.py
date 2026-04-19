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

# --- EasyOCR (LIGHT for detection only) ---
try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    logging.info("EasyOCR initialized")
except Exception as e:
    logging.error(f"EasyOCR error: {e}")
    easyocr_reader = None

# --- Tesseract full language support ---
TESS_LANG = "eng+hin+ben+tam+tel+kan+mal+mar+guj+pan+ori+asm+urd"

# --- File helper ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocess ---
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    return cv2.adaptiveThreshold(
        blurred,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )

# --- AUTO LANGUAGE DETECTION ---
def detect_language(text):
    for ch in text:
        code = ord(ch)

        if 0x0900 <= code <= 0x097F:
            return "hin"
        elif 0x0B80 <= code <= 0x0BFF:
            return "tam"
        elif 0x0C00 <= code <= 0x0C7F:
            return "tel"
        elif 0x0C80 <= code <= 0x0CFF:
            return "kan"
        elif 0x0D00 <= code <= 0x0D7F:
            return "mal"
        elif 0x0980 <= code <= 0x09FF:
            return "ben"
        elif 0x0A80 <= code <= 0x0AFF:
            return "guj"
        elif 0x0A00 <= code <= 0x0A7F:
            return "pan"
        elif 0x0B00 <= code <= 0x0B7F:
            return "ori"
        elif 0x0780 <= code <= 0x07BF:
            return "urd"

    return "eng"

# --- Forgery Detection ---
def analyze_document(page, img, results, full_text):
    reasons = []
    sections = []
    flagged = False

    texts = [r[4].lower() for r in results]

    if any("₹" in t for t in texts):
        flagged = True
        reasons.append("Currency symbol detected")

    if any("urgent payment" in t for t in texts):
        flagged = True
        reasons.append("Suspicious keyword detected")

    if len(texts) != len(set(texts)):
        flagged = True
        reasons.append("Repeated text detected")

    if len(results) < 5 and page > 1:
        flagged = True
        reasons.append("Sparse layout")

    confidence = 0.6 if flagged else 0.1

    return {
        "overall_forgery_confidence": round(confidence,2),
        "is_flagged_for_review": flagged,
        "flagging_reasons": reasons if flagged else ["No issues detected"],
        "suspicious_sections": []
    }

# --- Draw boxes ---
def draw_boxes(img, sections, page, path):
    cv2.imwrite(path, img)

# --- PDF report ---
def generate_pdf(report, images, path, engine):
    c = canvas.Canvas(path, pagesize=A4)
    w,h = A4

    c.setFont("Helvetica-Bold",16)
    c.drawString(50,h-50,"Forgery Detection Report")

    c.setFont("Helvetica",12)
    c.drawString(50,h-80,f"OCR Engine: AUTO")
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

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({"error":"Invalid file"}),400

    filepath = os.path.join(UPLOAD_FOLDER,file.filename)
    file.save(filepath)

    pages = convert_from_path(filepath) if file.filename.endswith(".pdf") else [cv2.imread(filepath)]

    full_text = ""
    annotated = {}
    report = None

    for i,page in enumerate(pages,1):

        img = np.array(page) if not isinstance(page,np.ndarray) else page
        img = preprocess(img)

        results = []

        try:
            # --- Step 1: preview text ---
            preview = easyocr_reader.readtext(img, detail=0) if easyocr_reader else []
            preview_text = " ".join(preview)

            # --- Step 2: detect language ---
            lang = detect_language(preview_text)
            logging.info(f"Detected: {lang}")

            # --- Step 3: OCR using Tesseract ---
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )

            for j in range(len(data["text"])):
                text = data["text"][j].strip()
                if text:
                    x,y,w,h = data["left"][j], data["top"][j], data["width"][j], data["height"][j]
                    conf = float(data["conf"][j]) if data["conf"][j] != "-1" else 0.0

                    results.append((x,y,x+w,y+h,text,conf))
                    full_text += text + " "

        except Exception as e:
            logging.error(f"OCR failed: {e}")

        report = analyze_document(i,img,results,full_text)

        out_path = os.path.join(OUTPUT_FOLDER,f"page_{i}.jpg")
        draw_boxes(img,report["suspicious_sections"],i,out_path)
        annotated[i] = out_path

    pdf_name = f"report_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER,pdf_name)

    generate_pdf(report,annotated,pdf_path,"auto")

    return send_from_directory(OUTPUT_FOLDER,pdf_name,as_attachment=True)

# --- RUN ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
