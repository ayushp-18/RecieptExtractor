from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, re, tempfile, requests
from pdf2image import convert_from_path
import pytesseract

# ==========================
# CONFIG
# ==========================
POPPLER_BIN = r"D:\invoice-extractor\poppler\poppler-25.11.0\Library\bin"
TESSERACT_EXE = r"D:\orc\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ==========================
# HELPERS
# ==========================
def download_pdf(url: str) -> str:
    """Downloads a PDF from a URL and returns the local file path."""
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status_code}")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")


def clean_num(s):
    if not s:
        return None
    s = str(s).replace(",", "").replace("₹", "").replace("$", "")
    m = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(m[0]) if m else None


numeric_suffix_re = re.compile(r"""
    (?P<prefix>.*\S)\s+
    (?P<qty>\d+(?:\.\d+)?)\s+
    (?P<rate>[-\d,\.]+)\s+
    (?P<discount>[-\d,\.]+)\s+
    (?P<net>[-\d,\.]+)\s*$
""", re.VERBOSE)

amount_only_re = re.compile(r'(?P<prefix>.*\S)\s+(?P<net>[-\d,\.]+)\s*$')


def parse_page_text(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items = []
    buf_name = None

    for ln in lines:
        low = ln.lower()
        if any(x in low for x in ["description", "qty", "rate", "discount", "net amt", "total"]):
            buf_name = None
            continue

        m = numeric_suffix_re.match(ln)
        if m:
            items.append({
                "item_name": m.group("prefix").strip(),
                "item_quantity": clean_num(m.group("qty")),
                "item_rate": clean_num(m.group("rate")),
                "item_amount": clean_num(m.group("net"))
            })
            buf_name = None
            continue

        m2 = amount_only_re.match(ln)
        if buf_name and m2:
            items.append({
                "item_name": (buf_name + " " + m2.group("prefix")).strip(),
                "item_quantity": 1.0,
                "item_rate": None,
                "item_amount": clean_num(m2.group("net"))
            })
            buf_name = None
            continue

        buf_name = ln

    return items


JUNK_PATTERNS = [
    "pagewise line items",
    "response format",
    "item name",
    "tem_amount",
    "tem quantity",
]

def is_junk_page(txt):
    low = txt.lower()
    return any(p in low for p in JUNK_PATTERNS)


def normalize_name(s):
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(title="Bill OCR Extractor API")


class ExtractRequest(BaseModel):
    document: str   # URL or local path


@app.post("/extract-bill-data")
async def extract_bill_data(req: ExtractRequest):
    doc = req.document.strip()

    # Case 1: URL download
    if doc.startswith("http://") or doc.startswith("https://"):
        try:
            r = requests.get(doc)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF URL: {e}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content)
        tmp.close()
        pdf_path = tmp.name

    # Case 2: Local file
    elif os.path.exists(doc):
        pdf_path = doc

    # Invalid input
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid document path. Provide a URL or a valid local file path."
        )

    # Convert PDF
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_BIN)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")

    pagewise = []
    collected = []

    for i, page in enumerate(pages, 1):
        txt = pytesseract.image_to_string(page)

        if is_junk_page(txt):
            continue

        items = parse_page_text(txt)

        for it in items:
            it["_page_no"] = str(i)
            collected.append(it)

        pagewise.append({
            "page_no": str(i),
            "page_type": "Bill Detail",
            "bill_items": items
        })

    # Dedupe
    seen = set()
    final_items = []
    for it in collected:
        key = (normalize_name(it["item_name"]), it.get("item_amount"))
        if key not in seen:
            seen.add(key)
            final_items.append(it)

    total_amt = round(sum((it["item_amount"] or 0) for it in final_items), 2)

    return {
        "is_success": True,
        "data": {
            "pagewise_line_items": pagewise,
            "unique_line_items": final_items,
            "total_items_count": len(final_items),
            "sum_total": total_amt
        }
    }
