# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import tempfile
import requests
from pdf2image import convert_from_path
import pytesseract
from typing import Optional
from PIL import Image

# ==========================
# CONFIG (use env vars in deployment)
# ==========================
POPPLER_BIN = os.environ.get("POPPLER_BIN")  # e.g. "/usr/bin" or "/usr/local/bin"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")  # e.g. "/usr/bin/tesseract"

if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ==========================
# HELPERS
# ==========================
def download_file(url: str, suffix: str = "") -> str:
    """Downloads a file from a URL and returns the local file path."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


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

    # Determine whether doc is a URL or local path
    if doc.startswith("http://") or doc.startswith("https://"):
        # download
        try:
            # try to guess suffix
            lower = doc.lower()
            if lower.endswith(".pdf"):
                pdf_path = download_file(doc, suffix=".pdf")
                is_pdf = True
            elif any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]):
                pdf_path = download_file(doc, suffix=os.path.splitext(doc)[1])
                is_pdf = False
            else:
                # download as bytes and try to inspect simple magic
                tmp_path = download_file(doc)
                ext = os.path.splitext(doc)[1].lower()
                if ext == ".pdf":
                    pdf_path = tmp_path
                    is_pdf = True
                else:
                    # fallback assume pdf
                    pdf_path = tmp_path
                    is_pdf = True
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    elif os.path.exists(doc):
        # local file path
        ext = os.path.splitext(doc)[1].lower()
        is_pdf = ext == ".pdf"
        pdf_path = doc
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid document path. Provide a URL or a valid local file path."
        )

    pagewise = []
    collected = []

    # If the input is an image file: handle directly with PIL -> pytesseract
    try:
        if not is_pdf and os.path.exists(pdf_path):
            with Image.open(pdf_path) as im:
                txt = pytesseract.image_to_string(im)
                if not is_junk_page(txt):
                    items = parse_page_text(txt)
                    for it in items:
                        it["_page_no"] = "1"
                        collected.append(it)
                    pagewise.append({
                        "page_no": "1",
                        "page_type": "Bill Detail",
                        "bill_items": items
                    })
        else:
            # PDF handling: requires poppler (pdftoppm). Use POPPLER_BIN if provided.
            poppler_arg = POPPLER_BIN if POPPLER_BIN else None
            try:
                pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_arg) if poppler_arg else convert_from_path(pdf_path, dpi=300)
            except Exception as e:
                # helpful message for deployments without poppler
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "PDF conversion failed. Library `pdf2image` requires the Poppler `pdftoppm` binary.\n"
                        "In deployment, install poppler and set the POPPLER_BIN environment variable to its bin directory, "
                        "e.g. `/usr/bin` or `/usr/local/bin`. Error: " + str(e)
                    )
                )

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

    finally:
        # If we downloaded a temporary file (not user-supplied local path), try to remove it
        # Only remove if it's in the system temp directory
        try:
            tmpdir = tempfile.gettempdir()
            if pdf_path and os.path.exists(pdf_path) and pdf_path.startswith(tmpdir):
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
        except Exception:
            pass

    # Dedupe
    seen = set()
    final_items = []
    for it in collected:
        key = (normalize_name(it.get("item_name", "")), it.get("item_amount"))
        if key not in seen:
            seen.add(key)
            final_items.append(it)

    total_amt = round(sum((it.get("item_amount") or 0) for it in final_items), 2)

    return {
        "is_success": True,
        "data": {
            "pagewise_line_items": pagewise,
            "unique_line_items": final_items,
            "total_items_count": len(final_items),
            "sum_total": total_amt
        }
    }
