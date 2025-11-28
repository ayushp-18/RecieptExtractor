# main.py
import os
import json
import re
import requests
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import google.generativeai as genaiii

# --- 0. Configuration and Initialization ---

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # optional helper configure
    genaiii.configure(api_key=GOOGLE_API_KEY)
except ValueError as e:
    print(f"Configuration Error: {e}")
    # allow server to start; errors will surface on requests

# Initialize the Gemini Client (google-genai package)
client = genai.Client(api_key=GOOGLE_API_KEY)

# Model and generation settings
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# --- 1. Pydantic Models for Request and Response (Schema Definition) ---

class BillItem(BaseModel):
    item_name: str = Field(description="Exactly as mentioned in the bill")
    item_amount: float = Field(description="Net Amount of the item post discounts as mentioned in the bill")
    item_rate: float = Field(description="Exactly as mentioned in the bill")
    item_quantity: float = Field(description="Exactly as mentioned in the bill")

class PagewiseLineItem(BaseModel):
    page_no: str
    page_type: str = Field(description="Bill Detail | Final Bill | Pharmacy")
    bill_items: List[BillItem]

class GeminiData(BaseModel):
    pagewise_line_items: List[PagewiseLineItem]
    total_item_count: int
    bill_sub_total: Optional[float] = Field(default=None, description="The Sub-total amount found on the bill, if it exists, otherwise None.")
    bill_final_total_amount: float = Field(description="The Final Total amount explicitly stated on the bill.")

class ExtractionRequest(BaseModel):
    document: str = Field(description="URL from where the file can be accessed and processed")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png?..."
                }
            ]
        }
    }

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class ExtractionData(BaseModel):
    pagewise_line_items: List[PagewiseLineItem]
    total_item_count: int

class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: ExtractionData

# --- 2. Optional: keep JSON schema for reference (not used with SDK) ---
def pydantic_to_json_schema(model: BaseModel) -> Dict[str, Any]:
    return model.model_json_schema()

GEMINI_OUTPUT_JSON_SCHEMA = pydantic_to_json_schema(GeminiData)

# --- 3. Gemini Prompt Definition ---

SYSTEM_PROMPT = """
You are an expert financial document processor. Your task to accurately extract ALL line item details
from the provided bill/invoice image and summarize the totals.

Adhere to the following strict rules:
1. Line Items: Extract all item names, net amounts, rates, and quantities for every line item.
2. No Double Counting: If a line item spans multiple pages, it must only be included once.
3. Totals: Extract the final Sub-total (if present) and the Final Total amount explicitly stated on the bill for validation.
4. The final output MUST strictly adhere to the provided JSON schema.
"""

USER_PROMPT = "Extract all line item details, the bill sub-total (if present), and the final total amount from this document. Provide the output strictly in the requested JSON format."

# --- 4. Core Logic and Helper Functions ---

def download_and_read_file(url: str) -> Optional[types.Part]:
    """Downloads file from a URL and prepares it as a Gemini Part object."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', 'application/octet-stream')
        if 'image/png' in content_type:
            mime_type = "image/png"
        elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
            mime_type = "image/jpeg"
        elif 'application/pdf' in content_type:
            mime_type = "application/pdf"
        else:
            print(f"Warning: Unknown Content-Type: {content_type}. Assuming image/jpeg.")
            mime_type = "image/jpeg"

        return types.Part.from_bytes(
            data=response.content,
            mime_type=mime_type
        )
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from URL: {e}")
        return None

def robust_json_loads(text: str) -> Any:
    """
    Attempt to parse JSON from `text`. If direct json.loads fails, try to extract the
    first JSON object/array substring using heuristics.
    """
    # Quick direct attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    s = text.strip() if text else ""
    # Try to find first JSON object {...}
    first = s.find('{')
    last = s.rfind('}')
    if first != -1 and last != -1 and last > first:
        try:
            candidate = s[first:last+1]
            return json.loads(candidate)
        except Exception:
            pass

    # Try to find first JSON array [...]
    first = s.find('[')
    last = s.rfind(']')
    if first != -1 and last != -1 and last > first:
        try:
            candidate = s[first:last+1]
            return json.loads(candidate)
        except Exception:
            pass

    # Try extracting JSON-like blocks and parsing them individually
    try:
        braces_blocks = re.findall(r'\{(?:[^{}]|\n|\r)*\}', text, flags=re.DOTALL)
        for block in braces_blocks:
            try:
                return json.loads(block)
            except Exception:
                continue
    except Exception:
        pass

    # Nothing worked
    raise json.JSONDecodeError("Could not parse JSON from text", text, 0)

# --- 5. FastAPI Endpoint Definition ---

app = FastAPI(
    title="HackRx Bill Extraction API",
    description="Intelligent Document Processing (IDP) solution for multi-page invoices using Google Gemini."
)

@app.post("/extract-bill-data", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_bill_data(request: ExtractionRequest):
    document_url = request.document

    # Step 1: Download the document
    document_part = download_and_read_file(document_url)
    if not document_part:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download or process document from URL: {document_url}. Check URL validity."
        )

    # Step 2: Prepare the content for Gemini (system text + file + user text)
    contents = [SYSTEM_PROMPT, document_part, USER_PROMPT]

    # Step 3: Configure and Call Gemini (do NOT pass response_schema for broad SDK compatibility)
    generation_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json"
        # response_schema omitted intentionally
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generation_config
        )
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API call failed: {e}"
        )

    # ---------------- Robust parsing and mapping ----------------
    # Safely extract raw text from the response object
    raw_text = None
    if hasattr(response, "text") and response.text is not None:
        raw_text = response.text
    else:
        # fallback: try to extract from candidates or possible attributes
        try:
            if hasattr(response, "candidates") and isinstance(response.candidates, (list, tuple)) and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if isinstance(candidate, dict):
                    raw_text = candidate.get("content") or candidate.get("message") or candidate.get("text")
                else:
                    raw_text = getattr(candidate, "content", None) or getattr(candidate, "message", None) or getattr(candidate, "text", None)
            else:
                raw_text = json.dumps(response.__dict__)
        except Exception:
            raw_text = None

    if not raw_text:
        print("Unable to extract text from Gemini response. Response repr:", repr(response))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model returned no textual output. Check model response object on server logs."
        )

    # Parse JSON robustly
    try:
        parsed_json = robust_json_loads(raw_text)
    except json.JSONDecodeError:
        print("Failed to JSON-decode model output. Raw output:", raw_text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decode model JSON output. See server logs for raw output."
        )

    # Now we have parsed_json (a dict, list, or a single item dict)
    gemini_payload_obj = None

    # helper to map alternative item->BillItem-compatible dict
    def _map_item_from_alt(it: Dict[str, Any]) -> Dict[str, Any]:
        # Accept various name fields
        name = it.get("item_name") or it.get("item") or it.get("name") or ""
        # helper to coerce number-like values to float safely
        def to_float_safe(x):
            if x is None or x == "":
                return 0.0
            try:
                return float(x)
            except Exception:
                try:
                    return float(str(x).replace(',', '').strip())
                except Exception:
                    return 0.0

        # support multiple amount keys
        amount = it.get("net_amount") if "net_amount" in it else (it.get("net") if "net" in it else it.get("amount"))
        return {
            "item_name": name,
            "item_amount": to_float_safe(amount),
            "item_rate": to_float_safe(it.get("rate")),
            "item_quantity": to_float_safe(it.get("quantity") or it.get("qty") or it.get("count"))
        }

    # Case A: The model returned the expected schema already
    if isinstance(parsed_json, dict) and "pagewise_line_items" in parsed_json:
        try:
            gemini_payload_obj = GeminiData.model_validate(parsed_json)
        except Exception as e:
            print("Validation against expected GeminiData failed. Raw parsed_json:", parsed_json, "Error:", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model output JSON did not validate against expected schema. See server logs."
            ) from e

    # Case B: The model returned consolidated schema like {"line_items": [...], "sub_total": ..., "final_total": ...}
    elif isinstance(parsed_json, dict) and "line_items" in parsed_json:
        try:
            items = parsed_json.get("line_items", [])
            mapped_items = [_map_item_from_alt(it) for it in items]

            page_obj = {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": mapped_items
            }

            gemini_dict = {
                "pagewise_line_items": [page_obj],
                "total_item_count": len(mapped_items),
                "bill_sub_total": float(parsed_json.get("sub_total")) if parsed_json.get("sub_total") not in (None, "") else None,
                "bill_final_total_amount": float(parsed_json.get("final_total")) if parsed_json.get("final_total") not in (None, "") else 0.0
            }

            gemini_payload_obj = GeminiData.model_validate(gemini_dict)
        except Exception as e:
            print("Error while mapping alternative model schema to GeminiData. parsed_json:", parsed_json, "Error:", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to transform model output into expected schema. See server logs."
            ) from e

    # Case C: parsed_json is a single item object (e.g., {'item': 'Consultation Charge ...', 'net_amount': 200.0, ...})
    elif isinstance(parsed_json, dict) and any(k in parsed_json for k in ("item", "item_name", "net_amount", "amount", "rate", "quantity")):
        try:
            # wrap single item into list
            mapped_item = _map_item_from_alt(parsed_json)
            page_obj = {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [mapped_item]
            }
            gemini_dict = {
                "pagewise_line_items": [page_obj],
                "total_item_count": 1,
                "bill_sub_total": None,
                "bill_final_total_amount": mapped_item.get("item_amount", 0.0)
            }
            gemini_payload_obj = GeminiData.model_validate(gemini_dict)
        except Exception as e:
            print("Error while mapping single-item model output to GeminiData. parsed_json:", parsed_json, "Error:", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to transform single-item model output into expected schema. See server logs."
            ) from e

    else:
        print("Unknown model JSON schema returned. Raw parsed_json:", parsed_json)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model returned an unexpected JSON structure. See server logs for raw output."
        )

    # gemini_payload_obj is now a validated GeminiData instance
    gemini_output_data = gemini_payload_obj

    # Tally the final total_item_count (defensive)
    final_item_count = 0
    for page in gemini_output_data.pagewise_line_items:
        final_item_count += len(page.bill_items)

    # Extract token usage if available; provide safe defaults if not
    usage = getattr(response, "usage_metadata", None)
    if usage:
        total_tokens = getattr(usage, "total_token_count", 0)
        input_tokens = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)
    else:
        total_tokens = input_tokens = output_tokens = 0

    final_response_data = ExtractionResponse(
        is_success=True,
        token_usage=TokenUsage(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        ),
        data=ExtractionData(
            pagewise_line_items=gemini_output_data.pagewise_line_items,
            total_item_count=final_item_count
        )
    )

    return final_response_data
