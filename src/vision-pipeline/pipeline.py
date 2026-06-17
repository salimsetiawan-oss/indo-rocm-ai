#!/usr/bin/env python3
"""
IndoROC: Indonesian Document Vision Pipeline
Multi-modal pipeline for extracting structured data from Indonesian documents.

Supported document types:
- KTP (Kartu Tanda Penduduk) - Indonesian ID card
- Faktur Pajak - Tax invoice
- Struk/Receipt - Purchase receipts

Usage:
    python pipeline.py --image path/to/ktp.jpg
    python pipeline.py --image path/to/invoice.jpg --type faktur
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image

# Conditional imports for vision components
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Document type schemas for structured extraction
DOCUMENT_SCHEMAS = {
    "ktp": {
        "fields": [
            "provinsi", "kabupaten", "nik", "nama", "tempat_lahir",
            "tanggal_lahir", "jenis_kelamin", "golongan_darah",
            "alamat", "rt_rw", "kelurahan", "kecamatan", "agama",
            "status_perkawinan", "pekerjaan", "kewarganegaraan",
            "berlaku_hingga"
        ],
        "description": "Kartu Tanda Penduduk (Indonesian National ID Card)"
    },
    "faktur": {
        "fields": [
            "nomor_faktur", "tanggal_faktur", "nama_penjual", "npwp_penjual",
            "nama_pembeli", "npwp_pembeli", "dpp", "ppn", "total",
            "items", "alamat_penjual", "alamat_pembeli"
        ],
        "description": "Faktur Pajak (Tax Invoice)"
    },
    "receipt": {
        "fields": [
            "nama_toko", "alamat_toko", "tanggal", "waktu",
            "items", "subtotal", "pajak", "total", "metode_pembayaran"
        ],
        "description": "Struk/Receipt (Purchase Receipt)"
    }
}


class VisionPipeline:
    """Multi-modal pipeline for Indonesian document understanding."""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.ocr_engine = None
        self.vision_model = None
        self.vision_processor = None
        self.llm = None
        self.llm_tokenizer = None

        logger.info(f"🔧 Initializing VisionPipeline on {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ Using GPU: {gpu_name}")
                return "cuda"
            else:
                logger.warning("⚠️  No GPU found, falling back to CPU")
                return "cpu"
        return device

    def load_ocr(self, lang: str = "en"):
        """Initialize OCR engine (PaddleOCR)."""
        if not HAS_PADDLE:
            logger.warning("PaddleOCR not available, OCR disabled")
            return

        logger.info("📝 Loading PaddleOCR...")
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            use_gpu=self.device == "cuda"
        )
        logger.info("✅ PaddleOCR loaded")

    def load_vision_model(self, model_name: str = "microsoft/Florence-2-large"):
        """Load vision-language model."""
        if not HAS_TRANSFORMERS:
            logger.error("Transformers not available")
            return

        logger.info(f"👁️  Loading vision model: {model_name}")
        self.vision_processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.vision_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        logger.info("✅ Vision model loaded")

    def ocr_extract(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract text from image using OCR."""
        if self.ocr_engine is None:
            self.load_ocr()

        logger.info(f"📝 Running OCR on: {image_path}")
        results = self.ocr_engine.ocr(image_path, cls=True)

        extracted = []
        if results and results[0]:
            for line in results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                extracted.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                })

        logger.info(f"   Extracted {len(extracted)} text regions")
        return extracted

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess document image."""
        import cv2
        import numpy as np

        logger.info(f"🔄 Preprocessing: {image_path}")

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deskew
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:  # Only deskew if angle is significant
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Contrast enhancement (CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Convert to PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        logger.info(f"   Preprocessed: {pil_img.size}")
        return pil_img

    def extract_ktp(self, image_path: str) -> Dict[str, Any]:
        """Extract structured data from KTP (ID card)."""
        logger.info("🇮🇩 Extracting KTP data...")

        # Step 1: OCR
        ocr_results = self.ocr_extract(image_path)
        raw_text = " ".join([r["text"] for r in ocr_results])

        # Step 2: Parse KTP fields
        ktp_data = {field: "" for field in DOCUMENT_SCHEMAS["ktp"]["fields"]}

        # Simple pattern matching for common KTP fields
        for result in ocr_results:
            text = result["text"].strip()
            text_upper = text.upper()

            # NIK (16 digits)
            if len(text) == 16 and text.isdigit():
                ktp_data["nik"] = text

            # Field labels
            if "PROVINSI" in text_upper:
                ktp_data["provinsi"] = text.replace("PROVINSI", "").strip().strip(":")
            elif "KABUPATEN" in text_upper or "KOTA" in text_upper:
                ktp_data["kabupaten"] = text.strip()
            elif "NIK" in text_upper:
                # Next line might be the actual NIK
                pass
            elif "NAMA" in text_upper:
                ktp_data["nama"] = text.replace("Nama", "").replace("NAMA", "").strip().strip(":")
            elif "LAHIR" in text_upper:
                ktp_data["tempat_lahir"] = text.strip()
            elif any(date_pattern in text for date_pattern in ["-", "/"]):
                if len(text) >= 8 and len(text) <= 12:
                    ktp_data["tanggal_lahir"] = text
            elif "ALAMAT" in text_upper:
                ktp_data["alamat"] = text.replace("Alamat", "").replace("ALAMAT", "").strip().strip(":")
            elif "RT" in text_upper and "RW" in text_upper:
                ktp_data["rt_rw"] = text.strip()
            elif "KEL" in text_upper or "DESA" in text_upper:
                ktp_data["kelurahan"] = text.strip()
            elif "KECAMATAN" in text_upper:
                ktp_data["kecamatan"] = text.strip()

        # Step 3: Use vision model for better extraction (if available)
        if self.vision_model is not None:
            ktp_data = self._vision_enhance(image_path, ktp_data, "ktp")

        ktp_data["_raw_ocr"] = raw_text
        ktp_data["_document_type"] = "ktp"

        logger.info(f"✅ KTP extraction complete: {ktp_data.get('nama', 'N/A')}")
        return ktp_data

    def extract_faktur(self, image_path: str) -> Dict[str, Any]:
        """Extract structured data from faktur pajak (tax invoice)."""
        logger.info("📄 Extracting faktur data...")

        ocr_results = self.ocr_extract(image_path)
        raw_text = " ".join([r["text"] for r in ocr_results])

        faktur_data = {field: "" for field in DOCUMENT_SCHEMAS["faktur"]["fields"]}
        faktur_data["items"] = []

        for result in ocr_results:
            text = result["text"].strip()
            text_upper = text.upper()

            if "FAKTUR" in text_upper and "PAJAK" in text_upper:
                continue  # Header
            elif "NPWP" in text_upper:
                # Extract NPWP number
                digits = "".join(filter(str.isdigit, text))
                if len(digits) >= 15:
                    if "PENJUAL" in text_upper or "PENERBIT" in text_upper:
                        faktur_data["npwp_penjual"] = digits
                    elif "PEMBELI" in text_upper:
                        faktur_data["npwp_pembeli"] = digits
            elif "DPP" in text_upper or "DASAR PENGENAAN" in text_upper:
                amount = self._extract_amount(text)
                if amount:
                    faktur_data["dpp"] = amount
            elif "PPN" in text_upper and "TOTAL" not in text_upper:
                amount = self._extract_amount(text)
                if amount:
                    faktur_data["ppn"] = amount
            elif "TOTAL" in text_upper:
                amount = self._extract_amount(text)
                if amount:
                    faktur_data["total"] = amount

        if self.vision_model is not None:
            faktur_data = self._vision_enhance(image_path, faktur_data, "faktur")

        faktur_data["_raw_ocr"] = raw_text
        faktur_data["_document_type"] = "faktur"

        logger.info(f"✅ Faktur extraction complete")
        return faktur_data

    def extract_receipt(self, image_path: str) -> Dict[str, Any]:
        """Extract structured data from receipt/struk."""
        logger.info("🧾 Extracting receipt data...")

        ocr_results = self.ocr_extract(image_path)
        raw_text = " ".join([r["text"] for r in ocr_results])

        receipt_data = {field: "" for field in DOCUMENT_SCHEMAS["receipt"]["fields"]}
        receipt_data["items"] = []

        for result in ocr_results:
            text = result["text"].strip()
            text_upper = text.upper()

            if "TOTAL" in text_upper:
                amount = self._extract_amount(text)
                if amount:
                    receipt_data["total"] = amount
            elif "PAJAK" in text_upper or "PB1" in text_upper or "PPN" in text_upper:
                amount = self._extract_amount(text)
                if amount:
                    receipt_data["pajak"] = amount

        receipt_data["_raw_ocr"] = raw_text
        receipt_data["_document_type"] = "receipt"

        logger.info(f"✅ Receipt extraction complete")
        return receipt_data

    def _extract_amount(self, text: str) -> Optional[str]:
        """Extract monetary amount from text."""
        import re
        # Match patterns like: Rp 1.000.000, 1000000, IDR 50.000
        patterns = [
            r'[Rr][Pp]\.?\s*[\d.,]+',
            r'IDR\s*[\d.,]+',
            r'[\d.,]+(?:,\d{2})?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group().strip()
        return None

    def _vision_enhance(
        self, image_path: str, base_data: Dict, doc_type: str
    ) -> Dict[str, Any]:
        """Use vision model to enhance extraction results."""
        if self.vision_model is None:
            return base_data

        logger.info("🔍 Using vision model for enhanced extraction...")

        image = Image.open(image_path)
        prompt = f"Extract all text from this Indonesian {doc_type} document. List each field and its value."

        inputs = self.vision_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.vision_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=3,
            )

        result = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"   Vision model result: {result[:200]}...")

        # Parse vision model output and merge with base_data
        # (simplified - real implementation would be more sophisticated)
        base_data["_vision_raw"] = result

        return base_data

    def process(self, image_path: str, doc_type: str = "auto") -> Dict[str, Any]:
        """Main processing pipeline."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Auto-detect document type if not specified
        if doc_type == "auto":
            doc_type = self._detect_type(image_path)

        logger.info(f"📋 Processing: {image_path} (type: {doc_type})")

        # Route to appropriate extractor
        if doc_type == "ktp":
            return self.extract_ktp(image_path)
        elif doc_type == "faktur":
            return self.extract_faktur(image_path)
        elif doc_type == "receipt":
            return self.extract_receipt(image_path)
        else:
            # Generic OCR extraction
            ocr_results = self.ocr_extract(image_path)
            return {
                "document_type": "unknown",
                "raw_text": " ".join([r["text"] for r in ocr_results]),
                "ocr_results": ocr_results,
            }

    def _detect_type(self, image_path: str) -> str:
        """Auto-detect document type using quick OCR scan."""
        ocr_results = self.ocr_extract(image_path)
        text = " ".join([r["text"].upper() for r in ocr_results])

        if "KTP" in text or "NIK" in text or "KARTU TANDA PENDUDUK" in text:
            return "ktp"
        elif "FAKTUR" in text or "PAJAK" in text or "NPWP" in text:
            return "faktur"
        elif "TOTAL" in text and ("SUBTOTAL" in text or "PAJAK" in text):
            return "receipt"
        else:
            return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="IndoROC: Indonesian Document Vision Pipeline"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to document image")
    parser.add_argument("--type", type=str, default="auto",
                        choices=["auto", "ktp", "faktur", "receipt"],
                        help="Document type")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--load-vision", action="store_true", help="Load vision model")
    parser.add_argument("--vision-model", type=str, default="microsoft/Florence-2-large",
                        help="Vision model name")
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VisionPipeline(device=args.device)

    # Optionally load vision model
    if args.load_vision:
        pipeline.load_vision_model(args.vision_model)

    # Process document
    result = pipeline.process(args.image, doc_type=args.type)

    # Output
    print("\n" + "=" * 60)
    print("📋 EXTRACTION RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)

    # Save to file if specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
