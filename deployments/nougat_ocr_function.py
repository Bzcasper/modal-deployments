"""Nougat OCR Modal Function Deployment

This module provides a Modal function deployment for running Nougat OCR on PDF files.
It uses the Nougat CLI interface for lightweight processing without API dependencies.

Deployment:
    To deploy this function to Modal:

    ```bash
    modal deploy deployments/nougat_ocr_function.py
    ```

Usage:
    Import and use the deployed function:

    ```bash
    mkdir -p data && curl -L "https://arxiv.org/pdf/2308.13418.pdf" -o "data/nougat.pdf"
    ```

    ```python
    import modal

    # Look up the deployed function
    f = modal.Function.lookup("nougat-ocr-cli", "process_pdf")

    # Process an entire PDF
    with open("data/nougat.pdf", "rb") as file:
        pdf_bytes = file.read()
    text = f.remote(pdf_bytes)

    # Process specific pages
    text = f.remote(pdf_bytes, start_page=1, end_page=5)
    ```

    The function returns extracted text in markdown format.
"""

from pathlib import Path
import subprocess
import tempfile
from typing import Optional
from modal import Image, App, gpu
import modal

# Create base image with uv installed
base_image = (
    Image.debian_slim()
    .run_commands(
        "apt-get update && apt-get install -y curl",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Install nougat-ocr without API dependencies
        "/root/.local/bin/uv tool install nougat-ocr --python 3.12 --with transformers==4.38.2",
    )
    .pip_install("loguru")
)

app = App("nougat-ocr-cli")


@app.function(
    image=base_image,
    gpu=gpu.A10G(count=1),
    timeout=1800,
)
def process_pdf(
    pdf_bytes: bytes,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """Process a PDF document and extract text using Nougat OCR CLI.

    Uses the Nougat base model (0.1.0-base) with GPU acceleration and markdown
    compatibility. The --no-skipping flag is enabled to prevent false positives
    in failure detection when running on GPU.

    :param pdf_bytes: Raw bytes of the PDF file
    :param start_page: Optional starting page number (1-based indexing)
    :param end_page: Optional ending page number (1-based indexing)
    :return: Extracted text in markdown format
    """
    from loguru import logger

    logger.info("Processing PDF with Nougat OCR...")

    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        pdf_path = temp_dir / "input.pdf"
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Write PDF bytes to temporary file
        pdf_path.write_bytes(pdf_bytes)

        # Construct command
        cmd = [
            "/root/.local/bin/nougat",
            str(pdf_path),
            "--out",
            str(output_dir),
            "--model",
            "0.1.0-base",  # Using the base model for better quality
            "--no-skipping",  # Prevent false positives in failure detection on GPU
            "--markdown",  # Ensure markdown compatibility
        ]

        if start_page is not None and end_page is not None:
            cmd.extend(["--pages", f"{start_page}-{end_page}"])
        elif start_page is not None:
            cmd.extend(["--pages", f"{start_page}-"])
        elif end_page is not None:
            cmd.extend(["--pages", f"1-{end_page}"])

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            # Run nougat CLI command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Read and combine output files
            output_files = sorted(output_dir.glob("*.mmd"))
            if not output_files:
                raise RuntimeError("No output files generated")

            combined_text = ""
            for file in output_files:
                combined_text += file.read_text() + "\n\n"

            return combined_text.strip()

        except subprocess.CalledProcessError as e:
            logger.error(f"Nougat CLI error: {e.stderr}")
            raise RuntimeError(f"Nougat CLI failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
