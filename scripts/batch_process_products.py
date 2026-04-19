"""
Batch Product Image Processing Script

This script processes multiple product folders in batch by:
1. Loading the SegFormer model once
2. Iterating through all SKU folders in the products directory
3. Processing each product folder using SegFormer segmentation
4. Generating summary statistics

Features:
- Processes all products in a directory automatically
- Loads model once for efficiency
- Comprehensive error handling and logging
- Summary statistics at the end

Usage:
    python batch_process_products.py
    
Requirements:
    Same as process_product_images_segformer.py
"""

from pathlib import Path
from loguru import logger
import sys
import json
from datetime import datetime
from process_product_images_segformer import load_model, process_product_folder

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/batch_process_products_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)


def batch_process_products(
    products_base_dir="products",
    products_excel_path="products_asos.xlsx",
    extract_output_base="cropped_images",
    skip_errors=True
):
    """
    Process all product folders in batch.
    
    Parameters:
    -----------
    products_base_dir : str
        Base directory containing product folders (default: "products")
    products_excel_path : str
        Path to products Excel file (default: "products_asos.xlsx")
    extract_output_base : str
        Base directory for saving extracted images (default: "cropped_images")
    skip_errors : bool
        Whether to continue processing if an error occurs (default: True)
    
    Returns:
    --------
    dict: Summary statistics including total processed, successful, failed, etc.
    """
    products_path = Path(products_base_dir)
    
    if not products_path.exists():
        logger.error(f"Products directory not found: {products_base_dir}")
        return {
            "error": f"Products directory not found: {products_base_dir}",
            "total_folders": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0
        }
    
    logger.info(f"Scanning products directory: {products_path}")
    
    # Get all subdirectories (SKU folders)
    sku_folders = [f for f in products_path.iterdir() if f.is_dir()]
    
    if len(sku_folders) == 0:
        logger.warning(f"No product folders found in {products_base_dir}")
        return {
            "error": "No product folders found",
            "total_folders": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0
        }
    
    logger.info(f"Found {len(sku_folders)} product folder(s) to process")
    
    # Load model once for all processing
    logger.info("="*60)
    logger.info("LOADING MODEL")
    logger.info("="*60)
    processor, model, device, class_labels = load_model()
    
    # Statistics
    total_folders = len(sku_folders)
    processed_count = 0
    successful_count = 0
    failed_count = 0
    results = []
    
    # Process each folder
    logger.info("="*60)
    logger.info("BATCH PROCESSING STARTED")
    logger.info("="*60)
    
    for idx, sku_folder in enumerate(sorted(sku_folders), 1):
        sku = sku_folder.name
        logger.info("")
        logger.info(f"[{idx}/{total_folders}] Processing SKU: {sku}")
        logger.info("-" * 60)
        
        try:
            result = process_product_folder(
                sku_folder_path=str(sku_folder),
                processor=processor,
                model=model,
                device=device,
                class_labels=class_labels,
                products_excel_path=products_excel_path,
                extract_images=True,
                extract_output_base=extract_output_base
            )
            
            processed_count += 1
            
            # Check if processing was successful
            if result.get("status") == "success" and "error" not in result:
                successful_count += 1
                logger.success(f"✓ Successfully processed SKU: {sku}")
            else:
                failed_count += 1
                error_msg = result.get("error", "Unknown error")
                logger.error(f"✗ Failed to process SKU: {sku} - {error_msg}")
            
            results.append({
                "sku": sku,
                "status": result.get("status"),
                "error": result.get("error"),
                "category": result.get("category"),
                "mapped_class": result.get("mapped_class"),
                "extraction_result": result.get("extraction_result")
            })
            
        except Exception as e:
            failed_count += 1
            processed_count += 1
            logger.error(f"✗ Exception processing SKU: {sku}")
            logger.exception(e)
            
            results.append({
                "sku": sku,
                "status": "error",
                "error": str(e),
                "category": None,
                "mapped_class": None,
                "extraction_result": None
            })
            
            if not skip_errors:
                logger.error("Stopping batch processing due to error (skip_errors=False)")
                break
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total folders: {total_folders}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {(successful_count/processed_count*100):.1f}%" if processed_count > 0 else "N/A")
    
    # Calculate total extractions
    total_extractions = 0
    total_images_processed = 0
    for result in results:
        extraction_result = result.get("extraction_result")
        if extraction_result and isinstance(extraction_result, dict):
            total_extractions += extraction_result.get("total_extractions", 0)
            total_images_processed += extraction_result.get("processed_images", 0)
    
    logger.info(f"Total images processed: {total_images_processed}")
    logger.info(f"Total extractions saved: {total_extractions}")
    
    summary = {
        "total_folders": total_folders,
        "processed": processed_count,
        "successful": successful_count,
        "failed": failed_count,
        "total_images_processed": total_images_processed,
        "total_extractions": total_extractions,
        "results": results
    }
    
    return summary


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("BATCH PRODUCT IMAGE PROCESSING")
    logger.info("="*60)
    
    # Process all products
    summary = batch_process_products(
        products_base_dir="products",
        products_excel_path="products_asos.xlsx",
        extract_output_base="cropped_images",
        skip_errors=True  # Continue processing even if one fails
    )
    
    logger.info("")
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Save summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path("logs") / f"batch_processing_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to: {summary_file}")

