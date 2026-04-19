"""
Automated Product Image Processing with SegFormer B2 Clothes Segmentation

This script processes product images by:
1. Reading product category from Excel file
2. Mapping category to SegFormer clothing classes
3. Performing semantic segmentation to extract products
4. Saving extracted products with white and transparent backgrounds

Features:
- Uses SegFormer B2 Clothes model for semantic segmentation
- Extracts products using pixel-level masks
- Saves both white background and transparent background versions
- Comprehensive logging with loguru (console + file)
- Reads category directly from Excel file

Usage:
    python process_product_images_segformer.py
    
Requirements:
    pip install pandas openpyxl torch transformers pillow loguru numpy
"""

import pandas as pd
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from loguru import logger
import sys
import time

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/process_product_images_segformer_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """
    Load the SegFormer B2 Clothes model, processor, and device.
    
    Returns:
        tuple: (processor, model, device, class_labels)
    """
    logger.info("Loading SegFormer B2 Clothes model...")
    model_id = "mattmdjaga/segformer_b2_clothes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.debug(f"Model ID: {model_id}, Device: {device}")
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    
    # Get class labels from model config if available
    if hasattr(model.config, 'id2label') and model.config.id2label:
        class_labels = {int(k): v for k, v in model.config.id2label.items()}
    else:
        # Default class labels for segformer_b2_clothes
        class_labels = {
            0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
            5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe",
            10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg",
            14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
        }
    
    logger.success(f"Model loaded successfully on device: {device}")
    logger.debug(f"Available classes: {list(class_labels.values())}")
    return processor, model, device, class_labels


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

def map_category_to_segformer_class(category, class_labels):
    """
    Map category from Excel file to SegFormer clothing class.
    
    SegFormer B2 Clothes supports 18 classes:
    0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
    5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
    11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
    16: Bag, 17: Scarf
    
    Parameters:
    -----------
    category : str
        Category from Excel file (e.g., "t-shirt", "pants", "dress")
    class_labels : dict
        Dictionary mapping class IDs to class names
    
    Returns:
    --------
    str: SegFormer class name (e.g., "Upper-clothes", "Pants", "Dress")
    """
    category_lower = category.lower().strip()
    
    # Create reverse mapping: class name -> class ID
    name_to_id = {v.lower(): k for k, v in class_labels.items()}
    
    # Mapping rules: common product names -> SegFormer classes
    # Note: For shoes, we map to "Left-shoe" (class 9) as default
    # The model will detect both left and right shoes if present
    category_mapping = {
        # Upper body clothing (class 4)
        "t-shirt": "upper-clothes",
        "tshirt": "upper-clothes",
        "shirt": "upper-clothes",
        "top": "upper-clothes",
        "blouse": "upper-clothes",
        "sweater": "upper-clothes",
        "hoodie": "upper-clothes",
        "jacket": "upper-clothes",
        "coat": "upper-clothes",
        "upper-clothes": "upper-clothes",
        "upper clothes": "upper-clothes",
        "upper": "upper-clothes",
        "shacket": "upper-clothes",
        "gilet": "upper-clothes",
        "vest": "upper-clothes",
        "sweater": "upper-clothes",
        "blazer": "upper-clothes",
        "parka": "upper-clothes",
        "shirt": "upper-clothes",
        "cardigan": "upper-clothes",
        "waistcoat": "upper-clothes",
        "raincoat": "upper-clothes",
        "jumper": "upper-clothes",
        "none": "upper-clothes",
        "hoodie": "upper-clothes",
        "blouse": "upper-clothes",
        "dress": "upper-clothes",
        
        # Lower body clothing (class 6)
        "pants": "pants",
        "trousers": "pants",
        "jeans": "pants",
        "tights": "pants",
        "leggings": "pants",
        "trouser": "pants",
        
        # Dresses (class 7)
        "dress": "dress",
        "gown": "dress",
        "frock": "dress",
        
        # Skirts (class 5)
        "skirt": "skirt",
        "mini skirt": "skirt",
        "maxi skirt": "skirt",
        "mini-skirt": "skirt",
        "maxi-skirt": "skirt",
        
        # Hats (class 1)
        "hat": "hat",
        "cap": "hat",
        "beanie": "hat",
        "baseball cap": "hat",
        
        # Scarves (class 17)
        "scarf": "scarf",
        "scarves": "scarf",
        
        # Belts (class 8)
        "belt": "belt",
        "waistband": "belt",
        
        # Bags (class 16)
        "bag": "bag",
        "handbag": "bag",
        "backpack": "bag",
        "purse": "bag",
        "tote": "bag",
        "tote bag": "bag",
        "shoulder bag": "bag",
        
        # Shoes (class 9 - Left-shoe, class 10 - Right-shoe)
        # Note: We use "Left-shoe" as default, model detects both if present
        "shoe": "left-shoe",
        "shoes": "left-shoe",
        "sneaker": "left-shoe",
        "sneakers": "left-shoe",
        "boot": "left-shoe",
        "boots": "left-shoe",
        "sandal": "left-shoe",
        "sandals": "left-shoe",
        "left-shoe": "left-shoe",
        "right-shoe": "right-shoe",
        
        # Sunglasses (class 3)
        "sunglasses": "sunglasses",
        "sunglass": "sunglasses",
        "glasses": "sunglasses",
        "eyewear": "sunglasses",
        
        # Note: Hair (class 2), Face (class 11), and body parts (classes 12-15)
        # are typically not product categories, but included for completeness
        "hair": "hair",
        "face": "face",
    }
    
    # Check direct mapping first
    if category_lower in category_mapping:
        mapped_class = category_mapping[category_lower]
        logger.debug(f"Mapped '{category}' -> '{mapped_class}'")
        return mapped_class
    
    # Check if category directly matches a SegFormer class name
    if category_lower in name_to_id:
        logger.debug(f"Category '{category}' directly matches SegFormer class")
        return category_lower
    
    # Try partial matching
    for key, value in category_mapping.items():
        if key in category_lower or category_lower in key:
            mapped_class = value
            logger.debug(f"Partial match: '{category}' -> '{mapped_class}'")
            return mapped_class
    
    # Default to Upper-clothes if no match found (most common case)
    logger.warning(f"No mapping found for '{category}', defaulting to 'upper-clothes'")
    return "upper-clothes"


# ============================================================================
# IMAGE SEGMENTATION AND EXTRACTION FUNCTION
# ============================================================================

def extract_product_segmentation(
    folder_path,
    target_class_name,
    processor,
    model,
    device,
    class_labels,
    output_base="cropped_images"
):
    """
    Extract products from images using semantic segmentation.
    Saves both white background and transparent background versions.
    Saves to: cropped_images/{sku}/transparent/ and cropped_images/{sku}/white_background/
    
    SegFormer B2 Clothes supports 18 classes:
    0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
    5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
    11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
    16: Bag, 17: Scarf
    
    Note: For shoes, you can specify "Left-shoe" or "Right-shoe" to extract
    a specific side, or the model will detect whichever is present in the image.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing product images (folder name should be the SKU)
    target_class_name : str
        SegFormer class name to extract (e.g., "Upper-clothes", "Pants", "Dress", "Left-shoe")
    processor : SegformerImageProcessor
        The SegFormer processor
    model : AutoModelForSemanticSegmentation
        The SegFormer model
    device : torch.device
        Device to run the model on
    class_labels : dict
        Dictionary mapping class IDs to class names
    output_base : str
        Base directory for saving extracted images (default: "cropped_images")
    
    Returns:
    --------
    dict: Statistics about extraction (total images, total extractions, etc.)
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get SKU from folder name
    folder = Path(folder_path)
    sku = folder.name
    
    # Create output directories: cropped_images/{sku}/transparent/ and cropped_images/{sku}/white_background/
    output_dir_base = Path(output_base) / str(sku)
    output_dir_transparent = output_dir_base / "transparent"
    output_dir_white = output_dir_base / "white_background"
    
    output_dir_transparent.mkdir(parents=True, exist_ok=True)
    output_dir_white.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting products for SKU: {sku}")
    logger.info(f"Target class: {target_class_name}")
    logger.info(f"Output directories: {output_dir_transparent} and {output_dir_white}")
    
    # Find target class ID
    target_class_id = None
    for class_id, class_name in class_labels.items():
        if class_name.lower() == target_class_name.lower():
            target_class_id = class_id
            break
    
    if target_class_id is None:
        logger.error(f"Target class '{target_class_name}' not found in model classes")
        return {
            "error": f"Target class '{target_class_name}' not found",
            "total_images": 0,
            "total_extractions": 0
        }
    
    logger.info(f"Target class ID: {target_class_id}")
    
    # Get all image files in the folder
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {folder_path}")
        return {"error": "No images found", "total_images": 0, "total_extractions": 0}
    
    logger.info(f"Found {len(image_files)} image(s) to process")
    
    total_extractions = 0
    processed_images = 0
    total_processing_time = 0.0
    
    # Sort image files for consistent numbering
    image_files = sorted(image_files)
    
    # Process each image
    for img_idx, img_path in enumerate(image_files):
        logger.info(f"Processing: {img_path.name}")
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(img_path)
            logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]}")
            
            # Process with model
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get logits and upsample to original image size
            logits = outputs.logits.cpu()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            # Get segmentation mask
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            
            # Create binary mask for target class
            target_mask = (pred_seg == target_class_id).numpy()
            
            # Check if target class is present
            if not target_mask.any():
                elapsed_time = time.time() - start_time
                total_processing_time += elapsed_time
                logger.warning(f"Target class '{target_class_name}' not found in {img_path.name} (took {elapsed_time:.2f} seconds)")
                # Log detected classes for debugging
                unique_classes = torch.unique(pred_seg).numpy()
                detected_classes = []
                for class_id in unique_classes:
                    class_id_int = int(class_id)
                    if class_id_int in class_labels:
                        pixel_count = (pred_seg == class_id_int).sum().item()
                        percentage = (pixel_count / pred_seg.numel()) * 100
                        detected_classes.append(f"{class_labels[class_id_int]} ({percentage:.1f}%)")
                logger.debug(f"Detected classes: {', '.join(detected_classes)}")
                continue
            
            # Calculate coverage percentage
            coverage = (target_mask.sum() / target_mask.size) * 100
            logger.info(f"Found {target_class_name} with {coverage:.2f}% coverage")
            
            # Convert original image to numpy array
            image_array = np.array(image)
            
            # Extract product with white background
            product_white = image_array.copy()
            product_white[~target_mask] = [255, 255, 255]  # White background
            
            # Extract product with transparent background
            # Create RGBA image
            product_rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            product_rgba[:, :, :3] = image_array
            product_rgba[:, :, 3] = 255  # Full opacity
            product_rgba[~target_mask, 3] = 0  # Transparent background
            
            # Save both versions
            # Format: image_{number}_segmented.png
            filename = f"image_{img_idx}_segmented.png"
            
            # Save white background version
            product_image_white = Image.fromarray(product_white)
            if product_image_white.mode != 'RGB':
                product_image_white = product_image_white.convert('RGB')
            # Save as PNG even for white background to maintain consistency
            white_path = output_dir_white / filename
            product_image_white.save(white_path, 'PNG')
            logger.success(f"Saved white background: {filename}")
            
            # Save transparent background version
            product_image_transparent = Image.fromarray(product_rgba, 'RGBA')
            transparent_path = output_dir_transparent / filename
            product_image_transparent.save(transparent_path, 'PNG')
            logger.success(f"Saved transparent background: {filename}")
            
            # Calculate and log processing time
            elapsed_time = time.time() - start_time
            total_processing_time += elapsed_time
            logger.info(f"Processing time for {img_path.name}: {elapsed_time:.2f} seconds")
            
            total_extractions += 1
            processed_images += 1
            
        except Exception as e:
            # Log time even if error occurred
            elapsed_time = time.time() - start_time
            total_processing_time += elapsed_time
            logger.error(f"Error processing {img_path.name}: {str(e)} (took {elapsed_time:.2f} seconds)")
            logger.exception(e)
            continue
    
    logger.success(f"Completed extraction!")
    logger.info(f"Processed images: {processed_images}/{len(image_files)}")
    logger.info(f"Total extractions saved: {total_extractions}")
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    if processed_images > 0:
        avg_time = total_processing_time / processed_images
        logger.info(f"Average time per image: {avg_time:.2f} seconds")
    logger.info(f"Output directories: {output_dir_transparent} and {output_dir_white}")
    
    return {
        "sku": sku,
        "total_images": len(image_files),
        "processed_images": processed_images,
        "total_extractions": total_extractions,
        "output_directory_transparent": str(output_dir_transparent),
        "output_directory_white": str(output_dir_white),
        "status": "success"
    }


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_product_folder(
    sku_folder_path,
    processor,
    model,
    device,
    class_labels,
    products_excel_path="products_asos.xlsx",
    extract_images=True,
    extract_output_base="cropped_images"
):
    """
    Process product images by reading category from Excel file, mapping it to SegFormer 
    clothing classes, and extracting products using semantic segmentation.
    
    Parameters:
    -----------
    sku_folder_path : str
        Path to folder containing product images (folder name should be the SKU)
    processor : SegformerImageProcessor
        The SegFormer processor
    model : AutoModelForSemanticSegmentation
        The SegFormer model
    device : torch.device
        Device to run the model on
    class_labels : dict
        Dictionary mapping class IDs to class names
    products_excel_path : str
        Path to products Excel file (default: "products_asos.xlsx")
    extract_images : bool
        Whether to extract images based on segmentation (default: True)
    extract_output_base : str
        Base directory for saving extracted images (default: "cropped_images")
    
    Returns:
    --------
    dict: Dictionary containing category, mapped class, extraction_result, and processing status
    """
    # Extract SKU from folder path
    folder_path = Path(sku_folder_path)
    sku = folder_path.name
    
    logger.info(f"Processing SKU: {sku}")
    logger.info(f"Folder path: {sku_folder_path}")
    
    # Load products Excel file
    try:
        df = pd.read_excel(products_excel_path)
        logger.info(f"Loaded products Excel file with {len(df)} rows")
    except FileNotFoundError:
        logger.error(f"Products Excel file not found: {products_excel_path}")
        return {
            "error": f"Products Excel file not found: {products_excel_path}",
            "category": None,
            "mapped_class": None
        }
    
    # Find product by SKU
    # Handle float SKU in Excel - convert folder name SKU to float for comparison
    try:
        # Try to convert folder name SKU to float (since Excel has float SKUs)
        sku_float = float(sku)
        # Compare with float SKU column directly
        sku_matches = df[df['sku'] == sku_float]
    except ValueError:
        # If conversion fails, try string comparison as fallback
        sku_matches = df[df['sku'].astype(str) == str(sku)]
    
    if len(sku_matches) == 0:
        logger.error(f"SKU {sku} not found in products Excel file")
        return {
            "error": f"SKU {sku} not found in products Excel file",
            "category": None,
            "mapped_class": None
        }
    
    # Get the first match
    product_row = sku_matches.iloc[0]
    
    product_name = product_row.get('name', 'N/A')
    logger.info(f"Found product: {product_name}")
    
    # Read category directly from Excel file
    if 'category' not in product_row:
        logger.error("Category column not found in Excel file")
        return {
            "error": "Category column not found in Excel file",
            "category": None,
            "mapped_class": None
        }
    
    category = str(product_row['category']).strip()
    
    if pd.isna(product_row['category']) or category == '' or category.lower() == 'nan':
        logger.error(f"Category is empty or missing for SKU {sku}")
        return {
            "error": f"Category is empty or missing for SKU {sku}",
            "category": None,
            "mapped_class": None
        }
    
    logger.success(f"Category from Excel: {category}")
    
    # Map category to SegFormer class
    mapped_class = map_category_to_segformer_class(category, class_labels)
    
    logger.info(f"Mapped category '{category}' to SegFormer class: {mapped_class}")
    
    extraction_result = None
    
    # Extract images if requested
    if extract_images:
        logger.info("="*60)
        logger.info("EXTRACTING PRODUCTS USING SEGMENTATION")
        logger.info("="*60)
        try:
            extraction_result = extract_product_segmentation(
                folder_path=sku_folder_path,
                target_class_name=mapped_class,
                processor=processor,
                model=model,
                device=device,
                class_labels=class_labels,
                output_base=extract_output_base
            )
        except Exception as e:
            logger.error(f"Error calling extract_product_segmentation: {str(e)}")
            logger.exception(e)
    
    return {
        "sku": sku,
        "category": category,
        "mapped_class": mapped_class,
        "extraction_result": extraction_result,
        "status": "success"
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("PRODUCT IMAGE PROCESSING WITH SEGFORMER")
    logger.info("="*60)
    
    # Load model once at startup
    processor, model, device, class_labels = load_model()
    
    # Process a product folder
    # Replace with your actual folder path
    result = process_product_folder(
        sku_folder_path="products/123276209",  # Change this to your product folder
        processor=processor,
        model=model,
        device=device,
        class_labels=class_labels,
        products_excel_path="products_asos.xlsx",
        extract_images=True,
        extract_output_base="cropped_images"
    )
    
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Result: {json.dumps(result, indent=2)}")

