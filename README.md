# Clothes Segmentation

Automated product image processing using SegFormer B2 Clothes semantic segmentation model. This project extracts clothing items from product images with pixel-level precision, generating both white background and transparent background versions.

## Features

- **Semantic Segmentation**: Uses SegFormer B2 Clothes model for accurate clothing item detection
- **Category Mapping**: Automatically maps product categories from Excel files to SegFormer clothing classes
- **Dual Output Formats**: Generates extracted products with both white and transparent backgrounds
- **Batch Processing**: Process multiple product folders efficiently with a single model load
- **Comprehensive Logging**: Detailed logging with both console and file output using loguru
- **Error Handling**: Robust error handling with optional error skipping for batch operations
- **Statistics**: Provides detailed processing statistics and summaries

## Requirements

- Python >= 3.13
- CUDA-capable GPU (recommended) or CPU
- See `pyproject.toml` for full dependency list

## Installation

1. Clone or navigate to the project directory:
```bash
cd clothes-segmentation
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

## Project Structure

```
clothes-segmentation/
├── scripts/
│   ├── batch_process_products.py          # Batch processing script
│   └── process_product_images_segformer.py # Single product processing script
├── notebooks/
│   └── segformer_b2_clothes.ipynb         # SegFormer exploration notebook
├── pyproject.toml                         # Project dependencies
├── README.md                              # This file
└── logs/                                  # Log files (auto-created)
```

## Usage

### Single Product Processing

Process a single product folder:

```bash
python3 scripts/process_product_images_segformer.py
```

**Note**: Edit the script to change the `sku_folder_path` parameter (default: `"products/123276209"`).

### Batch Processing

Process all product folders in a directory:

```bash
python3 scripts/batch_process_products.py
```

The batch script will:
1. Load the SegFormer model once
2. Process all SKU folders in the `products` directory
3. Generate summary statistics
4. Save processing results to JSON

### Configuration

Both scripts support the following parameters:

- **products_base_dir**: Base directory containing product folders (default: `"products"`)
- **products_excel_path**: Path to products Excel file (default: `"products_asos.xlsx"`)
- **extract_output_base**: Base directory for saving extracted images (default: `"cropped_images"`)
- **skip_errors**: Whether to continue processing if an error occurs (default: `True`)

## Input Format

### Product Folders

Organize product images in folders named by SKU:
```
products/
├── 123276209/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 123276210/
│   └── ...
```

### Excel File Format

The Excel file (`products_asos.xlsx`) should contain:
- **sku**: Product SKU (must match folder name)
- **name**: Product name
- **category**: Product category (e.g., "t-shirt", "pants", "dress")

## Output Format

Extracted images are saved in the following structure:

```
cropped_images/
├── {sku}/
│   ├── transparent/
│   │   ├── image_0_segmented.png
│   │   ├── image_1_segmented.png
│   │   └── ...
│   └── white_background/
│       ├── image_0_segmented.png
│       ├── image_1_segmented.png
│       └── ...
```

## Supported Categories

The system maps common product categories to SegFormer clothing classes:

- **Upper Body**: t-shirt, shirt, top, blouse, sweater, hoodie, jacket, coat, etc. → `Upper-clothes`
- **Lower Body**: pants, trousers, jeans, leggings → `Pants`
- **Dresses**: dress, gown, frock → `Dress`
- **Skirts**: skirt, mini skirt, maxi skirt → `Skirt`
- **Accessories**: hat, scarf, belt, bag, sunglasses → Respective classes
- **Footwear**: shoes, sneakers, boots, sandals → `Left-shoe` / `Right-shoe`

## SegFormer Classes

The model supports 18 semantic classes:
- Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt
- Left-shoe, Right-shoe, Face, Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf

## Logging

Logs are automatically saved to:
- **Console**: Real-time INFO level logs with colored output
- **File**: `logs/process_product_images_segformer_{YYYY-MM-DD}.log` (DEBUG level)
- **Batch Summary**: `logs/batch_processing_summary_{timestamp}.json`

## Model Information

- **Model**: `mattmdjaga/segformer_b2_clothes`
- **Framework**: Hugging Face Transformers
- **Device**: Automatically uses CUDA if available, falls back to CPU

## Example Workflow

1. Prepare product images in SKU-named folders
2. Ensure Excel file contains SKU and category information
3. Run batch processing:
   ```bash
   python3 scripts/batch_process_products.py
   ```
4. Check output in `cropped_images/` directory
5. Review logs in `logs/` directory

## Notes

- The model is loaded once per batch run for efficiency
- Processing time varies based on image size and hardware
- Images are saved as PNG format to preserve transparency
- If a category cannot be mapped, it defaults to `Upper-clothes`

