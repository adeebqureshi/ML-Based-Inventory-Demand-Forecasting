import pandas as pd
from typing import Dict, List, Optional, Tuple


def detect_columns(df: pd.DataFrame) -> Dict:
    """
    Automatically detect key columns from dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    Dict
        Dictionary with 'candidates' and 'suggestions' for date, quantity, and product columns
    """

    date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower()]
    qty_cols = [c for c in df.columns if "qty" in c.lower() or "sold" in c.lower() or "quantity" in c.lower()]
    product_cols = [c for c in df.columns if any(k in c.lower() for k in ["product", "item", "item_name", "product_id", "sku"])]

    date_col = date_cols[0] if date_cols else None
    qty_col = qty_cols[0] if qty_cols else None
    product_col = product_cols[0] if product_cols else None

    return {
        "candidates": {
            "date": date_cols,
            "quantity": qty_cols,
            "product": product_cols,
        },
        "suggestions": {
            "date": date_col,
            "quantity": qty_col,
            "product": product_col,
        }
    }


def process_data(df: pd.DataFrame, date_col: Optional[str] = None, qty_col: Optional[str] = None, 
                 product_col: Optional[str] = None, sequential: bool = False, 
                 start_date: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Process and validate dataset, normalizing column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    date_col : str, optional
        Name of date column (will be renamed to 'date').
    qty_col : str, optional
        Name of quantity column (will be renamed to 'quantity_sold').
    product_col : str, optional
        Name of product column (will be renamed to 'product').
    sequential : bool, optional
        Whether to generate sequential dates instead of using date_col.
    start_date : str, optional
        Start date for sequential date generation (format: YYYY-MM-DD).

    Returns
    -------
    Tuple[pd.DataFrame, Optional[str]]
        Processed DataFrame and error message (None if successful)
    """
    try:
        df = df.copy()

        # Build column mapping BEFORE normalizing (use columns as they are)
        col_mapping = {}
        
        # Map provided column names to standard names
        if date_col:
            col_mapping[date_col] = "date"
        if qty_col:
            col_mapping[qty_col] = "quantity_sold"
        if product_col and product_col != "None":
            col_mapping[product_col] = "product"
        
        # Apply the mapping
        df = df.rename(columns=col_mapping)

        # Normalize remaining column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # Generate sequential dates if requested
        if sequential:
            if start_date:
                df["date"] = pd.date_range(start=start_date, periods=len(df), freq="D")
            else:
                df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
        
        # Handle missing product column
        if "product" not in df.columns:
            df["product"] = "General"

        # Remove duplicates and NaNs
        df = df.drop_duplicates()
        df = df.dropna(subset=["date", "quantity_sold"])

        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

        # Ensure quantity is numeric
        if "quantity_sold" in df.columns:
            df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce")
            df = df.dropna(subset=["quantity_sold"])

        return df, None

    except Exception as e:
        return None, f"Error processing data: {str(e)}"