import re
import ast
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# =========================
# Manual Config Variables
# =========================
CSV_URL = "https://raw.githubusercontent.com/vaibhav11062002/Churn-poc/main/llm_all_cust.csv"
GEMINI_API_KEY = "AIzaSyAsaqUkDY7IuFc12P9a7jBmJER9i2ft3BE"
MODEL_NAME = "gemini-2.5-flash"

MATERIAL_COL = "Material"
MATERIAL_GROUP_COL = "Material Group"
ITEM_DESC_COL = "Item Description"
REVENUE_COL = "Net Value"
QUANTITY_COL = "Order Quantity"
PRICE_COL = "Net Price"
CUSTOMER_COL = "Customer"
COMPANY_COL_CANDIDATES = [
    "Company Code", "company code", "company_code", "ccode to be billed", "c_code", "ccode", "CCode to Be Billed"
]

# CORS origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://chrun-ai.vercel.app"
]
ALLOW_CREDENTIALS = True

# =========================
# Logging config
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("product-performance")

# =========================
# LLM Auth
# =========================
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info("Configured Google Generative AI client")
except Exception as e:
    logger.exception("Failed to configure Generative AI: %s", e)
    raise

# =========================
# Helper Functions
# =========================
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lc:
            return cols_lc[name.lower()]
    return None

def sanitize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("|", "/")
    return s

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean numeric columns with commas and convert to float"""
    s = series.astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# =========================
# Pareto Analysis Functions
# =========================
def calculate_pareto_analysis(df: pd.DataFrame, group_by_col: str, value_col: str) -> pd.DataFrame:
    """
    Calculate Pareto analysis (80/20 rule) for products
    Returns dataframe with cumulative percentages and ABC classification
    """
    if group_by_col not in df.columns or value_col not in df.columns:
        logger.error(f"Missing columns. Available: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Aggregate by product
    pareto_df = (
        df.groupby(group_by_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(by=value_col, ascending=False)
        .reset_index(drop=True)
    )
    
    # Calculate percentages
    total_value = pareto_df[value_col].sum()
    if total_value == 0:
        pareto_df['revenue_pct'] = 0
        pareto_df['cumulative_revenue_pct'] = 0
    else:
        pareto_df['revenue_pct'] = (pareto_df[value_col] / total_value * 100).round(2)
        pareto_df['cumulative_revenue_pct'] = pareto_df['revenue_pct'].cumsum().round(2)
    
    # Count for frequency analysis
    count_df = df.groupby(group_by_col).size().reset_index(name='transaction_count')
    pareto_df = pareto_df.merge(count_df, on=group_by_col, how='left')
    
    total_count = pareto_df['transaction_count'].sum()
    if total_count == 0:
        pareto_df['count_pct'] = 0
        pareto_df['cumulative_count_pct'] = 0
    else:
        pareto_df['count_pct'] = (pareto_df['transaction_count'] / total_count * 100).round(2)
        pareto_df['cumulative_count_pct'] = pareto_df['count_pct'].cumsum().round(2)
    
    # ABC Classification based on cumulative revenue
    def classify_abc(cum_pct):
        if cum_pct <= 80:
            return 'A'
        elif cum_pct <= 95:
            return 'B'
        else:
            return 'C'
    
    pareto_df['abc_class'] = pareto_df['cumulative_revenue_pct'].apply(classify_abc)
    pareto_df['rank'] = range(1, len(pareto_df) + 1)
    
    return pareto_df

def calculate_product_performance_metrics(df: pd.DataFrame, product_id: str, analysis_type: str) -> Dict[str, Any]:
    """
    Calculate comprehensive product performance metrics for a specific product/group
    """
    metrics = {}
    
    analysis_col = ITEM_DESC_COL if analysis_type == "product" else MATERIAL_GROUP_COL
    product_df = df[df[analysis_col].astype(str) == str(product_id)]
    
    if product_df.empty:
        return metrics
    
    # Basic metrics
    metrics['total_revenue'] = float(product_df[REVENUE_COL].sum())
    metrics['total_quantity'] = float(product_df[QUANTITY_COL].sum()) if QUANTITY_COL in product_df.columns else 0
    metrics['transaction_count'] = int(len(product_df))
    
    # Price analysis
    if PRICE_COL in product_df.columns:
        metrics['avg_price'] = float(product_df[PRICE_COL].mean())
        metrics['price_std'] = float(product_df[PRICE_COL].std())
        metrics['min_price'] = float(product_df[PRICE_COL].min())
        metrics['max_price'] = float(product_df[PRICE_COL].max())
        metrics['median_price'] = float(product_df[PRICE_COL].median())
        metrics['price_volatility'] = float(product_df[PRICE_COL].std() / product_df[PRICE_COL].mean()) if product_df[PRICE_COL].mean() > 0 else 0.0
    
    # Customer metrics
    if CUSTOMER_COL in product_df.columns:
        metrics['unique_customers'] = int(product_df[CUSTOMER_COL].nunique())
        metrics['avg_revenue_per_customer'] = float(metrics['total_revenue'] / metrics['unique_customers']) if metrics['unique_customers'] > 0 else 0
    
    # Temporal analysis
    if 'Created On' in product_df.columns:
        product_df_copy = product_df.copy()
        product_df_copy['month'] = pd.to_datetime(product_df_copy['Created On'], errors='coerce').dt.to_period('M')
        monthly = product_df_copy.groupby('month')[REVENUE_COL].sum().reset_index()
        monthly['month'] = monthly['month'].astype(str)
        metrics['monthly_revenue'] = monthly.to_dict('records')
    
    # Material breakdown (for material_group analysis)
    if analysis_type == "material_group" and ITEM_DESC_COL in product_df.columns:
        top_products = (
            product_df.groupby(ITEM_DESC_COL)[REVENUE_COL]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_products.columns = ['product', 'revenue']
        metrics['top_products'] = top_products.to_dict('records')
    
    return metrics

# =========================
# GenAI Prompt for Product Performance
# =========================
PRODUCT_PROMPT_TEMPLATE = """
You are a product portfolio analyst providing strategic insights for sales and procurement teams.

Your role: Analyze product performance data using the Pareto principle (80/20 rule) and deliver actionable recommendations for portfolio optimization, pricing strategy, and inventory management.

Use a commercial lens focused on revenue growth, margin improvement, and portfolio rationalization.

JSON schema to return:
{
  "product_or_group": "<product/material group name>",
  "abc_classification": "A|B|C",
  "performance_summary": "<max 30 words on revenue, volume, and market position>",
  "pareto_insights": "<max 30 words on concentration and contribution to total revenue>",
  "pricing_analysis": "<max 30 words on price trends, volatility, and competitive positioning>",
  "growth_opportunities": "<max 30 words on upsell, cross-sell, or volume expansion potential>",
  "risk_factors": "<max 30 words on dependency risk, declining trends, or margin pressure>",
  "portfolio_action": "<max 20 words on push/hold/rationalize recommendation>",
  "revenue_metrics": {
    "total_revenue": <number>,
    "revenue_share_pct": <number>,
    "cumulative_revenue_pct": <number>,
    "rank": <number>
  },
  "volume_metrics": {
    "total_transactions": <number>,
    "transaction_share_pct": <number>,
    "avg_order_value": <number>
  },
  "observation": [
    {"key": "<insight title>", "value": "<business insight>"}
  ],
  "recommendation": [
    {"key": "<action title>", "value": "<specific action>"}
  ]
}

Tone and Style:
- Use clear, commercially actionable language for sales and procurement leadership
- Focus on portfolio profitability, inventory efficiency, and market competitiveness
- Frame insights around ABC classification and Pareto principle
- Emphasize high-impact actions that drive 80% of results

Competitive Context for Pricing (apply when relevant):
BEV SYRUP: Sysco (15% cheaper), US Foods (8% cheaper), Performance Food Group (12% cheaper)
SAUCES: Sysco (10% cheaper), US Foods (5% cheaper), Gordon Food Service (7% cheaper)
SYRUPS: US Foods (12% cheaper), Sysco (18% cheaper), Reinhart (9% cheaper)
TOPPINGS: Gordon Food Service (6% cheaper), US Foods (4% cheaper), Sysco (11% cheaper)
FILLINGS: Performance Food Group (13% cheaper), Sysco (8% cheaper), US Foods (10% cheaper)

Inputs:
product_id_or_group = [[PRODUCT_ID]]
analysis_type = [[ANALYSIS_TYPE]]
pareto_data = [[PARETO_DATA]]
performance_metrics = [[PERFORMANCE_METRICS]]

Analysis Rules:
1. Use ABC classification as primary lens (A=top 80% revenue, B=next 15%, C=remaining 5%)
2. For A-class products: Focus on protection, margin optimization, and competitive defense
3. For B-class products: Identify promotion opportunities and bundling strategies
4. For C-class products: Evaluate rationalization or niche positioning
5. Apply Pareto principle to identify high-leverage opportunities
6. Pricing recommendations should balance competitiveness with margin sustainability
7. Observations: 3-5 bullet insights on performance patterns and portfolio health
8. Recommendations: 3-5 actionable steps for portfolio optimization

Output MUST be a single JSON object exactly per schema; no extra keys.
"""

def build_product_prompt(product_id: str, analysis_type: str, pareto_data: Dict, performance_metrics: Dict) -> str:
    prompt = PRODUCT_PROMPT_TEMPLATE
    prompt = prompt.replace("[[PRODUCT_ID]]", json.dumps(product_id, ensure_ascii=False))
    prompt = prompt.replace("[[ANALYSIS_TYPE]]", json.dumps(analysis_type, ensure_ascii=False))
    prompt = prompt.replace("[[PARETO_DATA]]", json.dumps(pareto_data, separators=(",", ":"), ensure_ascii=False))
    prompt = prompt.replace("[[PERFORMANCE_METRICS]]", json.dumps(performance_metrics, separators=(",", ":"), ensure_ascii=False))
    return prompt

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return None

def parse_nested_json_list_field(field_value):
    if isinstance(field_value, str):
        try:
            val = ast.literal_eval(field_value)
            if isinstance(val, list):
                return val
        except Exception:
            pass
    return field_value

def coerce_product_schema(obj: Dict[str, Any], product_id: str, abc_class: str) -> Dict[str, Any]:
    out = {
        "product_or_group": str(obj.get("product_or_group", product_id)),
        "abc_classification": str(abc_class or obj.get("abc_classification", "")),
        "performance_summary": str(obj.get("performance_summary", "")),
        "pareto_insights": str(obj.get("pareto_insights", "")),
        "pricing_analysis": str(obj.get("pricing_analysis", "")),
        "growth_opportunities": str(obj.get("growth_opportunities", "")),
        "risk_factors": str(obj.get("risk_factors", "")),
        "portfolio_action": str(obj.get("portfolio_action", "")),
        "revenue_metrics": {},
        "volume_metrics": {},
        "observation": [],
        "recommendation": [],
    }
    
    # Parse metrics
    for key in ['revenue_metrics', 'volume_metrics']:
        metrics = obj.get(key, {})
        if isinstance(metrics, dict):
            out[key] = {k: float(v) if v else 0.0 for k, v in metrics.items()}
    
    # Parse lists
    out["observation"] = parse_nested_json_list_field(obj.get("observation", []))
    out["recommendation"] = parse_nested_json_list_field(obj.get("recommendation", []))
    
    # Trim text fields
    def trim_words(s: str, n: int) -> str:
        toks = str(s).split()
        return " ".join(toks[:n])
    
    out["performance_summary"] = trim_words(out["performance_summary"], 30)
    out["pareto_insights"] = trim_words(out["pareto_insights"], 30)
    out["pricing_analysis"] = trim_words(out["pricing_analysis"], 30)
    out["growth_opportunities"] = trim_words(out["growth_opportunities"], 30)
    out["risk_factors"] = trim_words(out["risk_factors"], 30)
    out["portfolio_action"] = trim_words(out["portfolio_action"], 20)
    
    return out

# =========================
# Load and Process Data
# =========================
def load_df_from_url(csv_url: str) -> pd.DataFrame:
    try:
        logger.info("Downloading CSV from URL %s", csv_url)
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        data_string = response.text
        
        # Try multiple parsing strategies
        df = None
        
        # Strategy 1: Tab-separated
        try:
            df = pd.read_csv(io.StringIO(data_string), sep='\t', dtype=str, on_bad_lines='skip', low_memory=False)
            if len(df.columns) > 5:
                logger.info("Successfully parsed with tab separator")
            else:
                df = None
        except Exception as e:
            logger.warning("Tab separator failed: %s", e)
        
        # Strategy 2: Auto-detect separator
        if df is None or len(df.columns) <= 5:
            try:
                df = pd.read_csv(io.StringIO(data_string), dtype=str, on_bad_lines='skip', low_memory=False, sep=None, engine='python')
                logger.info("Successfully parsed with auto-detect separator")
            except Exception as e:
                logger.warning("Auto-detect failed: %s", e)
        
        # Strategy 3: Comma-separated
        if df is None or len(df.columns) <= 5:
            try:
                df = pd.read_csv(io.StringIO(data_string), dtype=str, on_bad_lines='skip', low_memory=False)
                logger.info("Successfully parsed with comma separator")
            except Exception as e:
                logger.error("All parsing strategies failed: %s", e)
                raise
        
        logger.info("CSV loaded with shape=%s, columns=%d", df.shape, len(df.columns))
        return df
        
    except Exception as e:
        logger.exception("Failed to download CSV: %s", e)
        raise

# Load data
try:
    raw_df = load_df_from_url(CSV_URL)
    
    # Verify required columns
    required_cols = [MATERIAL_GROUP_COL, ITEM_DESC_COL, REVENUE_COL]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean numeric columns
    if REVENUE_COL in raw_df.columns:
        raw_df[REVENUE_COL] = clean_numeric_column(raw_df[REVENUE_COL])
        logger.info("Cleaned %s column, sum=%.2f", REVENUE_COL, raw_df[REVENUE_COL].sum())
    
    if PRICE_COL in raw_df.columns:
        raw_df[PRICE_COL] = clean_numeric_column(raw_df[PRICE_COL])
    
    if QUANTITY_COL in raw_df.columns:
        raw_df[QUANTITY_COL] = clean_numeric_column(raw_df[QUANTITY_COL])
    
    # Date parsing
    for col in ["Created On", "Billing Date"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors='coerce')
    
    # Calculate Pareto analysis
    logger.info("Starting Pareto analysis...")
    pareto_by_material_group = calculate_pareto_analysis(raw_df, MATERIAL_GROUP_COL, REVENUE_COL)
    pareto_by_product = calculate_pareto_analysis(raw_df, ITEM_DESC_COL, REVENUE_COL)
    
    # Create consolidated Pareto data with both products and material groups
    pareto_data = pd.DataFrame()
    
    if not pareto_by_material_group.empty:
        mg_data = pareto_by_material_group.copy()
        mg_data['type'] = 'material_group'
        mg_data['identifier'] = mg_data[MATERIAL_GROUP_COL]
        pareto_data = pd.concat([pareto_data, mg_data], ignore_index=True)
    
    if not pareto_by_product.empty:
        prod_data = pareto_by_product.copy()
        prod_data['type'] = 'product'
        prod_data['identifier'] = prod_data[ITEM_DESC_COL]
        pareto_data = pd.concat([pareto_data, prod_data], ignore_index=True)
    
    logger.info("Pareto analysis complete: %d material groups, %d products", 
                len(pareto_by_material_group), len(pareto_by_product))

except Exception as e:
    logger.exception("Error during data load: %s", e)
    raw_df = pd.DataFrame()
    pareto_by_material_group = pd.DataFrame()
    pareto_by_product = pd.DataFrame()
    pareto_data = pd.DataFrame()

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Product Performance and Pareto Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Product Performance API with Pareto Analysis"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "total_rows": int(len(raw_df)),
        "total_columns": int(len(raw_df.columns)) if not raw_df.empty else 0,
        "material_groups": int(len(pareto_by_material_group)),
        "products": int(len(pareto_by_product)),
        "columns_available": raw_df.columns.tolist() if not raw_df.empty else []
    }

@app.get("/pareto-data")
async def get_pareto_data():
    """
    Get consolidated Pareto analysis data for all products and material groups
    Includes ABC classification, revenue percentages, and rankings
    """
    logger.info("GET /pareto-data")
    
    if pareto_data.empty:
        raise HTTPException(status_code=503, detail="Pareto analysis not available")
    
    # Select relevant columns for frontend
    output_cols = ['identifier', 'type', REVENUE_COL, 'revenue_pct', 'cumulative_revenue_pct', 
                   'abc_class', 'rank', 'transaction_count', 'count_pct']
    
    available_cols = [col for col in output_cols if col in pareto_data.columns]
    result = pareto_data[available_cols].copy()
    
    # Rename for clarity
    result = result.rename(columns={
        'identifier': 'name',
        REVENUE_COL: 'total_revenue'
    })
    
    return result.to_dict(orient="records")

@app.get("/product-insights/{product_identifier}")
async def get_product_insights(
    product_identifier: str,
    analysis_type: str = Query("product", regex="^(product|material_group)$"),
    debug: bool = Query(False)
):
    """
    Get AI-powered insights for a specific product or material group using Pareto analysis
    
    Parameters:
    - product_identifier: Name of the product or material group
    - analysis_type: 'product' for individual products, 'material_group' for categories
    - debug: Include debug information in response
    
    Returns:
    - Comprehensive analysis with ABC classification, performance metrics, and recommendations
    """
    logger.info("GET /product-insights/%s type=%s debug=%s", product_identifier, analysis_type, debug)
    
    try:
        if raw_df.empty:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        # Determine analysis column
        analysis_col = ITEM_DESC_COL if analysis_type == "product" else MATERIAL_GROUP_COL
        
        # Filter data
        product_df = raw_df[raw_df[analysis_col].astype(str) == str(product_identifier)]
        if product_df.empty:
            raise HTTPException(status_code=404, detail=f"{analysis_type} '{product_identifier}' not found")
        
        # Get Pareto data
        pareto_df = pareto_by_product if analysis_type == "product" else pareto_by_material_group
        
        if pareto_df.empty:
            raise HTTPException(status_code=503, detail="Pareto analysis not available")
        
        pareto_row = pareto_df[pareto_df[analysis_col] == product_identifier].head(1)
        
        if pareto_row.empty:
            raise HTTPException(status_code=404, detail=f"{analysis_type} not found in Pareto analysis")
        
        pareto_data = pareto_row.to_dict('records')[0]
        abc_class = pareto_data.get('abc_class', 'C')
        
        # Calculate performance metrics
        performance_metrics = calculate_product_performance_metrics(raw_df, product_identifier, analysis_type)
        
        # Build prompt
        prompt = build_product_prompt(
            product_id=product_identifier,
            analysis_type=analysis_type,
            pareto_data=pareto_data,
            performance_metrics=performance_metrics
        )
        
        logger.info("Prompt length=%d chars for %s", len(prompt), product_identifier)
        
        # Call LLM
        raw_text = ""
        parsed = None
        try:
            resp = model.generate_content(prompt)
            raw_text = (resp.text or "")
            logger.info("LLM response length=%d", len(raw_text))
            parsed = try_parse_json(raw_text)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
        
        # Return results
        if isinstance(parsed, dict):
            coerced = coerce_product_schema(parsed, product_identifier, abc_class)
            if debug:
                return {
                    "result": coerced,
                    "debug": {
                        "product_rows": int(len(product_df)),
                        "pareto_data": pareto_data,
                        "performance_metrics": performance_metrics,
                        "raw_text_head": raw_text[:400],
                        "parsed": True,
                    }
                }
            return coerced
        
        # Fallback
        fallback = {
            "product_or_group": product_identifier,
            "abc_classification": abc_class,
            "performance_summary": "",
            "pareto_insights": "",
            "pricing_analysis": "",
            "growth_opportunities": "",
            "risk_factors": "",
            "portfolio_action": "",
            "revenue_metrics": pareto_data,
            "volume_metrics": performance_metrics,
            "observation": [],
            "recommendation": [],
        }
        
        if debug:
            return {
                "result": fallback,
                "debug": {
                    "product_rows": int(len(product_df)),
                    "raw_text_head": raw_text[:400],
                    "parsed": False,
                }
            }
        return fallback
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_product_insights: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
