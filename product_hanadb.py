import re
import ast
import json
import logging
from hdbcli import dbapi
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai


# =========================
# Manual Config Variables
# =========================
# SAP HANA Connection Parameters
GEMINI_API_KEY = "AIzaSyAsaqUkDY7IuFc12P9a7jBmJER9i2ft3BE"
MODEL_NAME = "gemini-2.5-flash"

dbuser = "DSP_CUST_CONTENT#DSP_CUST_CONTENT"
dbpassword = "g6D,$a%@D`3$!)-GaVO#_[]T+=3z~[Z6"
dbhost = "c0c94ed5-bef0-4ca4-95f8-55cf5a4ecbdc.hana.prod-us10.hanacloud.ondemand.com"
dbport = 443
dbschema = "DSP_CUST_CONTENT"
viewname = "SALES_ORDER_CUST_SEGMENTATION"

# Column mappings - will be set after loading data
MATERIAL_COL = "Product"
MATERIAL_GROUP_COL = "Product Group"
ITEM_DESC_COL = "Product Description"
REVENUE_COL = "Revenue"
QUANTITY_COL = "Volume"
PRICE_COL = "ASP"
CUSTOMER_COL = "Customer"
DATE_COL = "Date"

# CORS origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://churn-ai-product.vercel.app"
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


def quarter_key_from_period_str(s: str) -> str:
    """Convert period string like '2024Q1' to '2024-Q1'"""
    return s.replace("Q", "-Q")


def find_column_name(df: pd.DataFrame, possible_names: list) -> str:
    """Find the actual column name from a list of possible names (case-insensitive, handles spaces)"""
    df_cols_normalized = {col.strip().lower(): col for col in df.columns}
    
    for name in possible_names:
        normalized = name.strip().lower()
        if normalized in df_cols_normalized:
            return df_cols_normalized[normalized]
    
    return None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping spaces and standardizing"""
    df.columns = df.columns.str.strip()
    return df


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
    
    # Aggregate by material with description
    agg_dict = {value_col: 'sum'}
    
    if ITEM_DESC_COL in df.columns:
        agg_dict[ITEM_DESC_COL] = 'first'
    
    pareto_df = (
        df.groupby(group_by_col)
        .agg(agg_dict)
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


def calculate_product_performance_metrics(df: pd.DataFrame, material_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive product performance metrics for a specific material
    """
    metrics = {}
    
    product_df = df[df[MATERIAL_COL].astype(str) == str(material_id)]
    
    if product_df.empty:
        return metrics
    
    # Basic metrics
    metrics['material_id'] = str(material_id)
    metrics['total_revenue'] = float(product_df[REVENUE_COL].sum())
    metrics['total_quantity'] = float(product_df[QUANTITY_COL].sum()) if QUANTITY_COL in product_df.columns else 0
    metrics['transaction_count'] = int(len(product_df))
    
    # Get product description
    if ITEM_DESC_COL in product_df.columns:
        desc_values = product_df[ITEM_DESC_COL].dropna().unique()
        metrics['product_description'] = desc_values[0] if len(desc_values) > 0 else ""
    
    # Get material group
    if MATERIAL_GROUP_COL in product_df.columns:
        group_values = product_df[MATERIAL_GROUP_COL].dropna().unique()
        metrics['material_group'] = group_values[0] if len(group_values) > 0 else ""
    
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
    
    # Temporal analysis - Yearly and Quarterly revenue
    revenue_by_year: Dict[str, float] = {}
    revenue_by_quarter: Dict[str, float] = {}
    
    # Use the Date column for temporal analysis
    if DATE_COL in product_df.columns:
        g_valid = product_df.copy()
        # Date column is in format YYYYMM, convert to datetime
        g_valid[DATE_COL] = pd.to_datetime(g_valid[DATE_COL].astype(str), format='%Y%m', errors='coerce')
        g_valid = g_valid[pd.notna(g_valid[DATE_COL])].copy()
        
        if not g_valid.empty and REVENUE_COL in g_valid.columns:
            # Revenue by year
            rev_series = g_valid.groupby(g_valid[DATE_COL].dt.year)[REVENUE_COL].sum()
            revenue_by_year = {str(int(k)): float(v) for k, v in rev_series.fillna(0).to_dict().items()}
            
            # Revenue by quarter
            q_series = g_valid.groupby(g_valid[DATE_COL].dt.to_period("Q"))[REVENUE_COL].sum()
            revenue_by_quarter = {quarter_key_from_period_str(str(k)): float(v) for k, v in q_series.fillna(0).to_dict().items()}
            
            # Monthly revenue
            product_df_copy = g_valid.copy()
            product_df_copy['month'] = product_df_copy[DATE_COL].dt.to_period('M')
            monthly = product_df_copy.groupby('month')[REVENUE_COL].sum().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            metrics['monthly_revenue'] = monthly.to_dict('records')
    
    metrics['revenue_by_year'] = revenue_by_year
    metrics['revenue_by_quarter'] = revenue_by_quarter
    
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
  "material_id": "<material ID>",
  "product_name": "<product name>",
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
  "revenue_by_year": {
    "YYYY": <number>,
    "...": <number>
  },
  "revenue_by_quarter": {
    "YYYY-QN": <number>,
    "...": <number>
  },
  "trend_analysis": "<max 40 words on revenue trends across years and quarters>",
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
- Analyze year-over-year and quarter-over-quarter trends

Competitive Context for Pricing (apply when relevant):
BEV SYRUP: Sysco (15% cheaper), US Foods (8% cheaper), Performance Food Group (12% cheaper)
SAUCES: Sysco (10% cheaper), US Foods (5% cheaper), Gordon Food Service (7% cheaper)
SYRUPS: US Foods (12% cheaper), Sysco (18% cheaper), Reinhart (9% cheaper)
TOPPINGS: Gordon Food Service (6% cheaper), US Foods (4% cheaper), Sysco (11% cheaper)
FILLINGS: Performance Food Group (13% cheaper), Sysco (8% cheaper), US Foods (10% cheaper)

Inputs:
material_id = [[MATERIAL_ID]]
pareto_data = [[PARETO_DATA]]
performance_metrics = [[PERFORMANCE_METRICS]]

Analysis Rules:
1. Use ABC classification as primary lens (A=top 80% revenue, B=next 15%, C=remaining 5%)
2. For A-class products: Focus on protection, margin optimization, and competitive defense
3. For B-class products: Identify promotion opportunities and bundling strategies
4. For C-class products: Evaluate rationalization or niche positioning
5. Apply Pareto principle to identify high-leverage opportunities
6. Pricing recommendations should balance competitiveness with margin sustainability
7. Analyze revenue_by_year and revenue_by_quarter data to identify growth/decline patterns
8. Identify seasonality and trend patterns in quarterly data
9. Observations: 3-5 bullet insights on performance patterns and portfolio health
10. Recommendations: 3-5 actionable steps for portfolio optimization

Output MUST be a single JSON object exactly per schema; no extra keys.
"""


def build_product_prompt(material_id: str, pareto_data: Dict, performance_metrics: Dict) -> str:
    prompt = PRODUCT_PROMPT_TEMPLATE
    prompt = prompt.replace("[[MATERIAL_ID]]", json.dumps(material_id, ensure_ascii=False))
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


def coerce_product_schema(obj: Dict[str, Any], material_id: str, abc_class: str) -> Dict[str, Any]:
    out = {
        "material_id": str(obj.get("material_id", material_id)),
        "product_name": str(obj.get("product_name", material_id)),
        "abc_classification": str(abc_class or obj.get("abc_classification", "")),
        "performance_summary": str(obj.get("performance_summary", "")),
        "pareto_insights": str(obj.get("pareto_insights", "")),
        "pricing_analysis": str(obj.get("pricing_analysis", "")),
        "growth_opportunities": str(obj.get("growth_opportunities", "")),
        "risk_factors": str(obj.get("risk_factors", "")),
        "portfolio_action": str(obj.get("portfolio_action", "")),
        "revenue_metrics": {},
        "volume_metrics": {},
        "revenue_by_year": {},
        "revenue_by_quarter": {},
        "trend_analysis": str(obj.get("trend_analysis", "")),
        "observation": [],
        "recommendation": [],
    }
    
    # Parse metrics
    for key in ['revenue_metrics', 'volume_metrics']:
        metrics = obj.get(key, {})
        if isinstance(metrics, dict):
            out[key] = {k: float(v) if v else 0.0 for k, v in metrics.items()}
    
    # Parse revenue dictionaries
    for dict_key in ['revenue_by_year', 'revenue_by_quarter']:
        rb = obj.get(dict_key, {})
        if isinstance(rb, dict):
            fixed = {}
            for k, v in rb.items():
                try:
                    fixed[str(k)] = float(v)
                except Exception:
                    fixed[str(k)] = 0.0
            out[dict_key] = fixed
    
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
    out["trend_analysis"] = trim_words(out["trend_analysis"], 40)
    
    return out


# =========================
# Load DF from SAP HANA
# =========================
def load_df_from_hana():
    """Connect to SAP HANA Cloud and load data from the view"""
    connection = dbapi.connect(
        address=dbhost,
        port=dbport,
        user=dbuser,
        password=dbpassword,
        encrypt=True,
        sslValidateCertificate=False
    )
    cursor = connection.cursor()
    sqlquery = f"SELECT * FROM {dbschema}.{viewname}"
    cursor.execute(sqlquery)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    connection.close()
    logger.info("Successfully loaded data from SAP HANA: %d rows, %d columns", len(df), len(df.columns))
    return df


# =========================
# Load and Process Data
# =========================
try:
    raw_df = load_df_from_hana()
    
    # Normalize column names (strip spaces)
    raw_df = normalize_column_names(raw_df)
    
    logger.info("Available columns after normalization: %s", raw_df.columns.tolist())
    
    # Map column names to handle variations
    MATERIAL_COL = find_column_name(raw_df, ["Product", "Material", "PRODUCT", "MATERIAL"]) or "Product"
    MATERIAL_GROUP_COL = find_column_name(raw_df, ["Product Group", "ProductGroup", "PRODUCT_GROUP"]) or "Product Group"
    ITEM_DESC_COL = find_column_name(raw_df, ["Product Description", "Item_Description", "PRODUCT_DESCRIPTION"]) or "Product Description"
    REVENUE_COL = find_column_name(raw_df, ["Revenue", "NetAmount", "REVENUE"]) or "Revenue"
    QUANTITY_COL = find_column_name(raw_df, ["Volume", "OrderQuantity", "VOLUME"]) or "Volume"
    PRICE_COL = find_column_name(raw_df, ["ASP", "NetPriceAmount", "PRICE"]) or "ASP"
    CUSTOMER_COL = find_column_name(raw_df, ["Customer", "SoldToParty", "CUSTOMER"]) or "Customer"
    DATE_COL = find_column_name(raw_df, ["Date", "BillingDocumentDate", "CreationDate", "DATE"]) or "Date"
    
    logger.info("Mapped columns - Material: %s, Revenue: %s, Date: %s", MATERIAL_COL, REVENUE_COL, DATE_COL)
    
    # Verify required columns
    required_cols = [MATERIAL_COL, REVENUE_COL]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        logger.error("Available columns: %s", raw_df.columns.tolist())
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean numeric columns
    if REVENUE_COL in raw_df.columns:
        raw_df[REVENUE_COL] = clean_numeric_column(raw_df[REVENUE_COL])
        logger.info("Cleaned %s column, sum=%.2f", REVENUE_COL, raw_df[REVENUE_COL].sum())
    
    if PRICE_COL in raw_df.columns:
        raw_df[PRICE_COL] = clean_numeric_column(raw_df[PRICE_COL])
    
    if QUANTITY_COL in raw_df.columns:
        raw_df[QUANTITY_COL] = clean_numeric_column(raw_df[QUANTITY_COL])
    
    # Date parsing - Date column is in YYYYMM format
    if DATE_COL in raw_df.columns:
        raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL].astype(str), format='%Y%m', errors='coerce')
    
    # Calculate Pareto analysis for materials only
    logger.info("Starting Pareto analysis for materials...")
    pareto_by_material = calculate_pareto_analysis(raw_df, MATERIAL_COL, REVENUE_COL)
    
    logger.info("Pareto analysis complete: %d materials", len(pareto_by_material))

except Exception as e:
    logger.exception("Error during data load: %s", e)
    raw_df = pd.DataFrame()
    pareto_by_material = pd.DataFrame()


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
    return {"status": "ok", "message": "Product Performance API with Pareto Analysis (Materials Only)"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "total_rows": int(len(raw_df)),
        "total_columns": int(len(raw_df.columns)) if not raw_df.empty else 0,
        "materials": int(len(pareto_by_material)),
        "columns_available": raw_df.columns.tolist() if not raw_df.empty else []
    }


@app.get("/pareto-data")
async def get_pareto_data():
    """
    Get Pareto analysis data for all materials (products only)
    Includes ABC classification, revenue percentages, and rankings with Material IDs
    """
    logger.info("GET /pareto-data")
    
    if pareto_by_material.empty:
        raise HTTPException(status_code=503, detail="Pareto analysis not available")
    
    # Select relevant columns for frontend
    output_cols = [MATERIAL_COL, REVENUE_COL, 'revenue_pct', 'cumulative_revenue_pct', 
                   'abc_class', 'rank', 'transaction_count', 'count_pct']
    
    # Add description if available
    if ITEM_DESC_COL in pareto_by_material.columns:
        output_cols.append(ITEM_DESC_COL)
    
    available_cols = [col for col in output_cols if col in pareto_by_material.columns]
    result = pareto_by_material[available_cols].copy()
    
    # Rename for clarity - keeping exact same names as original
    rename_dict = {
        MATERIAL_COL: 'material_id',
        REVENUE_COL: 'total_revenue'
    }
    if ITEM_DESC_COL in result.columns:
        rename_dict[ITEM_DESC_COL] = 'description'
    
    result = result.rename(columns=rename_dict)
    
    return result.to_dict(orient="records")


@app.get("/materials")
async def get_materials():
    """
    Get list of all materials with their descriptions and basic metrics
    """
    logger.info("GET /materials")
    
    if raw_df.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Group by material
    materials = raw_df.groupby(MATERIAL_COL).agg({
        REVENUE_COL: 'sum',
        QUANTITY_COL: 'sum' if QUANTITY_COL in raw_df.columns else 'count'
    }).reset_index()
    
    # Add description if available
    if ITEM_DESC_COL in raw_df.columns:
        desc_map = raw_df.groupby(MATERIAL_COL)[ITEM_DESC_COL].first().to_dict()
        materials['description'] = materials[MATERIAL_COL].map(desc_map)
    
    # Add material group if available
    if MATERIAL_GROUP_COL in raw_df.columns:
        group_map = raw_df.groupby(MATERIAL_COL)[MATERIAL_GROUP_COL].first().to_dict()
        materials['material_group'] = materials[MATERIAL_COL].map(group_map)
    
    # Rename columns - keeping exact same names as original
    materials = materials.rename(columns={
        MATERIAL_COL: 'material_id',
        REVENUE_COL: 'total_revenue',
        QUANTITY_COL: 'total_quantity'
    })
    
    # Sort by revenue
    materials = materials.sort_values('total_revenue', ascending=False)
    
    return materials.to_dict(orient="records")


@app.get("/product-insights/{material_id}")
async def get_product_insights(
    material_id: str,
    debug: bool = Query(False)
):
    """
    Get AI-powered insights for a specific material using Pareto analysis
    
    Parameters:
    - material_id: Material ID
    - debug: Include debug information in response
    
    Returns:
    - Comprehensive analysis with ABC classification, performance metrics, quarterly/yearly revenue, and recommendations
    """
    logger.info("GET /product-insights/%s debug=%s", material_id, debug)
    
    try:
        if raw_df.empty:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        # Filter data
        product_df = raw_df[raw_df[MATERIAL_COL].astype(str) == str(material_id)]
        if product_df.empty:
            raise HTTPException(status_code=404, detail=f"Material '{material_id}' not found")
        
        # Get Pareto data
        if pareto_by_material.empty:
            raise HTTPException(status_code=503, detail="Pareto analysis not available")
        
        pareto_row = pareto_by_material[pareto_by_material[MATERIAL_COL] == material_id].head(1)
        
        if pareto_row.empty:
            raise HTTPException(status_code=404, detail=f"Material not found in Pareto analysis")
        
        pareto_data_single = pareto_row.to_dict('records')[0]
        abc_class = pareto_data_single.get('abc_class', 'C')
        
        # Calculate performance metrics
        performance_metrics = calculate_product_performance_metrics(raw_df, material_id)
        
        # Build prompt
        prompt = build_product_prompt(
            material_id=material_id,
            pareto_data=pareto_data_single,
            performance_metrics=performance_metrics
        )
        
        logger.info("Prompt length=%d chars for %s", len(prompt), material_id)
        
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
            coerced = coerce_product_schema(parsed, material_id, abc_class)
            
            # Ensure revenue data from performance_metrics is included
            coerced['revenue_by_year'] = coerced.get('revenue_by_year') or performance_metrics.get('revenue_by_year', {})
            coerced['revenue_by_quarter'] = coerced.get('revenue_by_quarter') or performance_metrics.get('revenue_by_quarter', {})
            coerced['material_id'] = material_id
            
            # Add product description if available
            if 'product_description' in performance_metrics:
                coerced['product_description'] = performance_metrics['product_description']
            
            # Add material group if available
            if 'material_group' in performance_metrics:
                coerced['material_group'] = performance_metrics['material_group']
            
            if debug:
                return {
                    "result": coerced,
                    "debug": {
                        "product_rows": int(len(product_df)),
                        "pareto_data": pareto_data_single,
                        "performance_metrics": performance_metrics,
                        "raw_text_head": raw_text[:400],
                        "parsed": True,
                    }
                }
            return coerced
        
        # Fallback with revenue data
        fallback = {
            "material_id": material_id,
            "product_name": material_id,
            "abc_classification": abc_class,
            "performance_summary": "",
            "pareto_insights": "",
            "pricing_analysis": "",
            "growth_opportunities": "",
            "risk_factors": "",
            "portfolio_action": "",
            "revenue_metrics": pareto_data_single,
            "volume_metrics": performance_metrics,
            "revenue_by_year": performance_metrics.get('revenue_by_year', {}),
            "revenue_by_quarter": performance_metrics.get('revenue_by_quarter', {}),
            "trend_analysis": "",
            "observation": [],
            "recommendation": [],
        }
        
        # Add product description if available
        if 'product_description' in performance_metrics:
            fallback['product_description'] = performance_metrics['product_description']
        
        # Add material group if available
        if 'material_group' in performance_metrics:
            fallback['material_group'] = performance_metrics['material_group']
        
        if debug:
            return {
                "result": fallback,
                "debug": {
                    "product_rows": int(len(product_df)),
                    "performance_metrics": performance_metrics,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
