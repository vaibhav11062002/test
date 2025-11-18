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
SERPER_API_KEY = "2bd97b3b478ccea78c1f5076af32d493fbef8bec"
MODEL_NAME = "models/gemini-2.5-flash"  # Updated with models/ prefix

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
# LLM Auth - Updated with generation config
# =========================
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    )
    logger.info("Configured Google Generative AI client with model: %s", MODEL_NAME)
except Exception as e:
    logger.exception("Failed to configure Generative AI: %s", e)
    raise

# =========================
# Serper.dev Search Function
# =========================
def search_competitive_pricing(product_name: str, category: str, avg_price: float = 0) -> Dict[str, Any]:
    """Search for competitive pricing using Serper.dev with improved query building"""
    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY_HERE":
        logger.warning("Serper API key not configured")
        return {"enabled": False, "message": "Live pricing search disabled - no API key"}
    
    try:
        url = "https://google.serper.dev/search"
        
        queries_to_try = [
            f"{product_name} Sysco US Foods price",
            f"{product_name.split()[0] if product_name else 'beverage'} wholesale distributor pricing",
        ]
        
        all_results = []
        
        for query_idx, query in enumerate(queries_to_try):
            payload = json.dumps({"q": query, "num": 10, "gl": "us", "hl": "en"})
            headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
            
            logger.info(f"Search attempt {query_idx + 1}: {query}")
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            if 'organic' in results and len(results['organic']) > 0:
                logger.info(f"Found {len(results['organic'])} results")
                all_results = results['organic']
                break
        
        pricing_data = {
            "enabled": True,
            "search_query": query,
            "your_avg_price": round(avg_price, 2),
            "organic_results": [],
            "pricing_mentions": [],
            "competitor_info": []
        }
        
        if all_results:
            for idx, result in enumerate(all_results[:5]):  # Top 5 only
                snippet = result.get('snippet', '')
                title = result.get('title', '')
                
                price_pattern = r'\$\d+(?:\.\d{2})?'
                prices_found = re.findall(price_pattern, snippet + ' ' + title)
                
                pricing_data['organic_results'].append({
                    "position": idx + 1,
                    "title": title[:80],
                    "snippet": snippet[:120],
                    "prices_mentioned": prices_found[:2]
                })
                
                if prices_found:
                    for price in prices_found[:2]:
                        pricing_data['pricing_mentions'].append({
                            "price": price,
                            "source": title[:25],
                            "context": snippet[:60]
                        })
                
                competitors = ['sysco', 'us foods', 'performance food', 'gordon', 'reinhart']
                for comp in competitors:
                    if comp.lower() in snippet.lower() or comp.lower() in title.lower():
                        pricing_data['competitor_info'].append({
                            "competitor": comp.title(),
                            "mention": snippet[:80]
                        })
                        break
        
        pricing_data['search_metadata'] = {
            "total_results": len(pricing_data['organic_results']),
            "prices_found": len(pricing_data['pricing_mentions']),
            "competitor_mentions": len(pricing_data['competitor_info'])
        }
        
        logger.info(f"Search complete: {pricing_data['search_metadata']}")
        return pricing_data
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"enabled": False, "error": str(e)}

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
    s = series.astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# =========================
# Pareto Analysis Functions
# =========================
def calculate_pareto_analysis(df: pd.DataFrame, group_by_col: str, value_col: str) -> pd.DataFrame:
    if group_by_col not in df.columns or value_col not in df.columns:
        logger.error(f"Missing columns")
        return pd.DataFrame()
    
    pareto_df = (
        df.groupby(group_by_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(by=value_col, ascending=False)
        .reset_index(drop=True)
    )
    
    total_value = pareto_df[value_col].sum()
    if total_value == 0:
        pareto_df['revenue_pct'] = 0
        pareto_df['cumulative_revenue_pct'] = 0
    else:
        pareto_df['revenue_pct'] = (pareto_df[value_col] / total_value * 100).round(2)
        pareto_df['cumulative_revenue_pct'] = pareto_df['revenue_pct'].cumsum().round(2)
    
    count_df = df.groupby(group_by_col).size().reset_index(name='transaction_count')
    pareto_df = pareto_df.merge(count_df, on=group_by_col, how='left')
    
    total_count = pareto_df['transaction_count'].sum()
    if total_count == 0:
        pareto_df['count_pct'] = 0
        pareto_df['cumulative_count_pct'] = 0
    else:
        pareto_df['count_pct'] = (pareto_df['transaction_count'] / total_count * 100).round(2)
        pareto_df['cumulative_count_pct'] = pareto_df['count_pct'].cumsum().round(2)
    
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
    metrics = {}
    analysis_col = ITEM_DESC_COL if analysis_type == "product" else MATERIAL_GROUP_COL
    product_df = df[df[analysis_col].astype(str) == str(product_id)]
    
    if product_df.empty:
        return metrics
    
    metrics['total_revenue'] = float(product_df[REVENUE_COL].sum())
    metrics['total_quantity'] = float(product_df[QUANTITY_COL].sum()) if QUANTITY_COL in product_df.columns else 0
    metrics['transaction_count'] = int(len(product_df))
    
    if PRICE_COL in product_df.columns:
        metrics['avg_price'] = float(product_df[PRICE_COL].mean())
        metrics['price_std'] = float(product_df[PRICE_COL].std())
        metrics['min_price'] = float(product_df[PRICE_COL].min())
        metrics['max_price'] = float(product_df[PRICE_COL].max())
        metrics['median_price'] = float(product_df[PRICE_COL].median())
        metrics['price_volatility'] = float(product_df[PRICE_COL].std() / product_df[PRICE_COL].mean()) if product_df[PRICE_COL].mean() > 0 else 0.0
    
    if CUSTOMER_COL in product_df.columns:
        metrics['unique_customers'] = int(product_df[CUSTOMER_COL].nunique())
        metrics['avg_revenue_per_customer'] = float(metrics['total_revenue'] / metrics['unique_customers']) if metrics['unique_customers'] > 0 else 0
    
    if 'Created On' in product_df.columns:
        product_df_copy = product_df.copy()
        product_df_copy['month'] = pd.to_datetime(product_df_copy['Created On'], errors='coerce').dt.to_period('M')
        monthly = product_df_copy.groupby('month')[REVENUE_COL].sum().reset_index()
        monthly['month'] = monthly['month'].astype(str)
        metrics['monthly_revenue'] = monthly.to_dict('records')
    
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
# Helper Functions for JSON
# =========================
def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
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
        "competitive_insights": str(obj.get("competitive_insights", "")),
        "growth_opportunities": str(obj.get("growth_opportunities", "")),
        "risk_factors": str(obj.get("risk_factors", "")),
        "portfolio_action": str(obj.get("portfolio_action", "")),
        "revenue_metrics": {},
        "volume_metrics": {},
        "observation": [],
        "recommendation": [],
    }
    
    for key in ['revenue_metrics', 'volume_metrics']:
        metrics = obj.get(key, {})
        if isinstance(metrics, dict):
            out[key] = {k: float(v) if v else 0.0 for k, v in metrics.items()}
    
    out["observation"] = parse_nested_json_list_field(obj.get("observation", []))
    out["recommendation"] = parse_nested_json_list_field(obj.get("recommendation", []))
    
    def trim_words(s: str, n: int) -> str:
        return " ".join(str(s).split()[:n])
    
    out["performance_summary"] = trim_words(out["performance_summary"], 30)
    out["pareto_insights"] = trim_words(out["pareto_insights"], 30)
    out["pricing_analysis"] = trim_words(out["pricing_analysis"], 40)
    out["competitive_insights"] = trim_words(out["competitive_insights"], 40)
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
        
        df = None
        try:
            df = pd.read_csv(io.StringIO(data_string), sep='\t', dtype=str, on_bad_lines='skip', low_memory=False)
            if len(df.columns) > 5:
                logger.info("Parsed with tab separator")
            else:
                df = None
        except Exception:
            pass
        
        if df is None or len(df.columns) <= 5:
            try:
                df = pd.read_csv(io.StringIO(data_string), dtype=str, on_bad_lines='skip', low_memory=False, sep=None, engine='python')
                logger.info("Parsed with auto-detect")
            except Exception:
                df = pd.read_csv(io.StringIO(data_string), dtype=str, on_bad_lines='skip', low_memory=False)
                logger.info("Parsed with comma separator")
        
        logger.info("CSV loaded: %s", df.shape)
        return df
        
    except Exception as e:
        logger.exception("Failed to download CSV: %s", e)
        raise

# Load data
try:
    raw_df = load_df_from_url(CSV_URL)
    
    required_cols = [MATERIAL_GROUP_COL, ITEM_DESC_COL, REVENUE_COL]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if REVENUE_COL in raw_df.columns:
        raw_df[REVENUE_COL] = clean_numeric_column(raw_df[REVENUE_COL])
    
    if PRICE_COL in raw_df.columns:
        raw_df[PRICE_COL] = clean_numeric_column(raw_df[PRICE_COL])
    
    if QUANTITY_COL in raw_df.columns:
        raw_df[QUANTITY_COL] = clean_numeric_column(raw_df[QUANTITY_COL])
    
    for col in ["Created On", "Billing Date"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors='coerce')
    
    logger.info("Starting Pareto analysis...")
    pareto_by_material_group = calculate_pareto_analysis(raw_df, MATERIAL_GROUP_COL, REVENUE_COL)
    pareto_by_product = calculate_pareto_analysis(raw_df, ITEM_DESC_COL, REVENUE_COL)
    
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
    
    logger.info("Pareto complete: %d groups, %d products", 
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
    return {
        "status": "ok",
        "message": "Product Performance API with Pareto Analysis",
        "features": {
            "pareto_analysis": "ABC classification with 80/20 rule",
            "live_pricing": "Real-time competitive pricing via Serper.dev",
            "ai_insights": "Gemini-powered recommendations"
        }
    }

@app.get("/health")
async def health():
    serper_status = "configured" if SERPER_API_KEY and SERPER_API_KEY != "YOUR_SERPER_API_KEY_HERE" else "not_configured"
    
    return {
        "status": "ok",
        "total_rows": int(len(raw_df)),
        "material_groups": int(len(pareto_by_material_group)),
        "products": int(len(pareto_by_product)),
        "serper_api": serper_status
    }

@app.get("/test-serper")
async def test_serper():
    """Test Serper API"""
    if not SERPER_API_KEY:
        return {"error": "API key not configured"}
    
    try:
        payload = json.dumps({"q": "beverage syrup wholesale price", "num": 5, "gl": "us"})
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        response = requests.post("https://google.serper.dev/search", headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        return {
            "status": "success",
            "results_count": len(results.get('organic', [])),
            "sample": results.get('organic', [])[:2]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/pareto-data")
async def get_pareto_data():
    """Get Pareto analysis data"""
    if pareto_data.empty:
        raise HTTPException(status_code=503, detail="Pareto analysis not available")
    
    output_cols = ['identifier', 'type', REVENUE_COL, 'revenue_pct', 'cumulative_revenue_pct', 
                   'abc_class', 'rank', 'transaction_count', 'count_pct']
    
    available_cols = [col for col in output_cols if col in pareto_data.columns]
    result = pareto_data[available_cols].copy()
    result = result.rename(columns={'identifier': 'name', REVENUE_COL: 'total_revenue'})
    
    return result.to_dict(orient="records")

@app.get("/product-insights/{product_identifier}")
async def get_product_insights(
    product_identifier: str,
    analysis_type: str = Query("product", regex="^(product|material_group)$"),
    enable_live_pricing: bool = Query(True),
    debug: bool = Query(False)
):
    """Get AI-powered product insights with competitive pricing"""
    logger.info("GET /product-insights/%s type=%s live=%s", product_identifier, analysis_type, enable_live_pricing)
    
    try:
        if raw_df.empty:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        analysis_col = ITEM_DESC_COL if analysis_type == "product" else MATERIAL_GROUP_COL
        product_df = raw_df[raw_df[analysis_col].astype(str) == str(product_identifier)]
        
        if product_df.empty:
            raise HTTPException(status_code=404, detail=f"{analysis_type} not found")
        
        pareto_df = pareto_by_product if analysis_type == "product" else pareto_by_material_group
        if pareto_df.empty:
            raise HTTPException(status_code=503, detail="Pareto not available")
        
        pareto_row = pareto_df[pareto_df[analysis_col] == product_identifier].head(1)
        if pareto_row.empty:
            raise HTTPException(status_code=404, detail="Not in Pareto")
        
        pareto_data_single = pareto_row.to_dict('records')[0]
        abc_class = pareto_data_single.get('abc_class', 'C')
        
        performance_metrics = calculate_product_performance_metrics(raw_df, product_identifier, analysis_type)
        
        competitive_pricing = {}
        if enable_live_pricing:
            avg_price = performance_metrics.get('avg_price', 0)
            competitive_pricing = search_competitive_pricing(product_identifier, analysis_type, avg_price)
        
        # Build minimal prompt
        price_context = ""
        if competitive_pricing.get('enabled') and competitive_pricing.get('pricing_mentions'):
            prices = [p['price'] for p in competitive_pricing['pricing_mentions'][:3]]
            price_context = f"Market prices: {', '.join(prices)}. Your avg: ${performance_metrics.get('avg_price', 0):.2f}"
        
        prompt = f"""Analyze product performance and return ONLY valid JSON:

Product: {product_identifier}
ABC Class: {abc_class}
Revenue: ${performance_metrics.get('total_revenue', 0):,.0f}
Transactions: {performance_metrics.get('transaction_count', 0)}
Avg Price: ${performance_metrics.get('avg_price', 0):.2f}
Rank: #{pareto_data_single.get('rank', 0)}
{price_context}

Return JSON:
{{
  "product_or_group": "{product_identifier}",
  "abc_classification": "{abc_class}",
  "performance_summary": "summary in 25 words",
  "pareto_insights": "position in 25 words",
  "pricing_analysis": "price analysis in 30 words",
  "competitive_insights": "market comparison in 30 words",
  "growth_opportunities": "opportunities in 25 words",
  "risk_factors": "risks in 25 words",
  "portfolio_action": "strategy in 15 words",
  "observation": [
    {{"key": "Revenue", "value": "insight"}},
    {{"key": "Position", "value": "insight"}},
    {{"key": "Customers", "value": "insight"}}
  ],
  "recommendation": [
    {{"key": "Pricing", "value": "action"}},
    {{"key": "Volume", "value": "action"}},
    {{"key": "Strategy", "value": "action"}}
  ]
}}"""

        # Call LLM
        raw_text = ""
        parsed = None
        
        try:
            response = model.generate_content(
                prompt,
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            )
            
            if response.candidates and response.candidates[0].finish_reason == 1:
                raw_text = response.text
                logger.info("LLM success: %d chars", len(raw_text))
                parsed = try_parse_json(raw_text)
            else:
                logger.warning("LLM blocked: reason %s", response.candidates[0].finish_reason if response.candidates else "unknown")
                
        except Exception as e:
            logger.error("LLM error: %s", str(e)[:100])
        
        # Process response
        if isinstance(parsed, dict):
            coerced = coerce_product_schema(parsed, product_identifier, abc_class)
            coerced['competitive_pricing_used'] = competitive_pricing.get('enabled', False)
            
            coerced['revenue_metrics'] = {
                "total_revenue": float(pareto_data_single.get(REVENUE_COL, 0)),
                "revenue_share_pct": float(pareto_data_single.get('revenue_pct', 0)),
                "cumulative_revenue_pct": float(pareto_data_single.get('cumulative_revenue_pct', 0)),
                "rank": int(pareto_data_single.get('rank', 0))
            }
            coerced['volume_metrics'] = {
                "total_transactions": int(performance_metrics.get('transaction_count', 0)),
                "transaction_share_pct": float(pareto_data_single.get('count_pct', 0)),
                "avg_order_value": float(performance_metrics.get('total_revenue', 0) / max(performance_metrics.get('transaction_count', 1), 1))
            }
            
            if debug:
                return {"result": coerced, "debug": {"competitive_pricing": competitive_pricing, "ai_generated": True}}
            return coerced
        
        # Smart fallback with competitive insights
        comp_insight = "No competitive pricing data available"
        if competitive_pricing.get('enabled') and competitive_pricing.get('pricing_mentions'):
            mentions = competitive_pricing.get('pricing_mentions', [])
            prices = [p['price'] for p in mentions[:3]]
            avg = performance_metrics.get('avg_price', 0)
            comp_insight = f"Market prices range {prices[0]}-{prices[-1]} vs your ${avg:.2f}. Consider competitive positioning"
        
        fallback = {
            "product_or_group": product_identifier,
            "abc_classification": abc_class,
            "performance_summary": f"Top performer: ${performance_metrics.get('total_revenue', 0):,.0f} revenue, {performance_metrics.get('transaction_count', 0)} transactions, {performance_metrics.get('unique_customers', 0)} customers",
            "pareto_insights": f"Ranked #{pareto_data_single.get('rank', 0)}, contributes {pareto_data_single.get('revenue_pct', 0):.1f}% of portfolio revenue, ABC Class {abc_class}",
            "pricing_analysis": f"Avg price ${performance_metrics.get('avg_price', 0):.2f}, {performance_metrics.get('price_volatility', 0):.1%} volatility, {'stable' if performance_metrics.get('price_volatility', 0) < 0.2 else 'variable'} pricing",
            "competitive_insights": comp_insight,
            "growth_opportunities": f"Strong base of {performance_metrics.get('unique_customers', 0)} customers with high frequency. Consider volume incentives and bundling",
            "risk_factors": f"Price volatility {performance_metrics.get('price_volatility', 0):.1%}, {'high' if abc_class == 'A' else 'moderate'} revenue concentration requires monitoring",
            "portfolio_action": f"Class {abc_class}: {'Protect and expand' if abc_class == 'A' else 'Promote actively' if abc_class == 'B' else 'Evaluate fit'}",
            "revenue_metrics": {
                "total_revenue": float(pareto_data_single.get(REVENUE_COL, 0)),
                "revenue_share_pct": float(pareto_data_single.get('revenue_pct', 0)),
                "cumulative_revenue_pct": float(pareto_data_single.get('cumulative_revenue_pct', 0)),
                "rank": int(pareto_data_single.get('rank', 0))
            },
            "volume_metrics": {
                "total_transactions": int(performance_metrics.get('transaction_count', 0)),
                "transaction_share_pct": float(pareto_data_single.get('count_pct', 0)),
                "avg_order_value": float(performance_metrics.get('total_revenue', 0) / max(performance_metrics.get('transaction_count', 1), 1))
            },
            "observation": [
                {"key": "Revenue Leadership", "value": f"Ranks #{pareto_data_single.get('rank', 0)} with {pareto_data_single.get('revenue_pct', 0):.1f}% share"},
                {"key": "Customer Base", "value": f"{performance_metrics.get('unique_customers', 0)} customers, ${performance_metrics.get('avg_revenue_per_customer', 0):,.0f} avg each"},
                {"key": "Volume", "value": f"{performance_metrics.get('transaction_count', 0)} transactions, ${performance_metrics.get('total_revenue', 0) / max(performance_metrics.get('transaction_count', 1), 1):,.2f} avg order"},
                {"key": "Market Position", "value": comp_insight[:100]}
            ],
            "recommendation": [
                {"key": "Priority", "value": f"ABC Class {abc_class} - {'High retention' if abc_class == 'A' else 'Growth opportunity' if abc_class == 'B' else 'Review'}"},
                {"key": "Pricing", "value": f"Monitor ${performance_metrics.get('avg_price', 0):.2f} vs market. {'Prices found in search' if competitive_pricing.get('pricing_mentions') else 'Track competitors'}"},
                {"key": "Volume", "value": f"Leverage {performance_metrics.get('unique_customers', 0)}-customer base. Target ${(performance_metrics.get('total_revenue', 0) * 1.15):,.0f} (+15%)"},
                {"key": "Risk", "value": f"Monitor {performance_metrics.get('price_volatility', 0):.1%} volatility, {'high dependency' if pareto_data_single.get('revenue_pct', 0) > 40 else 'concentration'}"}
            ],
            "competitive_pricing_used": competitive_pricing.get('enabled', False)
        }
        
        if debug:
            return {"result": fallback, "debug": {"competitive_pricing": competitive_pricing, "fallback_used": True}}
        return fallback
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
