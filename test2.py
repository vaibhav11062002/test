import re
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
from sklearn.cluster import KMeans
import google.generativeai as genai


# Manual Config Variables
CSV_URL = "https://raw.githubusercontent.com/vaibhav11062002/Churn-poc/main/llm_all_cust.csv"  # Your dataset CSV URL
GEMINI_API_KEY = "AIzaSyDDrwMUTp75A3Dc64auV-SHSv402mS1w4M"  # Your Gemini API key
MODEL_NAME = "gemini-2.5-flash"  # Or any other Gemini model as needed


REGION_COL = "Sales Organization"  # Main column to group regions, adjust as needed
REVENUE_COL = "Net Value"


# Fields to keep for region analysis
KEEP_COLS = [
    "Billing Date", "Created On", "Item Description", "Material Group", "Distribution Channel",
    "Terms of Payment", "Order Quantity", "Net Price", "Net Value", "Document Currency"
]


# CORS allowed origins including React dev and deployed frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://chrun-ai.vercel.app"
]
ALLOW_CREDENTIALS = True


# Logging config
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("region-insights")


# LLM Auth using API Key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required for LLM authentication")


try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info("Configured Google Generative AI client with API key")
except Exception as e:
    logger.exception("Failed to configure Generative AI client: %s", e)
    raise


# Helper Functions - same as customer analysis, adapted for region context
def sanitize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("|", "/")
    return s


def norm_date(d):
    if pd.isna(d):
        return ""
    val = pd.to_datetime(d, errors="coerce")
    return val.strftime("%Y%m%d") if pd.notna(val) else ""


def norm_num(x, ndigits=2):
    if pd.isna(x):
        return ""
    try:
        return str(round(float(x), ndigits))
    except Exception:
        return ""


def build_codebook(series: pd.Series) -> Dict[str, str]:
    vc = series.dropna().astype(str).value_counts()
    return {v: f"x{i+1}" for i, v in enumerate(vc.index.tolist())}


def try_parse_json(text: str):
    """Parse JSON from text, handling markdown code blocks."""
    if not text:
        return None
    
    # Remove markdown code blocks if present
    text = text.strip()
    
    # Check if wrapped in ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Remove opening ```json or ```
        text = re.sub(r'^```(?:json)?[\n\r]*', '', text)
        # Remove closing ```
        text = re.sub(r'[\n\r]*```$', '', text)
        text = text.strip()
    
    # Try to parse the cleaned text
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Fallback: try to extract JSON object from text
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    
    return None


def full_transaction_block_for_region_quarterly_agg(region_df: pd.DataFrame) -> Tuple[str, int]:
    g = region_df.copy()


    # Convert Billing Date or Created On to datetime if needed
    date_col = None
    if "Billing Date" in g.columns and g["Billing Date"].notna().any():
        date_col = "Billing Date"
    elif "Created On" in g.columns and g["Created On"].notna().any():
        date_col = "Created On"


    if date_col is None:
        return "", 0


    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    
    # Aggregate revenue and other numeric fields by quarter and material
    g["YearQuarter"] = g[date_col].dt.to_period("Q").astype(str)
    agg_cols = ["Order Quantity", "Net Price", "Net Value"]
    # Convert columns to numeric for aggregation
    for col in agg_cols:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0)


    # Aggregate metrics by YearQuarter and Item Description or Material Group
    item_col = "Item Description" if "Item Description" in g.columns else "Material Group" if "Material Group" in g.columns else None
    if item_col is None:
        return "", 0


    agg_df = g.groupby(["YearQuarter", item_col]).agg({
        "Order Quantity": "sum",
        "Net Price": "mean",  # average price per quarter-material
        "Net Value": "sum"
    }).reset_index()


    # Build codebooks for categorical fields
    codebooks = {item_col: build_codebook(agg_df[item_col])}


    keep_cols = ["YearQuarter", item_col, "Order Quantity", "Net Price", "Net Value"]


    header = "COLUMNS|" + "|".join(keep_cols)
    code_header = "CODES|" + json.dumps(codebooks, separators=(",", ":"), ensure_ascii=False)


    data_lines = []
    for _, row in agg_df.iterrows():
        fields = [
            sanitize_text(row["YearQuarter"]),
            codebooks[item_col].get(str(row[item_col]), "x0"),
            norm_num(row["Order Quantity"], 0),
            norm_num(row["Net Price"], 2),
            norm_num(row["Net Value"], 2)
        ]
        data_lines.append("ROW|" + "|".join(fields))


    compact_text = "\n".join([header, code_header] + data_lines)
    return compact_text, len(data_lines)



def compute_aggregates_for_region(region_df: pd.DataFrame) -> Dict[str, Any]:
    a: Dict[str, Any] = {}
    g = region_df.copy()
    date_cols = [c for c in ["Billing Date", "Created On"] if c in g.columns]
    date_col = date_cols[0] if date_cols else None


    if "Document Currency" in g.columns:
        cur_counts = g["Document Currency"].dropna().astype(str).value_counts()
        a["currency_mode"] = cur_counts.index[0] if not cur_counts.empty else ""
    else:
        a["currency_mode"] = ""


    if REVENUE_COL in g.columns:
        g[REVENUE_COL] = pd.to_numeric(g[REVENUE_COL], errors="coerce")
    a["total_revenue_local"] = float(g[REVENUE_COL].fillna(0).sum()) if REVENUE_COL in g.columns else 0.0


    revenue_by_year: Dict[str, float] = {}
    revenue_by_quarter: Dict[str, float] = {}


    if date_col:
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
        g_valid = g[pd.notna(g[date_col])].copy()
        if REVENUE_COL in g_valid.columns:
            rev_series = g_valid.groupby(g_valid[date_col].dt.year)[REVENUE_COL].sum(min_count=1)
            revenue_by_year = {str(int(k)): float(v) for k, v in rev_series.fillna(0).to_dict().items()}
            q_series = g_valid.groupby(g_valid[date_col].dt.to_period("Q"))[REVENUE_COL].sum(min_count=1)
            revenue_by_quarter = {str(k).replace("Q", "-Q"): float(v) for k, v in q_series.fillna(0).to_dict().items()}


    a["revenue_by_year"] = revenue_by_year
    a["revenue_by_quarter"] = revenue_by_quarter


    # Additional aggregates similar to customer - top products, price stats, purchase frequency


    item_col = None
    if "Item Description" in g.columns:
        item_col = "Item Description"
    elif "Material Group" in g.columns:
        item_col = "Material Group"


    a["top_materials_by_revenue"] = []
    a["price_stats"] = []
    a["top_copurchase_pairs"] = []


    if item_col:
        totals = g.groupby(item_col)[REVENUE_COL].sum(min_count=1).fillna(0).sort_values(ascending=False)
        grand = float(totals.sum()) if not totals.empty else 0.0
        top_rows = totals.head(10)
        top_materials = []
        for mat, val in top_rows.items():
            pct = (float(val) / grand) if grand > 0 else 0.0
            top_materials.append({"material": str(mat), "revenue": float(val), "share": round(pct, 4)})
        a["top_materials_by_revenue"] = top_materials


        if "Net Price" in g.columns:
            price_stats = []
            for mat, sub in g.groupby(item_col):
                prices = pd.to_numeric(sub["Net Price"], errors="coerce").dropna()
                if prices.empty:
                    continue
                avg = float(prices.mean())
                std = float(prices.std(ddof=0)) if len(prices) > 1 else 0.0
                cv = (std / avg) if avg > 0 else 0.0
                price_stats.append({"material": str(mat), "avg_price": round(avg, 4), "cv": round(cv, 4)})
            if top_materials:
                top_set = {t["material"] for t in top_materials}
                price_stats_sorted = sorted(price_stats, key=lambda d: (d["material"] not in top_set, d["material"]))
                a["price_stats"] = price_stats_sorted[:20]
            else:
                a["price_stats"] = price_stats[:20]


        date_key = None
        if "Billing Date" in g.columns and g["Billing Date"].notna().any():
            date_key = "Billing Date"
        elif "Created On" in g.columns and g["Created On"].notna().any():
            date_key = "Created On"


        pair_counts = Counter()
        if date_key:
            g[date_key] = pd.to_datetime(g[date_key], errors="coerce")
            for dt, sub in g.groupby(g[date_key].dt.date):
                mats = sorted(set(sub[item_col].dropna().astype(str).tolist()))
                if len(mats) < 2:
                    continue
                for a_mat, b_mat in combinations(mats, 2):
                    pair_counts[(a_mat, b_mat)] += 1
        top_pairs = [{"a": a_m, "b": b_m, "count": int(c)} for (a_m, b_m), c in pair_counts.most_common(5)]
        a["top_copurchase_pairs"] = top_pairs


    a["total_records"] = int(len(g))
    if date_col and g[date_col].notna().any():
        a["distinct_days"] = int(g[date_col].dt.date.nunique())
    else:
        a["distinct_days"] = 0


    return a


def cluster_regions(df_agg: pd.DataFrame) -> pd.DataFrame:
    df = df_agg.copy()
    df["rev_pos"] = df["total_revenue"].clip(lower=0)
    if len(df) < 3:
        ranks = df["rev_pos"].rank(method="first", ascending=False)
        labels = np.where(
            ranks <= 1, "high_revenue",
            np.where(ranks <= 2, "mixed_revenue", "low_revenue")
        )
        df["cluster_name"] = labels
        df["cluster_id"] = df["cluster_name"].map({"high_revenue": 0, "mixed_revenue": 1, "low_revenue": 2}).fillna(2).astype(int)
        return df.drop(columns=["rev_pos"])


    X = np.log1p(df["rev_pos"]).to_numpy().reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=50, random_state=42)
    df["km_id"] = km.fit_predict(X)
    means = df.groupby("km_id")["rev_pos"].mean().sort_values(ascending=False)
    order = {cid: idx for idx, cid in enumerate(means.index)}
    df["cluster_id"] = df["km_id"].map(order)
    df["cluster_name"] = df["cluster_id"].map({0: "high_revenue", 1: "mixed_revenue", 2: "low_revenue"})
    return df.drop(columns=["rev_pos", "km_id"])


# Load CSV from URL
def load_df_from_url(csv_url: str) -> pd.DataFrame:
    try:
        logger.info("Downloading CSV from URL %s", csv_url)
        response = requests.get(csv_url)
        response.raise_for_status()
        data_string = response.text
        df = pd.read_csv(io.StringIO(data_string), dtype=str)
        logger.info("CSV loaded with shape=%s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to download or parse CSV: %s", e)
        raise


# Load data
raw_df = load_df_from_url(CSV_URL)


for col in ["Created On", "Billing Date"]:
    if col in raw_df.columns:
        raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce")
if REGION_COL not in raw_df.columns:
    raw_df[REGION_COL] = "UNKNOWN"


# Clean revenue column
s = raw_df[REVENUE_COL].astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
raw_df[REVENUE_COL] = pd.to_numeric(s, errors="coerce").fillna(0.0)


# Clean Net Price for price stats
if "Net Price" in raw_df.columns:
    p = raw_df["Net Price"].astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
    raw_df["Net Price"] = pd.to_numeric(p, errors="coerce")


# Region-level aggregation
agg_region = (
    raw_df.groupby(REGION_COL)[REVENUE_COL]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={REVENUE_COL: "total_revenue"})
)
agg_region["total_revenue"] = agg_region["total_revenue"].fillna(0.0).astype(float)


# Cluster regions
clustered_regions = cluster_regions(agg_region)


# Prepare FastAPI app with a region insights endpoint
app = FastAPI(title="Region Clustering and Insights API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rows": int(len(raw_df)),
        "clustered_regions": int(len(clustered_regions))
    }


@app.get("/clustered-regions")
async def get_clustered_regions():
    logger.info("GET /clustered-regions")
    return clustered_regions.to_dict(orient="records")


@app.get("/region-insights/{region_id}")
async def get_region_insights(region_id: str, debug: bool = Query(False)):
    logger.info("GET /region-insights/%s debug=%s", region_id, debug)
    try:
        region_df = raw_df[raw_df[REGION_COL].astype(str) == str(region_id)]
        if region_df.empty:
            raise HTTPException(status_code=404, detail="Region not found")


        known_total_revenue = float(region_df[REVENUE_COL].fillna(0).sum())


        aggregates_json = compute_aggregates_for_region(region_df)
        compact, nlines = full_transaction_block_for_region_quarterly_agg(region_df)


        row = clustered_regions[clustered_regions[REGION_COL] == str(region_id)].head(1).to_dict("records")
        ctx = row[0] if row else {}
        context_json = {
            "cluster_name": ctx.get("cluster_name", ""),
            "known_total_revenue_from_cluster": float(ctx.get("total_revenue", known_total_revenue)),
        }


        # Build LLM prompt with comprehensive instructions for structured JSON output
        prompt_template = """You are a commercial analyst providing region-wise insights for sales leadership.

Your task is to analyze the region's commercial performance, revenue trends, product mix, and market dynamics.
Focus on actionable insights and strategic recommendations for revenue protection and growth.

Region Data Inputs:
- region_id: [[REGION_ID]]
- known_total_revenue: [[KNOWN_TOTAL_REVENUE]]
- aggregates_json: [[AGGREGATES_JSON]]

Recent History (Quarterly Aggregated Transaction Data):
[[COMPACT_BLOCK]]

Analysis Requirements:
1. Analyze revenue trends across quarters and years
2. Identify top-performing products and their revenue contribution
3. Assess product mix balance and diversification
4. Identify growth or decline patterns
5. Provide actionable retention and growth strategies
6. Consider seasonal patterns and customer behavior

CRITICAL: You MUST respond with ONLY a valid JSON object, nothing else.
Do not include any markdown formatting, code blocks, or explanatory text.
Do not wrap the JSON in triple backticks or any other formatting.

The JSON response must follow this exact schema:

{
  "region": "<region_id string>",
  "cluster": "<cluster classification: high_revenue, mixed_revenue, or low_revenue>",
  "revenue_by_year": {
    "YYYY": <numeric_value>,
    "YYYY": <numeric_value>
  },
  "revenue_by_quarter": {
    "YYYY-QN": <numeric_value>,
    "YYYY-QN": <numeric_value>
  },
  "trend_of_sales": "<Concise 40-word summary of sales trajectory, growth patterns, or decline trends>",
  "top_products": [
    {
      "material": "<product name or description>",
      "revenue": <numeric_value>,
      "share": <decimal_value between 0 and 1>
    }
  ],
  "observations": "<Minimum 75 words. 4-5 key business insights as bullet points. Example format: • Insight 1 with commercial context • Insight 2 about revenue drivers • Insight 3 on product performance • Insight 4 on customer behavior • Insight 5 on market opportunity>",
  "recommendations": "<Minimum 75 words. 4-5 actionable strategic recommendations as bullet points. Example format: • Action 1 with expected impact • Action 2 to protect revenue • Action 3 to drive growth • Action 4 for product optimization • Action 5 for customer engagement>"
}

Ensure all numeric values are properly formatted and all string fields are populated with meaningful content.
Remember: JSON ONLY, no additional text before or after."""


        prompt = prompt_template.replace("[[REGION_ID]]", json.dumps(region_id))
        prompt = prompt.replace("[[KNOWN_TOTAL_REVENUE]]", str(round(known_total_revenue, 4)))
        prompt = prompt.replace("[[AGGREGATES_JSON]]", json.dumps(aggregates_json, separators=(",", ":"), ensure_ascii=False))
        prompt = prompt.replace("[[COMPACT_BLOCK]]", compact)


        raw_text = ""
        parsed = None
        try:
            resp = model.generate_content(prompt)
            raw_text = resp.text or ""
            logger.info("LLM raw_text_len=%d", len(raw_text))
            logger.debug("LLM raw_text_head=%s", raw_text[:400])
            parsed = try_parse_json(raw_text)
            if parsed:
                logger.info("Successfully parsed LLM response to JSON")
            else:
                logger.warning("Failed to parse LLM response as JSON")
        except Exception as e:
            logger.error("LLM call failed: %s", e)


        # Simplified fallback if no LLM response or parsing failure
        fallback = {
            "region": region_id,
            "cluster": ctx.get("cluster_name", ""),
            "revenue_by_year": aggregates_json.get("revenue_by_year", {}),
            "revenue_by_quarter": aggregates_json.get("revenue_by_quarter", {}),
            "trend_of_sales": "",
            "top_products": aggregates_json.get("top_materials_by_revenue", []),
            "observations": "",
            "recommendations": "",
        }
        
        if debug:
            return {
                "result": parsed or fallback,
                "debug": {
                    "region_rows": int(len(region_df)),
                    "known_total_revenue": known_total_revenue,
                    "all_rows": int(nlines),
                    "raw_text_head": raw_text[:500],
                    "parsed": parsed is not None,
                }
            }
        return parsed or fallback


    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in get_region_insights for region %s: %s", region_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")