"""
Tools for EcoHome Energy Advisor Agent
"""

import os
import random
from datetime import datetime, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from models.energy import DatabaseManager

load_dotenv()
# Initialize database manager
db_manager = DatabaseManager()


# TODO: Implement get_weather_forecast tool
@tool
def get_weather_forecast(location: str, days: int = 3) -> dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-5)

    Returns:
        Dict[str, Any]: Current weather and forecast data including temperature, conditions, and solar irradiance

    """
    # Mock weather API or call OpenWeatherMap or similar
    with httpx.Client() as client:
        # Get IP location
        ip_location = client.get(os.environ["IPINFO_API_URL"])

        # Get lat, lon from ip_location
        lat = ip_location.json()["lat"]
        lon = ip_location.json()["lon"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": days,
            # Current weather data
            "current": ["temperature_2m", "weather_code"],
            # Hourly forecast data including conditions, temperature, and solar irradiance
            "hourly": [
                "temperature_2m",
                "weather_code",
                "shortwave_radiation",  # Global Horizontal Irradiance (GHI) in W/m²
                "direct_normal_irradiance",  # Direct Normal Irradiance (DNI) in W/m²
            ],
            "timezone": "auto",
        }

        # Get current weather and forecast
        response = client.get(f"{os.environ['WEATHER_API_URL']}", params=params)

    return response.json()


# TODO: Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
        E.g:
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date '{date}'. Use YYYY-MM-DD."}

    # Mock pricing - a real implementation would call a pricing API.
    # Seed from the date so repeated calls for the same day return identical
    # prices, while different days vary within a bounded range.
    rng = random.Random(date)

    base_rate = 0.12  # USD per kWh
    off_peak_multiplier = (0.6, 0.9)  # 22:00-05:59 -> $0.07-$0.11
    peak_multiplier = (1.3, 2.0)  # 06:00-21:59 -> $0.16-$0.24
    evening_surge = (1.0, 1.25)  # extra bump for 17:00-20:59
    demand_charge_range = (0.02, 0.08)  # USD per kWh, peak only

    hourly_rates = []
    for hour in range(24):
        is_peak = 6 <= hour < 22
        if is_peak:
            multiplier = rng.uniform(*peak_multiplier)
            if 17 <= hour < 21:
                multiplier *= rng.uniform(*evening_surge)
            demand_charge = round(rng.uniform(*demand_charge_range), 4)
        else:
            multiplier = rng.uniform(*off_peak_multiplier)
            demand_charge = 0.0

        hourly_rates.append(
            {
                "hour": hour,
                "rate": round(base_rate * multiplier, 4),
                "period": "peak" if is_peak else "off_peak",
                "demand_charge": demand_charge,
            }
        )

    return {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates,
    }


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": [],
        }

        for record in records:
            usage_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": record.consumption_kwh,
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_usd": record.cost_usd,
                }
            )

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {e!s}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records)
                / max(1, (end_dt - start_dt).days),
                2,
            ),
            "records": [],
        }

        for record in records:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {e!s}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(
                    sum(r.consumption_kwh for r in usage_records), 2
                ),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": "sunny" if generation_records else "unknown",
            },
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += (
                record.consumption_kwh
            )
            summary["usage"]["device_breakdown"][device]["cost_usd"] += (
                record.cost_usd or 0
            )
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {e!s}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in [
                "data/documents/tip_device_best_practices.txt",
                "data/documents/tip_energy_savings.txt",
            ]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings(
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_API_KEY"),
            )
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings(
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_API_KEY"),
            )
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            )

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {"query": query, "total_results": len(docs), "tips": []}

        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high"
                    if i < 2
                    else "medium"
                    if i < 4
                    else "low",
                }
            )

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {e!s}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    )

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
