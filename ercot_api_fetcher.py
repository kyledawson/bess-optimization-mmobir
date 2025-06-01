import os
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API endpoints configuration
ERCOT_ENDPOINTS_FILE = os.getenv("ERCOT_ENDPOINTS_FILE", "docs/ercot_endpoints.json")
with open(ERCOT_ENDPOINTS_FILE, "r") as f:
    ERCOT_ENDPOINTS = json.load(f)

# Base URLs and constants
ERCOT_API_BASE_URL = ERCOT_ENDPOINTS["servers"][0]["url"]
AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
API_KEY_HEADER = "Ocp-Apim-Subscription-Key"

# Environment variables
ERCOT_API_KEY = os.getenv("ERCOT_API_KEY")
ERCOT_USERNAME = os.getenv("ERCOT_USERNAME")
ERCOT_PASSWORD = os.getenv("ERCOT_PASSWORD")


class ERCOTAPIError(Exception):
    """Custom exception for ERCOT API errors"""

    pass


def get_auth_token() -> Optional[str]:
    """
    Gets an authentication token from ERCOT's B2C service.
    Returns:
        str: The access token if successful, None if failed
    """
    if not all([ERCOT_USERNAME, ERCOT_PASSWORD]):
        raise ERCOTAPIError(
            "Missing ERCOT_USERNAME or ERCOT_PASSWORD in environment variables"
        )

    auth_params = {
        "username": ERCOT_USERNAME,
        "password": ERCOT_PASSWORD,
        "grant_type": "password",
        "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
        "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
        "response_type": "token",
    }

    try:
        response = requests.post(AUTH_URL, data=auth_params)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Error getting authentication token: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        return None


def get_endpoint_info(category: str, endpoint_name: str) -> Dict[str, Any]:
    """
    Get endpoint information from the loaded configuration.

    Args:
        category: The endpoint category (real_time, day_ahead, historical, other)
        endpoint_name: The endpoint name

    Returns:
        Dict containing endpoint information
    """
    try:
        return ERCOT_ENDPOINTS["endpoints"][category][endpoint_name]
    except KeyError:
        raise ERCOTAPIError(
            f"Endpoint {category}/{endpoint_name} not found in configuration"
        )


def format_datetime(dt: Union[str, datetime], fmt: str = "%Y-%m-%dT%H:%M:%S") -> str:
    """Format datetime object or string according to ERCOT's requirements"""
    if isinstance(dt, str):
        return dt
    return dt.strftime(fmt)


def fetch_ercot_data(
    category: str,
    endpoint_name: str,
    params: Dict[str, Any] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetch data from any ERCOT API endpoint using the endpoints configuration.

    Args:
        category: Category in the endpoints.json (e.g., 'real_time')
        endpoint_name: Name of the endpoint in that category
        params: Dictionary of query parameters to send with the request
        debug: Whether to print debug information

    Returns:
        Dict containing the parsed JSON response from the API, or None if an error occurs
    """
    if not ERCOT_API_KEY:
        raise ERCOTAPIError("Missing ERCOT_API_KEY in environment variables")

    # Get endpoint configuration
    endpoint_info = get_endpoint_info(category, endpoint_name)
    endpoint_url = f"{ERCOT_API_BASE_URL}{endpoint_info['path']}"

    # Get authentication token
    access_token = get_auth_token()
    if not access_token:
        raise ERCOTAPIError("Failed to get authentication token")

    headers = {
        API_KEY_HEADER: ERCOT_API_KEY,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    if debug:
        print(f"Endpoint URL: {endpoint_url}")
        print(f"Parameters: {params}")
        print(f"Headers: {headers}")

    try:
        response = requests.get(
            endpoint_url, headers=headers, params=params, timeout=30
        )
        response.raise_for_status()

        data = response.json()

        if debug:
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            if data.get("_meta"):
                print(f"Total Records: {data['_meta'].get('totalRecords', 0)}")

        return data["data"]

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        return None


def fetch_real_time_lmp(
    settlement_point: str = "HB_NORTH",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time LMP data for a specific settlement point.

    Args:
        settlement_point: The settlement point to fetch data for
        start_time: Start time for the query (defaults to 15 minutes ago)
        end_time: End time for the query (defaults to now)
        debug: Whether to print debug information

    Returns:
        Dict containing the LMP data
    """
    if start_time is None:
        # Default to the start of the current day in UTC
        start_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if end_time is None:
        # Default to the end of the current day in UTC (or just the same day as start_time)
        end_time = datetime.now(timezone.utc)

    # Extract date parts for the API parameters
    start_date_str = format_datetime(start_time, "%Y-%m-%d")
    end_date_str = format_datetime(end_time, "%Y-%m-%d")

    # Note: This endpoint uses date ranges, not specific timestamps.
    # If you need finer granularity (hour/interval), add deliveryHourFrom/To and deliveryIntervalFrom/To parameters.
    params = {
        "settlementPoint": settlement_point,
        "deliveryDateFrom": start_date_str,
        "deliveryDateTo": end_date_str,
        # "page": 1,
        # "size": 100,
    }

    return fetch_ercot_data(
        category="other", endpoint_name="np6_905_cd", params=params, debug=debug
    )


def fetch_day_ahead_prices(
    settlement_point: str = "HB_NORTH",
    delivery_date: Optional[Union[str, datetime]] = None,
    debug: bool = False,
) -> Optional[list[list[Any]]]:
    """
    Fetch day-ahead settlement point prices.

    Args:
        settlement_point: The settlement point to fetch data for
        delivery_date: The delivery date to fetch data for (defaults to today)
        debug: Whether to print debug information

    Returns:
        list of lists containing the day-ahead prices data
    """
    if delivery_date is None:
        # Default to today's date in UTC
        delivery_date_obj = datetime.now(timezone.utc).date()
    elif isinstance(delivery_date, datetime):
        delivery_date_obj = delivery_date.date()
    elif isinstance(delivery_date, str):
        # Assume YYYY-MM-DD format if string is passed
        try:
            delivery_date_obj = datetime.strptime(delivery_date, "%Y-%m-%d").date()
        except ValueError:
            raise ERCOTAPIError(
                "Invalid string format for delivery_date. Use YYYY-MM-DD."
            )
    else:
        delivery_date_obj = delivery_date  # Assume it's already a date object

    # Format the date string
    date_str = format_datetime(delivery_date_obj, "%Y-%m-%d")

    # This endpoint requires From and To dates. We'll use the same date for both
    # to fetch data for a single day.
    params = {
        "settlementPoint": settlement_point,
        "deliveryDateFrom": date_str,
        "deliveryDateTo": date_str,
        "page": 1,
        "size": 100,
    }

    return fetch_ercot_data(
        category="other", endpoint_name="np4_190_cd", params=params, debug=debug
    )


# Example usage
if __name__ == "__main__":
    print("ERCOT API Fetcher Test")

    try:
        # Test real-time LMP data
        print("\nFetching Real-Time LMP Data...")
        rt_data = fetch_real_time_lmp(debug=True)
        if rt_data and rt_data.get("data"):
            print("\nReal-Time LMP Data Sample:")
            print(json.dumps(rt_data["data"][:2], indent=2))

        # Test day-ahead prices
        print("\nFetching Day-Ahead Prices...")
        da_data = fetch_day_ahead_prices(debug=True)
        if da_data and da_data.get("data"):
            print("\nDay-Ahead Price Data Sample:")
            print(json.dumps(da_data["data"][:2], indent=2))

    except ERCOTAPIError as e:
        print(f"\nERCOT API Error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\nFetcher Test Complete")
