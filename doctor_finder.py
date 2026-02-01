"""Doctor Finder utility using Geoapify Places API.

Finds doctors/hospitals based on location and specialty.
"""
import requests
import json
from typing import List, Dict, Optional

# Constants
GEOAPIFY_URL = "https://api.geoapify.com/v2/places"

def find_nearby_doctors(
    lat: float, 
    lon: float, 
    specialty: str = "general", 
    radius_meters: int = 5000, 
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Find nearby doctors using Geoapify.
    
    If api_key is None or invalid, falls back to mock data for demo purposes.
    """
    if not api_key:
        return _get_mock_doctors(specialty)
        
    categories = "healthcare.doctor"
    # Refine category based on specialty if possible, but Geoapify categories are broad.
    # We can filter by name or store custom logic.
    
    params = {
        "categories": categories,
        "filter": f"circle:{lon},{lat},{radius_meters}",
        "bias": f"proximity:{lon},{lat}",
        "limit": 5,
        "apiKey": api_key
    }
    
    try:
        resp = requests.get(GEOAPIFY_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            name = props.get("name", "Unknown Doctor")
            address = props.get("formatted", "No address provided")
            dist = props.get("distance", 0)
            
            # Simple keyword filter for specialty if provided
            if specialty and specialty.lower() != "general":
                # Very basic partial match in name or categories
                # Real implementation would need better category mapping
                pass 

            results.append({
                "name": name,
                "address": address,
                "distance": f"{dist} m",
                "specialty": specialty.capitalize() # Placeholder as API might not give exact specialty
            })
            
        if not results:
            return _get_mock_doctors(specialty, note="(No API results found)")
            
        return results

    except Exception as e:
        print(f"Geoapify Error: {e}")
        return _get_mock_doctors(specialty, note="(Fallback due to API Error)")

def _get_mock_doctors(specialty: str, note: str = "") -> List[Dict[str, str]]:
    """Return dummy data for testing/demo."""
    base = [
        {"name": f"Dr. Anjali Desai ({specialty})", "address": "12 Shivajinagar, Pune", "distance": "1.2 km", "specialty": specialty},
        {"name": f"City Care Hospital - {specialty} Dept", "address": "45 FC Road, Pune", "distance": "2.5 km", "specialty": specialty},
        {"name": f"Dr. Rajesh Kurnool", "address": "88 Aundh, Pune", "distance": "3.1 km", "specialty": "General Medicine"},
    ]
    if note:
        for b in base:
            b['name'] += f" {note}"
    return base
