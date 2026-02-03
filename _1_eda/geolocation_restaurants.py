import pandas as pd
import requests
import time
from typing import List, Dict

# ==================== CONFIGURATION ====================
GOOGLE_API_KEY = "AIzaSyAFVk_x1rnmuPvPue9spUPiVpvoc6iHtQw"
API_DELAY = 0.2  # Slight delay to be safe
# =======================================================

def extract_city_from_components(components: List[Dict]) -> Dict:
    """
    Parses the new API's addressComponents to find City and Country.
    """
    loc = {
        'city': None,
        'country': None,
        'postal_code': None
    }
    
    if not components:
        return loc

    for comp in components:
        types = comp.get('types', [])
        
        # Priority 1: Locality (City)
        if 'locality' in types:
            loc['city'] = comp.get('longText') or comp.get('shortText')
        
        # Priority 2: Postal Town (common in UK/Europe if locality is missing)
        elif 'postal_town' in types and not loc['city']:
            loc['city'] = comp.get('longText')
            
        # Priority 3: Admin Level 2 (District) - Fallback
        elif 'administrative_area_level_2' in types and not loc['city']:
             loc['city'] = comp.get('longText')

        # Country
        if 'country' in types:
            loc['country'] = comp.get('longText')
            
        # Postal Code
        if 'postal_code' in types:
            loc['postal_code'] = comp.get('longText')
            
    return loc

def search_place_new_api(query_string: str, api_key: str) -> Dict:
    """
    Uses the Google Places API (New) to search for a place.
    Endpoint: https://places.googleapis.com/v1/places:searchText
    """
    
    url = "https://places.googleapis.com/v1/places:searchText"
    
    # 1. Define the fields we want (Field Mask)
    # This is efficient: we pay only for what we ask for.
    field_mask = [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.addressComponents",  # <--- This gives us the city directly!
        "places.types",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.websiteUri",
        "places.nationalPhoneNumber"
    ]
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(field_mask)
    }
    
    payload = {
        "textQuery": query_string
    }

    # Initialize empty result
    result_data = {
        'input_string': query_string,
        'found': False,
        'official_name': None,
        'city': None,
        'country': None,
        'formatted_address': None,
        'rating': None,
        'website': None,
        'error': None
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        data = resp.json()
        
        # Check for error in response body
        if 'error' in data:
            result_data['error'] = data['error'].get('message', 'Unknown API Error')
            return result_data

        # Check if any places were found
        if not data.get('places'):
            result_data['error'] = "No results found"
            return result_data

        # Get the first match
        place = data['places'][0]
        
        # Extract Location details
        address_components = place.get('addressComponents', [])
        location_info = extract_city_from_components(address_components)
        
        # Update result dictionary
        result_data.update({
            'found': True,
            'official_name': place.get('displayName', {}).get('text'),
            'city': location_info['city'],
            'country': location_info['country'],
            'formatted_address': place.get('formattedAddress'),
            'rating': place.get('rating'),
            'website': place.get('websiteUri'),
            'error': None
        })
        
        return result_data

    except Exception as e:
        result_data['error'] = str(e)
        return result_data

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Error: Please put your Google API Key in line 7")
        exit(1)

    # 1. Load Data
    try:
        print("📂 Reading 'names_to_search.csv'...")
        df_input = pd.read_csv('names_to_search.csv')
        restaurant_names = df_input['name'].dropna().tolist()
        print(f"   Loaded {len(restaurant_names)} names.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        # Fallback for testing if file doesn't exist
        print("   ⚠️ specific file not found, using test list...")
        restaurant_names = ["Nobu London", "Audrey Cafe Bangkok"]

    # 2. Process
    results = []
    print(f"\n🚀 Starting Search using Places API (New)...")
    
    for i, name in enumerate(restaurant_names):
        print(f"   [{i+1}/{len(restaurant_names)}] Searching: {name}")
        
        data = search_place_new_api(name, GOOGLE_API_KEY)
        results.append(data)
        
        # Show immediate feedback if city is missing
        if data['found'] and not data['city']:
             print(f"      ⚠️ Found place, but city field was empty. Address: {data.get('formatted_address')}")
             
        time.sleep(API_DELAY)

    # 3. Save
    df_results = pd.DataFrame(results)
    
    # Preview
    cols = ['input_string', 'official_name', 'city', 'rating', 'error']
    print("\n📊 Results Preview:")
    print(df_results[[c for c in cols if c in df_results.columns]].head().to_string(index=False))
    
    df_results.to_csv('places_api_new_results.csv', index=False)
    print("\n✅ Saved to 'places_api_new_results.csv'")