import os
import re
import time
import datetime
import random
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# 110+ major global destinations to gather a rich, diverse, and large dataset of 5,000+ entries
CITIES = [
    # Europe
    "London", "Paris", "Rome", "Barcelona", "Amsterdam", "Berlin", "Prague", "Madrid",
    "Lisbon", "Milan", "Venice", "Florence", "Munich", "Budapest", "Dublin", "Edinburgh",
    "Vienna", "Zurich", "Geneva", "Brussels", "Copenhagen", "Stockholm", "Oslo", "Helsinki",
    "Athens", "Krakow", "Warsaw", "Nice", "Lyon", "Marseille", "Seville", "Porto", "Reykjavik",
    "Dubrovnik", "Salzburg", "Bratislava",

    # North America
    "New York", "Los Angeles", "San Francisco", "Las Vegas", "Miami", "Chicago", "Toronto",
    "Vancouver", "Montreal", "Seattle", "Boston", "Washington DC", "Austin", "Orlando", "Honolulu",
    "New Orleans", "San Diego", "Denver", "Philadelphia", "Atlanta", "Nashville", "Mexico City",
    "Cancun", "Quebec City", "Calgary", "Boston", "Houston", "Dallas", "Phoenix",

    # Asia
    "Tokyo", "Kyoto", "Osaka", "Singapore", "Bangkok", "Hong Kong", "Seoul", "Shanghai",
    "Beijing", "Taipei", "Mumbai", "New Delhi", "Bengaluru", "Dubai", "Abu Dhabi", "Doha",
    "Kuala Lumpur", "Jakarta", "Manila", "Ho Chi Minh City", "Hanoi", "Bali", "Phuket",
    "Chiang Mai", "Macau", "Fukuoka", "Sapporo", "Nagoya", "Incheon",

    # South/Central America
    "Rio de Janeiro", "Sao Paulo", "Buenos Aires", "Santiago", "Lima", "Bogota", "Quito",
    "San Jose", "Panama City", "Cartagena", "Cusco",

    # Oceania
    "Sydney", "Melbourne", "Brisbane", "Auckland", "Queenstown", "Christchurch", "Perth",
    "Adelaide", "Wellington", "Gold Coast",

    # Africa & Middle East
    "Cairo", "Cape Town", "Johannesburg", "Marrakech", "Casablanca", "Nairobi", "Sharm El Sheikh",
    "Riyadh", "Tel Aviv", "Istanbul", "Amman", "Muscat"
]

# --- NOUVEAU : mapping ville -> région, dérivé directement des groupes ci-dessus.
# Sert à créer une feature "region" à faible cardinalité (7 valeurs) en complément
# du one-hot sur "city" (124 valeurs), qui pénalisait Random Forest / Gradient Boosting
# à cause de la sparsité (cf. notebook : R2 RF 0.58 vs R2 regression lineaire 0.67).
CITY_REGION = {}
_regions_def = {
    "Europe": ["London", "Paris", "Rome", "Barcelona", "Amsterdam", "Berlin", "Prague", "Madrid",
               "Lisbon", "Milan", "Venice", "Florence", "Munich", "Budapest", "Dublin", "Edinburgh",
               "Vienna", "Zurich", "Geneva", "Brussels", "Copenhagen", "Stockholm", "Oslo", "Helsinki",
               "Athens", "Krakow", "Warsaw", "Nice", "Lyon", "Marseille", "Seville", "Porto", "Reykjavik",
               "Dubrovnik", "Salzburg", "Bratislava"],
    "North America": ["New York", "Los Angeles", "San Francisco", "Las Vegas", "Miami", "Chicago", "Toronto",
                       "Vancouver", "Montreal", "Seattle", "Boston", "Washington DC", "Austin", "Orlando",
                       "Honolulu", "New Orleans", "San Diego", "Denver", "Philadelphia", "Atlanta", "Nashville",
                       "Mexico City", "Cancun", "Quebec City", "Calgary", "Houston", "Dallas", "Phoenix"],
    "Asia": ["Tokyo", "Kyoto", "Osaka", "Singapore", "Bangkok", "Hong Kong", "Seoul", "Shanghai",
             "Beijing", "Taipei", "Mumbai", "New Delhi", "Bengaluru", "Dubai", "Abu Dhabi", "Doha",
             "Kuala Lumpur", "Jakarta", "Manila", "Ho Chi Minh City", "Hanoi", "Bali", "Phuket",
             "Chiang Mai", "Macau", "Fukuoka", "Sapporo", "Nagoya", "Incheon"],
    "South/Central America": ["Rio de Janeiro", "Sao Paulo", "Buenos Aires", "Santiago", "Lima", "Bogota",
                               "Quito", "San Jose", "Panama City", "Cartagena", "Cusco"],
    "Oceania": ["Sydney", "Melbourne", "Brisbane", "Auckland", "Queenstown", "Christchurch", "Perth",
                "Adelaide", "Wellington", "Gold Coast"],
    "Africa & Middle East": ["Cairo", "Cape Town", "Johannesburg", "Marrakech", "Casablanca", "Nairobi",
                              "Sharm El Sheikh", "Riyadh", "Tel Aviv", "Istanbul", "Amman", "Muscat"],
}
for _region, _cities in _regions_def.items():
    for _c in _cities:
        CITY_REGION[_c] = _region

# Output file path
OUTPUT_FILE = "booking_hotels_dataset.csv"

# Configuration
PAGES_PER_CITY = 3  # Attempt up to 3 pages to get extra data if pagination is available
MAX_TOTAL_ENTRIES = 5100  # Target clean entries count


def scroll_down(page):
    """
    Scrolls down the page slowly to trigger lazy loading of prices/ratings
    and ensure footer pagination elements are rendered and interactive.
    """
    try:
        for i in range(5):
            page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {i+1} / 5)")
            time.sleep(0.8)
    except Exception as e:
        print(f" (Scroll error: {e})", end="")


def dismiss_popups(page):
    """
    NOUVEAU : ferme la bannière cookies et la popup "sign in and save 10%" si présentes.
    Ces popups avaient bloqué la première tentative de scraping (documentée dans le rapport) ;
    on les gère explicitement ici plutôt que de compter uniquement sur click(force=True) plus loin.
    """
    selectors_to_try = [
        '#onetrust-accept-btn-handler',                  # cookie banner
        'button[aria-label="Dismiss sign in information."]',  # sign-in popup close
        '[data-testid="modal-close-icon"]',
    ]
    for sel in selectors_to_try:
        try:
            btn = page.locator(sel).first
            if btn.count() > 0 and btn.is_visible(timeout=1000):
                btn.click(timeout=2000)
                time.sleep(0.3)
        except Exception:
            pass  # popup absent, on continue sans bloquer


def normalize_distance_to_km(distance_val, distance_unit):
    """
    NOUVEAU : convertit systématiquement la distance en km au moment du scraping,
    au lieu de stocker valeur + unité séparément. Corrige à la source le bug
    (404 lignes en mètres traitées comme des km) découvert plus tard dans le notebook.
    """
    if distance_val is None or distance_unit is None:
        return None
    unit = distance_unit.lower()
    if unit in ("km",):
        return round(distance_val, 3)
    if unit in ("m", "meters", "meter"):
        return round(distance_val / 1000.0, 3)
    if unit in ("mile", "miles"):
        return round(distance_val * 1.60934, 3)
    return None


def parse_price_block(price_el):
    """
    NOUVEAU : remplace l'extraction "tous les chiffres concaténés" par une lecture
    structurée. Si le bloc contient un prix barré (original) ET un prix final,
    l'ancienne version ('.join sur tous les digits) les fusionnait en un seul
    nombre absurde (ex: "US$120" + "US$95" -> 12095). Ici on isole chaque montant
    via une regex sur des groupes de chiffres consécutifs, et on garde:
      - le dernier montant trouvé comme prix final (typiquement affiché en dernier)
      - le premier comme prix original s'il y a plus d'un montant (donc une promo)
    """
    if price_el is None:
        return None, None, None

    price_text = price_el.get_text(separator=' ', strip=True)
    # Cherche des montants du type "1,234" ou "150" (on ignore les décimales de centimes)
    amounts = re.findall(r'\d[\d,]*', price_text)
    amounts = [int(a.replace(',', '')) for a in amounts if a.replace(',', '').isdigit()]

    if not amounts:
        return None, None, None

    if len(amounts) == 1:
        return amounts[0], None, None

    # Plus d'un montant = prix barré + prix réduit. Le prix final est le plus petit
    # des deux (une réduction diminue toujours le prix), pas juste "le dernier".
    original = max(amounts)
    final = min(amounts)
    discount_pct = round((1 - final / original) * 100, 1) if original > 0 else None
    return final, original, discount_pct


def parse_property_card(card, city):
    """
    Parses a BeautifulSoup property card element and extracts hotel features.
    Filters out 'trash data' (listings without names, prices, ratings, or review counts).
    """
    try:
        # 1. Hotel Name
        title_el = card.find(attrs={"data-testid": "title"})
        if not title_el:
            return None
        hotel_name = title_el.get_text(strip=True)

        # 2. Hotel Link
        link_el = card.find(attrs={"data-testid": "title-link"})
        url = ""
        if link_el and link_el.has_attr('href'):
            url = link_el['href']
            if url.startswith('/'):
                url = "https://www.booking.com" + url
            if '?' in url:
                url = url.split('?')[0]

        # 3. Location / Neighborhood
        loc_el = card.find(attrs={"data-testid": "address"}) or card.find(attrs={"data-testid": "location"}) or card.find(attrs={"data-testid": "address-link"})
        location = loc_el.get_text(strip=True) if loc_el else ""

        # 4. Distance from Center -> normalisée en km directement (NOUVEAU)
        dist_el = card.find(attrs={"data-testid": "distance"})
        distance_km = None
        if dist_el:
            dist_text = dist_el.get_text(strip=True).lower()
            match = re.search(r'([\d.,]+)\s*(miles|mile|km|meters|meter|m)\b', dist_text)
            if match:
                raw_val = float(match.group(1).replace(',', ''))
                raw_unit = match.group(2)
                distance_km = normalize_distance_to_km(raw_val, raw_unit)

        # 5. Price -> extraction corrigée (NOUVEAU), + original price + discount %
        price_el = card.find(attrs={"data-testid": "price-and-discounted-price"})
        price_usd, original_price_usd, discount_pct = parse_price_block(price_el)

        # 6 & 7. Rating Score & Review Count
        rating_score = None
        review_count = None
        review_score_el = card.find(attrs={"data-testid": "review-score"})
        if review_score_el:
            review_text = review_score_el.get_text(separator=' ')
            score_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', review_text)
            for m in score_matches:
                try:
                    val = float(m)
                    if 1.0 <= val <= 10.0:
                        rating_score = val
                        break
                except ValueError:
                    pass
            match = re.search(r'([\d,]+)\s*(?:reviews|review|external reviews)', review_text, re.IGNORECASE)
            if match:
                review_count = int(match.group(1).replace(',', ''))

        # 8. Star Rating (1-5 stars)
        star_rating = None
        stars_el = card.find(attrs={"data-testid": "rating-stars"}) or card.find(attrs={"data-testid": "rating-squares"})
        if stars_el:
            svgs = stars_el.find_all('svg')
            if svgs:
                star_rating = len(svgs) / 2

        # 9. Property type (NOUVEAU) -- selecteur best-effort, a verifier au premier run
        # Booking affiche parfois un badge "Hotel" / "Apartment" / "Resort" au-dessus du nom.
        property_type = None
        type_el = card.find(attrs={"data-testid": "property-type-badge"})
        if type_el:
            property_type = type_el.get_text(strip=True)

        # 10. Amenities (texte) + versions booléennes dédiées (NOUVEAU)
        card_text_lower = card.get_text(separator=' ').lower()
        amenities = []
        free_cancellation = 'free cancellation' in card_text_lower
        breakfast_included = 'breakfast included' in card_text_lower
        free_wifi = 'free wifi' in card_text_lower or 'free wi-fi' in card_text_lower
        no_prepayment = 'no prepayment' in card_text_lower or 'no deposit' in card_text_lower
        has_parking = 'parking' in card_text_lower
        has_pool = 'pool' in card_text_lower or 'swimming pool' in card_text_lower
        has_gym = 'fitness' in card_text_lower or 'gym' in card_text_lower

        if free_cancellation:
            amenities.append('Free Cancellation')
        if breakfast_included:
            amenities.append('Breakfast Included')
        if free_wifi:
            amenities.append('Free WiFi')
        if no_prepayment:
            amenities.append('No Prepayment')
        if has_parking:
            amenities.append('Parking')
        if has_pool:
            amenities.append('Pool')
        if has_gym:
            amenities.append('Gym')
        amenities_str = '; '.join(amenities)

        # 11. "Sold out" urgency signal (NOUVEAU) -- ex: "Booked X times today"
        booked_today_match = re.search(r'booked\s+(\d+)\s+times?\s+today', card_text_lower)
        booked_today_count = int(booked_today_match.group(1)) if booked_today_match else None

        # --- Trash Filter ---
        if not hotel_name or price_usd is None or rating_score is None or review_count is None:
            return None

        return {
            "hotel_name": hotel_name,
            "city": city,
            "region": CITY_REGION.get(city),
            "location": location,
            "property_type": property_type,
            "distance_from_center_km": distance_km,
            "price_usd": price_usd,
            "original_price_usd": original_price_usd,
            "discount_pct": discount_pct,
            "rating_score": rating_score,
            "review_count": review_count,
            "star_rating": star_rating,
            "free_cancellation": free_cancellation,
            "breakfast_included": breakfast_included,
            "free_wifi": free_wifi,
            "no_prepayment": no_prepayment,
            "has_parking": has_parking,
            "has_pool": has_pool,
            "has_gym": has_gym,
            "booked_today_count": booked_today_count,
            "amenities": amenities_str,
            "url": url
        }
    except Exception as e:
        print(f"Error parsing card: {e}")
        return None


def save_incremental(new_records, filepath):
    """
    Appends new records to the CSV dataset, removes duplicates, and saves it.
    """
    if not new_records:
        return 0

    df_new = pd.DataFrame(new_records)
    if os.path.exists(filepath):
        try:
            df_old = pd.read_csv(filepath)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.drop_duplicates(subset=["url", "hotel_name"], inplace=True)
    df_combined.to_csv(filepath, index=False, encoding='utf-8')
    return len(df_combined)


def run_scraper():
    today = datetime.date.today()
    checkin = (today + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    checkout = (today + datetime.timedelta(days=31)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("   BOOKING.COM DATASET GENERATOR FOR MACHINE LEARNING")
    print("=" * 60)
    print(f"Check-in date: {checkin}")
    print(f"Check-out date: {checkout}")
    print(f"Targeting: At least 1000 clean entries")
    print(f"Output File: {os.path.abspath(OUTPUT_FILE)}")
    print("-" * 60)

    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
            print("Reset old dataset file.")
        except Exception as e:
            print(f"Could not remove old file: {e}")

    total_clean_count = 0
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

    with sync_playwright() as p:
        print("Launching Playwright Chromium browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1280, "height": 800},
            locale="en-US"
        )
        page = context.new_page()

        for city_idx, city in enumerate(CITIES):
            print(f"\n[{city_idx + 1}/{len(CITIES)}] Starting city: {city.upper()}")
            city_records = []

            url = f"https://www.booking.com/searchresults.html?ss={city}&selected_currency=USD&checkin={checkin}&checkout={checkout}"
            print(f"  Loading city search page...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                dismiss_popups(page)  # NOUVEAU : ferme cookies / sign-in popup avant de parser
            except Exception as e:
                print(f"  Failed to load search results for {city}: {e}")
                continue

            for page_num in range(PAGES_PER_CITY):
                print(f"  -> Page {page_num + 1}/{PAGES_PER_CITY} | Parsing...", end="", flush=True)

                # NOUVEAU : retry léger si le sélecteur des cartes timeout une première fois
                # (Booking peut afficher un interstitiel/captcha ponctuel plutôt qu'un vrai
                # manque de résultats -- un seul essai supplémentaire évite de perdre la ville entière)
                cards = []
                for attempt in range(2):
                    try:
                        page.wait_for_selector('[data-testid="property-card"]', timeout=15000)
                        scroll_down(page)
                        soup = BeautifulSoup(page.content(), 'html.parser')
                        cards = soup.find_all(attrs={"data-testid": "property-card"})
                        break
                    except Exception as e:
                        if attempt == 0:
                            print(f" (retry after: {str(e)[:30]})", end="", flush=True)
                            time.sleep(2.0)
                        else:
                            print(f" Failed: {str(e)[:40]}...")

                print(f" Found {len(cards)} items...", end="", flush=True)

                page_clean_records = []
                for card in cards:
                    record = parse_property_card(card, city)
                    if record:
                        page_clean_records.append(record)

                city_records.extend(page_clean_records)
                print(f" Parsed {len(page_clean_records)} clean properties.")

                if page_num == PAGES_PER_CITY - 1:
                    break

                try:
                    next_btn = page.locator('button[aria-label="Next page"], a[aria-label="Next page"]').first
                    if next_btn and next_btn.count() > 0:
                        next_btn.scroll_into_view_if_needed(timeout=5000)
                        time.sleep(1.0)
                        next_btn.click(force=True)
                        page.wait_for_load_state("networkidle", timeout=15000)
                        time.sleep(random.uniform(2.0, 3.5))
                    else:
                        break
                except Exception:
                    break

            total_clean_count = save_incremental(city_records, OUTPUT_FILE)
            print(f"  => City {city} completed. Total clean entries in dataset: {total_clean_count}")

            # NOUVEAU : petite pause aléatoire entre villes pour reduire le risque de
            # détection anti-bot / captcha silencieux sur de longues sessions
            time.sleep(random.uniform(1.5, 3.0))

            if total_clean_count >= MAX_TOTAL_ENTRIES:
                print(f"\nTarget achieved! {total_clean_count} clean entries collected.")
                break

        browser.close()

    print("\n" + "=" * 60)
    print("   SCRAPING PROCESS COMPLETE")
    print("=" * 60)
    print(f"Final Dataset Path: {os.path.abspath(OUTPUT_FILE)}")

    if os.path.exists(OUTPUT_FILE):
        final_df = pd.read_csv(OUTPUT_FILE)
        print(f"Total Rows Generated: {len(final_df)}")
        print("\nDataset columns and types:")
        print(final_df.dtypes)
        print("\nFirst 5 entries:")
        print(final_df[["hotel_name", "city", "region", "price_usd", "rating_score"]].head())
    else:
        print("Error: Dataset file was not generated.")


if __name__ == "__main__":
    run_scraper()