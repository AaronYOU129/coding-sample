"""
Purpose:    Scrape marketing data for prebiotic-soda products from three brand
            websites (Olipop, Poppi, Coca-Cola Simply Pop), capturing both the
            "list page" view (name, price, position) and the "product page" view
            (photos, flavor, ingredients, nutrition).

            Methodology choices and *why*:
            - Position is taken from the DOM order of product cards on the live
              list page, because "position" is a property of how the page renders
              its grid, not of any catalog API ordering.
            - For the two Shopify stores (Olipop, Poppi) prices and photo counts
              come from each product's Shopify `/products/<handle>.js` endpoint
              keyed by the exact handle scraped from the list page. The list-page
              cards do not render a price, so the authoritative per-variant price
              is the only correct source; reading it from the API avoids brittle
              price-string scraping while staying tied to the scraped handle.
            - Ingredients and nutrition live in theme-specific metafields that the
              `.js`/`.json` APIs do not expose, so they are parsed from the
              product-page HTML with small per-pattern extractors.
            - Coca-Cola's page is one Adobe-AEM document that is both the list page
              and the product page (the assignment gives the same URL for both).
              It contains five flavor sections rendered server-side, so all data
              is read from that single document; the site never sells direct, so
              price is intentionally blank.

Inputs:     None. All data is fetched live over HTTPS from:
              - https://drinkolipop.com/                       (list page)
              - https://drinkpoppi.com/collections/drinks       (list page)
              - https://www.coca-cola.com/.../simply/products/pop (list + product)
              plus each Olipop/Poppi product page and `/products/<handle>.js`.

Outputs:    output/task1_products.csv  — one row per (product x page-type) with
            columns: URL, list/product page, product name, price,
            discounted price, position, number of photos, flavor, ingredients,
            nutrition facts.

Key Steps:  Data        -> fetch list-page HTML for each site
            Processing  -> read product order/handles from the list-page DOM
                        -> enrich each product (price/photos via API, then
                           ingredients/nutrition from the product-page HTML)
            Output      -> explode each product into a list-page row and a
                           product-page row and write the CSV.

How to Run: python3 task1_webscraping/scrape_task1.py
            (requires: requests, beautifulsoup4)
"""

import csv
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

OUTPUT_CSV = Path(__file__).parent / "output" / "task1_products.csv"
MAX_PRODUCTS_PER_SITE = 10
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

CSV_COLUMNS = [
    "URL",
    "list/product page",
    "product name",
    "price",
    "discounted price",
    "position",
    "number of photos",
    "flavor",
    "ingredients",
    "nutrition facts",
]


@dataclass
class Product:
    """One product with both its list-page facts and its product-page facts."""

    list_url: str
    product_url: str
    name: str
    position: int
    price: str = ""
    discounted_price: str = ""
    number_of_photos: str = ""
    flavor: str = ""
    ingredients: str = ""
    nutrition: str = ""


@dataclass
class SiteConfig:
    """A site adapter: how to find its products and how to enrich each one."""

    name: str
    list_url: str
    collect: Callable[["SiteConfig"], list]


# ----------------------------------------------------------------------------
# Main workflow (newspaper structure: top-level intent first, details below)
# ----------------------------------------------------------------------------

def main() -> None:
    sites = [olipop_site(), poppi_site(), cocacola_site()]
    products = collect_all_products(sites)
    rows = [row for product in products for row in product_to_rows(product)]
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote {len(rows)} rows for {len(products)} products to {OUTPUT_CSV}")


def collect_all_products(sites: list) -> list:
    products = []
    for site in sites:
        print(f"[{site.name}] scraping list page: {site.list_url}")
        site_products = site.collect(site)
        print(f"[{site.name}] collected {len(site_products)} products")
        products.extend(site_products)
    return products


def product_to_rows(product: Product) -> list:
    """Each product becomes a list-page row and a product-page row.

    The two pages carry different fields per the assignment, so each row leaves
    the other page's fields blank rather than duplicating data that page did
    not actually show.
    """
    list_row = {
        "URL": product.list_url,
        "list/product page": "list page",
        "product name": product.name,
        "price": product.price,
        "discounted price": product.discounted_price,
        "position": product.position,
        "number of photos": "",
        "flavor": "",
        "ingredients": "",
        "nutrition facts": "",
    }
    product_row = {
        "URL": product.product_url,
        "list/product page": "product page",
        "product name": product.name,
        "price": "",
        "discounted price": "",
        "position": "",
        "number of photos": product.number_of_photos,
        "flavor": product.flavor,
        "ingredients": product.ingredients,
        "nutrition facts": product.nutrition,
    }
    return [list_row, product_row]


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# Shared HTTP + parsing helpers
# ----------------------------------------------------------------------------

def fetch_text(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    # Some origins omit charset; trust the document's own UTF-8 bytes rather
    # than requests' ISO-8859-1 fallback (otherwise '®' arrives as 'Â®').
    response.encoding = response.apparent_encoding or "utf-8"
    time.sleep(0.5)  # be polite to the origin between requests
    return response.text


def fetch_json(url: str) -> dict:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(0.5)
    return response.json()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# ----------------------------------------------------------------------------
# Shopify sites (Olipop, Poppi): same pipeline, different list-card selector
# ----------------------------------------------------------------------------

@dataclass
class ShopifySite(SiteConfig):
    domain: str = ""
    list_card_selector: str = ""


def olipop_site() -> ShopifySite:
    return ShopifySite(
        name="Olipop",
        list_url=(
            "https://drinkolipop.com/?srsltid="
            "AfmBOopuvIIZsTR44Pp8FsU2dEIrn3jBeRAJ4nSR-RUDZ1r_9eGjjPFn"
        ),
        domain="https://drinkolipop.com",
        # Homepage flavor grid; nav/menu cards use different class tokens.
        list_card_selector=".product-card.tw-grid",
        collect=collect_shopify_products,
    )


def poppi_site() -> ShopifySite:
    return ShopifySite(
        name="Poppi",
        list_url="https://drinkpoppi.com/collections/drinks",
        domain="https://drinkpoppi.com",
        list_card_selector="[data-product]",
        collect=collect_shopify_products,
    )


def collect_shopify_products(site: ShopifySite) -> list:
    handles = read_list_page_handles(site)
    products = []
    for position, handle in enumerate(handles, start=1):
        product_url = f"{site.domain}/products/{handle}"
        catalog = fetch_json(f"{product_url}.js")
        page_html = fetch_text(product_url)
        products.append(
            build_shopify_product(site, handle, position, product_url, catalog, page_html)
        )
        print(f"  [{site.name}] {position:>2}. {catalog.get('title', handle)}")
    return products


def read_list_page_handles(site: ShopifySite) -> list:
    """Return the first N product handles in the order the list page renders them."""
    soup = soup_of(fetch_text(site.list_url))
    handles = []
    for card in soup.select(site.list_card_selector):
        handle = product_handle_from_card(card)
        if handle and handle not in handles:
            handles.append(handle)
        if len(handles) >= MAX_PRODUCTS_PER_SITE:
            break
    return handles


def product_handle_from_card(card) -> Optional[str]:
    link = card.select_one('a[href*="/products/"]')
    if not link:
        return None
    match = re.search(r"/products/([^/?#]+)", link.get("href", ""))
    return match.group(1) if match else None


def build_shopify_product(site, handle, position, product_url, catalog, page_html) -> Product:
    price, discounted = shopify_prices(catalog)
    return Product(
        list_url=site.list_url,
        product_url=product_url,
        name=catalog.get("title", handle),
        position=position,
        price=price,
        discounted_price=discounted,
        number_of_photos=str(len(catalog.get("images", []))),
        flavor=catalog.get("title", handle),
        ingredients=extract_shopify_ingredients(page_html),
        nutrition=extract_shopify_nutrition(soup_of(page_html)),
    )


def shopify_prices(catalog: dict) -> tuple:
    """Map Shopify cents to (regular price, discounted price).

    Shopify stores the live selling price in `price` and the pre-sale price in
    `compare_at_price`. When an item is on sale we report the original as the
    regular price and the live price as the discount; otherwise the discount
    column stays blank.
    """
    price_cents = catalog.get("price")
    compare_cents = catalog.get("compare_at_price")
    if compare_cents and compare_cents > price_cents:
        return dollars(compare_cents), dollars(price_cents)
    return dollars(price_cents), ""


def dollars(cents: Optional[int]) -> str:
    if cents is None:
        return ""
    return f"${cents / 100:.2f}"


def extract_shopify_ingredients(html: str) -> str:
    """Capture the ingredient list that follows an 'Ingredients:' label.

    The colon disambiguates the real product ingredient list from navigation
    links such as 'Ingredients' or headings like 'Ingredients You Can Love'.
    """
    marker = re.search(r"Ingredients:\s*", html, re.IGNORECASE)
    if not marker:
        return ""
    text = unescape(re.sub(r"<[^>]+>", " ", html[marker.end():]))
    text = clean(text).lstrip(" .:;,-")  # drop leftover label punctuation/&nbsp;
    for stop_word in ("Contains", "Nutrition", "Storage", "Allergen"):
        cut = text.find(stop_word)
        if cut > 0:
            text = text[:cut]
    return clean(text)[:600]


def extract_shopify_nutrition(soup: BeautifulSoup) -> str:
    """Climb from the 'Serving Size' label to the smallest block holding the table."""
    anchor = soup.find(string=re.compile(r"Serving Size", re.IGNORECASE))
    if not anchor:
        return ""
    for ancestor in anchor.parents:
        text = clean(ancestor.get_text(" "))
        lowered = text.lower()
        if "calories" in lowered and "serving size" in lowered:
            return slice_from_nutrition_start(text)
    return ""


def slice_from_nutrition_start(text: str) -> str:
    lowered = text.lower()
    start = lowered.find("nutrition facts")
    if start < 0:
        start = lowered.find("serving size")
    return text[start:start + 600]


# ----------------------------------------------------------------------------
# Coca-Cola Simply Pop: one server-rendered page holding five flavor sections
# ----------------------------------------------------------------------------

COCACOLA_URL = "https://www.coca-cola.com/us/en/brands/simply/products/pop"


def cocacola_site() -> SiteConfig:
    return SiteConfig(
        name="Coca-Cola Simply Pop",
        list_url=COCACOLA_URL + "#shop",
        collect=collect_cocacola_products,
    )


def collect_cocacola_products(site: SiteConfig) -> list:
    soup = soup_of(fetch_text(COCACOLA_URL))
    names = cocacola_flavor_names(soup)
    ingredients = cocacola_ingredient_blocks(soup)
    nutrition = cocacola_nutrition_blocks(soup)
    image_srcs = cocacola_image_srcs(soup)

    products = []
    for position, name in enumerate(names[:MAX_PRODUCTS_PER_SITE], start=1):
        flavor = strip_flavor(name)
        products.append(
            Product(
                list_url=site.list_url,
                product_url=COCACOLA_URL,
                name=name,
                position=position,
                price="",  # Coca-Cola.com does not sell direct; no list-page price
                discounted_price="",
                number_of_photos=str(count_flavor_photos(flavor, image_srcs)),
                flavor=flavor,
                ingredients=nth(ingredients, position - 1),
                nutrition=nth(nutrition, position - 1),
            )
        )
        print(f"  [Coca-Cola] {position:>2}. {name}")
    return products


def cocacola_flavor_names(soup: BeautifulSoup) -> list:
    return [
        clean(heading.get_text())
        for heading in soup.find_all("h3")
        if "Prebiotic Soda" in heading.get_text()
    ]


def cocacola_ingredient_blocks(soup: BeautifulSoup) -> list:
    blocks = []
    for heading in soup.find_all("h5", string=re.compile("INGREDIENTS", re.IGNORECASE)):
        paragraph = heading.find_next("p")
        blocks.append(clean(paragraph.get_text()) if paragraph else "")
    return blocks


def cocacola_nutrition_blocks(soup: BeautifulSoup) -> list:
    return [clean(table.get_text(" "))[:600] for table in soup.select(".nutritional-information")]


def cocacola_image_srcs(soup: BeautifulSoup) -> list:
    return [img.get("src") or img.get("data-src") or "" for img in soup.find_all("img")]


def count_flavor_photos(flavor: str, image_srcs: list) -> int:
    """Count page images whose asset path references this flavor.

    The page is a single document shared by all flavors, so a flavor's photo
    count is the number of product assets naming that flavor (e.g. the can
    render plus any campaign shot).
    """
    token = flavor.split()[0].lower()
    return sum(1 for src in image_srcs if token in src.lower())


def strip_flavor(name: str) -> str:
    flavor = re.sub(r"Simply\W*Pop", "", name, flags=re.IGNORECASE)
    flavor = re.sub(r"Prebiotic Soda", "", flavor, flags=re.IGNORECASE)
    return clean(flavor)


def nth(items: list, index: int) -> str:
    return items[index] if 0 <= index < len(items) else ""


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        print(f"Network error while scraping: {error}", file=sys.stderr)
        sys.exit(1)
