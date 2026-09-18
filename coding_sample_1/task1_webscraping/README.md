# Task 1 — Web Scraping

Scrapes marketing data for prebiotic sodas from three brand sites, capturing the
**list-page** view (name, price, position) and the **product-page** view
(photos, flavor, ingredients, nutrition).

## How to run

```bash
pip install requests beautifulsoup4
python3 scrape_task1.py
```

Output: `output/task1_products.csv` (50 rows = 25 products × 2 page types).

## Sites & coverage

| Site | List page | Products captured |
|------|-----------|-------------------|
| Olipop | `drinkolipop.com` homepage flavor grid | first 10 |
| Poppi | `drinkpoppi.com/collections/drinks` | first 10 |
| Coca-Cola Simply Pop | `coca-cola.com/.../simply/products/pop` | all 5 flavors (fewer than 10 exist) |

## CSV columns

`URL, list/product page, product name, price, discounted price, position, number of photos, flavor, ingredients, nutrition facts`

Each product yields two rows: a `list page` row (name / price / discounted price
/ position) and a `product page` row (name / number of photos / flavor /
ingredients / nutrition facts). Fields a given page does not show are left blank
rather than duplicated, so every value in the file is something that page
actually presented.

## Methodology & why

- **Position** is read from the DOM order of product cards on the live list
  page — position is a property of how the page renders its grid, not of any
  catalog ordering.
- **Price / discounted price (Shopify: Olipop, Poppi).** The list-page cards do
  not render a price, so the authoritative source is each product's Shopify
  `/products/<handle>.js` endpoint, queried by the exact handle scraped from the
  list page. Shopify keeps the live selling price in `price` and the pre-sale
  price in `compare_at_price`; when an item is on sale the original is reported
  as *price* and the live price as *discounted price*. No item was on sale at
  scrape time, so the discount column is blank (accurate, not missing).
- **Number of photos (Shopify)** is the length of the product's media gallery
  from the same endpoint — the authoritative gallery count.
- **Ingredients / nutrition** live in theme-specific metafields that the JSON
  APIs do not expose, so they are parsed from the product-page HTML.
- **Coca-Cola** serves one Adobe-AEM document that is both the list page and the
  product page (the assignment gives the same URL for both). It renders five
  flavor sections server-side; all fields are read from that single document.
  The flavor sections, ingredient blocks, and nutrition tables appear in the
  same order and are matched by position.

## Assumptions

- Coca-Cola.com does not sell direct (it points to retailers), so Simply Pop has
  no on-page price; those price cells are intentionally blank.
- Coca-Cola "number of photos" counts the page's product images that reference a
  flavor (e.g. the can render plus any campaign shot), since all flavors share
  one page.
- "Flavor" is the product/flavor name (e.g. *Blackberry Vanilla*, *Raspberry*).

## Files

```
task1_webscraping/
├── scrape_task1.py            # the scraper
├── README.md                  # this file
└── output/task1_products.csv  # result
```
