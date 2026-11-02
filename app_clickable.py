from pathlib import Path
import re
import difflib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KFC Food Finder", page_icon="🍗", layout="wide")

# Look for the menu CSV in common locations in the GitHub repository.
BASE_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    BASE_DIR / "nutrition-booklet.csv",
    BASE_DIR / "main" / "nutrition-booklet.csv",
    BASE_DIR / "data" / "nutrition-booklet.csv",
]


def find_csv():
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    return None


def clean_number(value):
    """Convert values such as '12.5 g' into numbers for comparisons."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def clean_text(value):
    if pd.isna(value):
        return "Not listed"
    value = str(value).strip()
    return value if value and value.lower() not in {"nan", "none", "unknown"} else "Not listed"


@st.cache_data
def load_menu(csv_path, modified_time):
    # The supplied notebook's dataset may have two initial rows containing units/metadata.
    raw = pd.read_csv(csv_path)
    raw = raw.dropna(axis=1, how="all")
    product_col = next(
        (c for c in raw.columns if c.strip().lower() in {"products", "product", "food product"}),
        None,
    )
    if product_col is None:
        raise ValueError("The CSV needs a 'Products' column.")

    if len(raw) > 2:
        first_values = raw[product_col].astype(str).str.lower().tolist()[:2]
        if any(x in {"product", "products", "nan", "g", "kcal"} for x in first_values):
            raw = raw.iloc[2:].copy()

    raw = raw.dropna(subset=[product_col]).copy()
    raw[product_col] = raw[product_col].astype(str).str.strip()
    raw = raw[raw[product_col] != ""]
    raw = raw.drop_duplicates(subset=[product_col]).reset_index(drop=True)
    return raw, product_col


def find_column(df, *needles):
    for col in df.columns:
        normalized = re.sub(r"\s+", " ", str(col)).strip().lower()
        if any(needle in normalized for needle in needles):
            return col
    return None


def similar_items(df, product_col, selected_row, limit=5):
    # Prefer foods with similar nutrition values where available.
    numeric_cols = []
    for col in df.columns:
        if col == product_col:
            continue
        name = str(col).lower()
        if any(word in name for word in ["energy", "calorie", "protein", "carb", "fat", "sugar", "sodium"]):
            values = df[col].map(clean_number)
            if values.notna().sum() >= 3:
                numeric_cols.append(col)

    if numeric_cols:
        matrix = pd.DataFrame({c: df[c].map(clean_number) for c in numeric_cols})
        matrix = matrix.fillna(matrix.median())
        ranges = (matrix.max() - matrix.min()).replace(0, 1).fillna(1)
        normalized = (matrix - matrix.min()) / ranges
        distances = np.sqrt(((normalized - normalized.loc[selected_row.name]) ** 2).mean(axis=1))
        distances.loc[selected_row.name] = np.inf
        indices = distances.nsmallest(limit).index
        return df.loc[indices, product_col].tolist()

    # Fallback: find products with similar names.
    names = df[product_col].tolist()
    return [
        n for n in difflib.get_close_matches(selected_row[product_col], names, n=limit + 1, cutoff=0.0)
        if n != selected_row[product_col]
    ][:limit]


def choose_product(product_name):
    """Callback: select a suggestion before Streamlit reruns the page."""
    st.session_state["product_search"] = product_name
    st.session_state["selected_product"] = product_name
    st.session_state["suggested_products"] = []
    st.session_state["show_result"] = True
    st.session_state["search_error"] = ""


st.title("🍗 KFC Food Finder")
st.write(
    "Search a menu item by name to see the information available in the menu dataset "
    "and discover similar foods."
)

csv_path = find_csv()
if csv_path is None:
    st.error("Menu dataset not found. Upload `nutrition-booklet.csv` to your GitHub repository.")
    st.markdown(
        "Place it beside `app.py` (recommended), or inside a `data/` folder. "
        "The app needs this CSV because the model file alone does not contain ingredient lists or the full product menu."
    )
    st.stop()

try:
    menu, product_col = load_menu(str(csv_path), csv_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read the menu dataset: {exc}")
    st.stop()

if menu.empty:
    st.warning("No food products were found in the CSV.")
    st.stop()

# Identify columns that may contain ingredients and related food information.
ingredients_col = find_column(menu, "ingredient")
allergen_col = find_column(menu, "allergen")
msg_col = find_column(menu, "msg")
caffeine_col = find_column(menu, "caffeine", "artificial sweet")
energy_col = find_column(menu, "total energy", "energy", "calorie")
protein_col = find_column(menu, "protein")
carb_col = find_column(menu, "carbohydrate", "carb")
fat_col = find_column(menu, "total fat", "fat")
sugar_col = find_column(menu, "sugar")
sodium_col = find_column(menu, "sodium")

product_names = menu[product_col].tolist()

# Initialise persistent page state.
for key, default in {
    "product_search": "",
    "selected_product": "",
    "suggested_products": [],
    "show_result": False,
    "search_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.text_input(
    "Enter product name",
    placeholder="e.g. Zinger Burger",
    key="product_search",
    help="Type the food or drink name as it appears on the menu.",
)

if st.button("Find food", type="primary", use_container_width=True):
    typed_query = st.session_state["product_search"].strip()
    st.session_state["search_error"] = ""
    st.session_state["suggested_products"] = []
    st.session_state["show_result"] = False
    st.session_state["selected_product"] = ""

    if not typed_query:
        st.session_state["search_error"] = "Please enter a food or drink name first."
    else:
        normalized_names = {name.casefold(): name for name in product_names}
        exact_match = normalized_names.get(typed_query.casefold())

        if exact_match:
            st.session_state["selected_product"] = exact_match
            st.session_state["show_result"] = True
        else:
            matches = difflib.get_close_matches(typed_query, product_names, n=5, cutoff=0.30)
            if matches:
                st.session_state["suggested_products"] = matches
            else:
                st.session_state["search_error"] = (
                    "No close menu match found. Try a shorter name or check the spelling."
                )

if st.session_state["search_error"]:
    st.warning(st.session_state["search_error"])

# Every suggestion is a real button. Clicking it selects that item and opens its details.
if st.session_state["suggested_products"]:
    st.info("I couldn't find an exact match. Click one of these foods or drinks to select it:")
    suggestion_cols = st.columns(2)
    for index, item in enumerate(st.session_state["suggested_products"]):
        with suggestion_cols[index % 2]:
            st.button(
                f"🍽️  {item}",
                key=f"suggestion_{index}_{item}",
                use_container_width=True,
                on_click=choose_product,
                args=(item,),
            )

if st.session_state["show_result"] and st.session_state["selected_product"]:
    selected_matches = menu[menu[product_col] == st.session_state["selected_product"]]
    if not selected_matches.empty:
        selected = selected_matches.iloc[0]
        st.success(f"Selected: {selected[product_col]}")
        st.subheader(f"🥡 {selected[product_col]}")

        # Show an ingredients list only when the source dataset actually has one.
        # The nutrition/allergen "Menu information" metrics section has been removed.
        if ingredients_col:
            st.markdown("**Ingredients**")
            st.write(clean_text(selected[ingredients_col]))

        st.subheader("You may also like")
        recommendations = similar_items(menu, product_col, selected, limit=5)
        if recommendations:
            st.caption("Click a recommendation to view its details:")
            rec_cols = st.columns(2)
            for index, item in enumerate(recommendations):
                with rec_cols[index % 2]:
                    st.button(
                        f"🍴  {item}",
                        key=f"recommendation_{index}_{item}",
                        use_container_width=True,
                        on_click=choose_product,
                        args=(item,),
                    )
        else:
            st.write("There are no other menu items available to compare.")

st.divider()
st.caption(
    "Information is limited to the uploaded menu dataset. For allergies, always verify the latest "
    "official ingredient and allergen information with the restaurant."
)
