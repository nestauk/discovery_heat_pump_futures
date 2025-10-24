"""
Script to tag abstracts and patents with the categories defined in the taxonomy.

Usage:
```
python discovery_heat_pump_futures/pipeline/classify_innovations.py
```
"""

import pandas as pd
from discovery_utils.utils.llm import batch_check

import logging
from typing import Dict

from discovery_heat_pump_futures import PROJECT_DIR
from discovery_heat_pump_futures.getters.taxonomy import get_taxonomy

INPUT_FILE = PROJECT_DIR / "inputs/heat_pump_relevant_data.csv"
INTERIM_OUTPUT_FILE = PROJECT_DIR / "inputs/heat_pumps_w_categories.jsonl"
OUTPUT_FILE = PROJECT_DIR / "inputs/heat_pump_data_w_categories.csv"

SYSTEM_MESSAGE = """
    You are a sustainable-heating technology analyst evaluating heat pump innovations.

    Extract structured information about heat pump technologies from the provided text.

    Unless requested otherwise, adhere as precisely as possible to the language and text that is used in the provided text document.

    If the requested information is not described, return N/A. DO NOT make up any false information or false inferences.
"""

def df_to_nested_dict(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Converts the taxonomy dataframe into a nested dictionary structure.

    Each top-level key is a category, and each sub-key is a subcategory whose value is the explanation.

    The taxonomy with columns 'Category', 'Subcategory' and 'Subcategory explanation'
    ends up looking like this:

    ```
    {Category: {Subcategory1: Subcategory explanation,
                Subcategory2: Subcategory explanation,
                ...},
     Category2: {Subcategory1: Subcategory explanation,
                 Subcategory2: Subcategory explanation,
                 ...},
    }
    ```

    Args:
        df (pd.DataFrame): A DataFrame with columns 'Category', 'Subcategory', and
            'Subcategory explanation'. All values are expected to be strings.

    Returns:
        Dict[str, Dict[str, str]]: A nested dictionary where the first level of keys are categories,
        the second level are subcategories, and the values are subcategory explanations.
    """
    nested_dict = {}
    for _, row in df.iterrows():
        category = row["Category"].strip()
        subcategory = row["Subcategory"].strip().strip('"')
        explanation = row["Subcategory explanation"].strip()

        if category not in nested_dict:
            nested_dict[category] = {}
        nested_dict[category][subcategory] = explanation
    return nested_dict


def escape_braces(s: str) -> str:
    """Escape braces so that we can paste a dict into the prompt"""
    return s.replace("{", "{{").replace("}", "}}")


if __name__ == "__main__":
    heat_pump_df = pd.read_csv(INPUT_FILE)

    ids = heat_pump_df["id"].tolist()
    text = heat_pump_df["title_abstract"].tolist()
    text_data = dict(zip(ids, text, strict=True))
    logging.info("Number of texts to classify: %d", len(text_data))

    taxonomy_df = get_taxonomy().reset_index(drop=True)

    category_dict = df_to_nested_dict(taxonomy_df)

    fields = [
        # Summary
        {
            "name": "summary",
            "type": "str",
            "description": "A brief summary (≤25 words) of the innovation or technology described.",
        },
        # Application context
        {
        "name": "application_type",
        "type": "str",
        "description": """Identify the primary application scale based on context and capacity:

        'domestic': Residential/household applications, typically <20kW, single-family homes, apartments, small residential buildings.
        
        'industrial': Commercial/industrial applications, typically >100kW, office buildings, factories, district heating, process heat, large multi-family buildings.
        
        'both': Explicitly mentions multiple scales or scalable across domestic and industrial.
        
        'unclear': Insufficient information to determine scale.
        
        Consider both stated capacity (kW) and application context (building type, use case)."""
        },
        # Heat pump type (ASHP vs GSHP)
        {
            "name": "heat_pump_type",
            "type": "str",
            "description": """Identify the type of heat pump based on the heat source:

            'ASHP': Air-source heat pump - Uses external/ambient air as heat source. Indicators include:
            - Mentions of 'air source', 'air-source', 'ASHP'
            - References to outdoor air, ambient air, external air
            - Air-to-air or air-to-water systems
            - Outdoor units, fans for air circulation
            
            'GSHP': Ground-source heat pump - Uses underground heat sources. Indicators include:
            - Mentions of 'ground source', 'ground-source', 'GSHP', 'geothermal'
            - References to underground, buried pipes/tubes, boreholes
            - Open-loop or closed-loop ground systems
            - Groundwater, aquifer, or soil as heat source
            - Horizontal or vertical ground heat exchangers
            
            'both': Document explicitly discusses both ASHP and GSHP technologies
            
            'unclear': Insufficient information to determine the heat source type
            
            Note: Focus on the primary heat source, not the distribution method."""
        },
        #Specific application
        {
            "name": "specific_applications",
            "type": "list[str]",
            "description": "Specific use cases mentioned (e.g., 'space heating', 'water heating', 'process heat', 'district heating').",
        },
        # Traditional components
        {
            "name": "trad_components",
            "type": "list[str]",
            "description": f"Identify which component(s) of the heat pump, if any, are relevant to the text. This could be one or more of the following: {escape_braces(str(category_dict['Traditional Components']))}. Return an empty list if none apply.",
        },
        # Non traditional technologies
        {
            "name": "non_trad_technologies",
            "type": "list[str]",
            "description": f"Identify which non-traditional technology/technologies is relevant to the text. This could be one or more of the following: {escape_braces(str(category_dict['Non traditional technologies']))}. Return an empty list if none apply.",
        },
        # System design
        {
            "name": "system_design",
            "type": "list[str]",
            "description": f"Identify which system design aspect(s) is relevant to the text. This could be one or more of the following: {escape_braces(str(category_dict['System Design']))}. Return an empty list if none apply.",
        },
        # Other improvements
        {
            "name": "other_improvements",
            "type": "list[str]",
            "description": f"Identify which other improvements aspect(s) is relevant to the text. This could be one or more of the following: {escape_braces(str(category_dict['Other Improvements']))}. Return an empty list if none apply.",
        },
        # Circular economy
        {
            "name": "circular_economy",
            "type": "list[str]",
            "description": f"Identify which principle(s) of the circular economy are relevant to the text. This could be one or more of the following: {escape_braces(str(category_dict['Circular economy']))}. Return an empty list if none apply.",
        },
        #TRL
        {
    "name": "technology_readiness_level",
    "type": "str", 
    "description": """Classify the innovation's development phase based on Technology Readiness Level (TRL) indicators.

    IMPORTANT: The Technology Readiness Level scale ranges from 1-9, which we group into three phases:
    - TRL 1-3 = Research phase
    - TRL 4-6 = Development phase  
    - TRL 7-9 = Deployment phase

    Return ONLY one of: 'Research phase', 'Development phase', 'Deployment phase', or 'unclear'

    RESEARCH PHASE (equivalent to TRL 1-3):
    - Basic principles, theoretical concepts, or fundamental research
    - Computer simulations, mathematical models, or analytical studies
    - Laboratory experiments to validate basic concepts
    - Keywords: theoretical, conceptual, simulation, modeling, feasibility study, basic research, fundamental
    - NO physical prototypes or real-world testing

    DEVELOPMENT PHASE (equivalent to TRL 4-6):  
    - Component or prototype development and testing
    - Laboratory or controlled environment validation
    - System/subsystem model or prototype demonstration
    - Keywords: prototype, test rig, experimental setup, laboratory testing, validation, proof-of-concept, pilot scale, bench scale
    - Testing is LIMITED to controlled/laboratory environments

    DEPLOYMENT PHASE (equivalent to TRL 7-9):
    - System prototype demonstration in operational environment
    - Actual system completed and qualified through testing
    - Commercial products, field installations, or market-ready technologies
    - Keywords: commercial, product, field test, demonstration project, installed system, operational, market-ready, real-world application, customer site
    - Real-world testing, deployment, or commercial availability

    Decision rules:
    - If multiple phases are evident → classify based on the HIGHEST phase achieved
    - Patents describing commercial products → 'Deployment phase'
    - Only simulations without prototypes → 'Research phase'
    - Prototype testing in labs only → 'Development phase'
    - Field testing or commercial use → 'Deployment phase'
    - Insufficient information → 'unclear'"""
},
{
    "name": "trl_reasoning",
    "type": "str",
    "description": """Provide concise step by step reasoning as to why you classified this innovation into the chosen development phase. 
    Reference specific keywords or phrases from the text that support your classification."""
}
    ]

    logging.info("Fields for extraction: %s")
    for field in fields:
        logging.info(" - %s: %s", field["name"], field["description"])

    processor = batch_check.LLMProcessor(
        model_name="gpt-4o-mini",
        temperature=0,
        output_path=str(INTERIM_OUTPUT_FILE),
        system_message=SYSTEM_MESSAGE,
        session_name="classify_heat_pumps",
        output_fields=fields,
    )

    processor.run(text_data, batch_size=30, sleep_time=0.5)

    classified_data = pd.read_json(INTERIM_OUTPUT_FILE, lines=True)

    output_data = pd.merge(
        heat_pump_df,
        classified_data[
            [
                "id",
                "application_type",
                "heat_pump_type",
                "specific_applications",
                "trad_components",
                "non_trad_technologies",
                "other_improvements",
                "system_design",
                "circular_economy",
                "technology_readiness_level",
                "trl_reasoning",
            ]
        ],
        on="id",
        how="left",
    )

    output_data.to_csv(OUTPUT_FILE, index=False)
