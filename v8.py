import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
# print(pd.__version__)
# Set page config
st.set_page_config(page_title="NBCSP Cost-Effectiveness Dashboard", layout="wide")

#The password to access the data - could remove, not so important
PASSWORD = "111"  #"dohacneverscreeners"

# #Verify password before continuing
# def check_password():
#     password = st.text_input("Enter password", type="password")
#     if password != PASSWORD:
#         st.stop()
#
# # check_password()

st.sidebar.title("Login")
user_pass = st.sidebar.text_input("Enter password:", type="password")

# Only check password if user has typed something
if user_pass:
    if user_pass != PASSWORD:
        st.warning("Incorrect password. Please enter the correct password to access the app.")
        st.stop()
else:
    st.info("Please enter the password to access the app.")
    st.stop()

@st.cache_data
def load_data():
    """Load and process data from Excel file"""
    # Read main data
    annual_data = pd.read_excel("input_file_2025_09_12_annual.xlsx", 'inputs_annual')
    total_data = pd.read_excel("input_file_2025_09_12_annual.xlsx", 'inputs_total')

    # Read cost inputs
    cost_inputs = pd.read_excel("input_file_2025_09_12_annual.xlsx", 'Cost inputs',
                                skiprows=6, usecols="A:B", nrows=14)

    return annual_data, total_data, cost_inputs


# Use cached data
annual_df, total_df, cost_inputs = load_data()

list_cost = cost_inputs['Baseline cost*'].to_list()
# default item cost
postage = list_cost[0]  # Mailing the kit
cost_test_kit = list_cost[1]
cost_process_kit = list_cost[3]
cost_GP_consultation = list_cost[4]
cost_COL_no_complication = list_cost[5]
cost_COL_w_complication = list_cost[6]

prob_incorrectCompleteFOBT = list_cost[11]
# print(prob_incorrectCompleteFOBT)
COL_complication_rate = list_cost[12]
# print(COL_complication_rate)
cost_letter = list_cost[0]  # Mailing the letter, default same as mailing a kit

@st.cache_data
def get_outcome_data(df, outcome, year):
    """Extract data for a specific outcome from the first scenario found"""
    if df is None:
        return []

    # Generate year range from 2026 to end_year (inclusive)
    years = list(range(2026, year + 1))  # Keep as integers since your columns are integers

    # Filter for the specific outcome (use first scenario found)
    outcome_rows = df[df['Outcome'] == outcome]
    if not outcome_rows.empty:
        # Take the first row (first scenario with this outcome)
        row = outcome_rows.iloc[0]
        # Extract values for each year that exists in the dataframe
        return [float(row[yr]) for yr in years if yr in df.columns]

    return []

@st.cache_data
def interpolate_all_data(df, participation_multiplier):
    """
    Interpolate all outcomes for all years based on participation multiplier

    Args:
        df: DataFrame with columns [Scenario, Outcome, 2026, 2027, ..., 2099]
        participation_multiplier: Float between 0.5 and 1.5

    Returns:
        DataFrame with all outcomes interpolated for the given multiplier
    """
    if df is None:
        return None

    # Available scenarios mapping
    available_scenarios = {
        0.5: '0.5x',
        0.75: '0.75x',
        1.0: '1.0x',
        1.25: '1.25x',
        1.5: '1.5x'
    }

    year_range = list(range(start_year, lifetime_year + 1))
    year_columns = sorted([year for year in year_range if year in df.columns])
    # Get all unique outcomes
    outcomes = df['Outcome'].unique()

    # Comparator
    if participation_multiplier == 0:
        result_df = df[df['Scenario'] == '1.0x'].copy()
        result_df['Scenario'] = f'{participation_multiplier:.2f}x'

        # Reset all values in row 'persistent_never_screeners' to 0
        result_df.loc[result_df['Outcome'] == 'persistent_never_screeners', result_df.columns.difference(['Outcome'])] = 0
        return result_df

    # Check bounds
    min_mult = min(available_scenarios.keys())
    max_mult = max(available_scenarios.keys())

    if participation_multiplier < min_mult:
        print(f"Warning: Multiplier {participation_multiplier} below range. Using {min_mult}")
        participation_multiplier = min_mult
    elif participation_multiplier > max_mult:
        print(f"Warning: Multiplier {participation_multiplier} above range. Using {max_mult}")
        participation_multiplier = max_mult

    # If exact match, return original data with new scenario name
    if participation_multiplier in available_scenarios:
        scenario_name = available_scenarios[participation_multiplier]
        result_df = df[df['Scenario'] == scenario_name].copy()
        result_df['Scenario'] = f'{participation_multiplier:.2f}x'
        return result_df

    # Find bounding scenarios for interpolation
    multipliers = sorted(available_scenarios.keys())

    lower_mult = None
    upper_mult = None

    for i, mult in enumerate(multipliers):
        if mult <= participation_multiplier:
            lower_mult = mult
        if mult >= participation_multiplier and upper_mult is None:
            upper_mult = mult
            break

    # Safety check
    if lower_mult is None or upper_mult is None or lower_mult == upper_mult:
        # Fallback to 1.0x scenario
        result_df = df[df['Scenario'] == '1.0x'].copy()
        result_df['Scenario'] = f'{participation_multiplier:.2f}x'
        return result_df

    # Get scenario names
    lower_scenario = available_scenarios[lower_mult]
    upper_scenario = available_scenarios[upper_mult]

    # Calculate interpolation weight
    weight = (participation_multiplier - lower_mult) / (upper_mult - lower_mult)

    # Create interpolated data
    interpolated_rows = []

    for outcome in outcomes:
        # Get data for this outcome from both scenarios
        lower_row = df[(df['Scenario'] == lower_scenario) & (df['Outcome'] == outcome)]
        upper_row = df[(df['Scenario'] == upper_scenario) & (df['Outcome'] == outcome)]

        if not lower_row.empty and not upper_row.empty:
            # Create new row for this outcome
            new_row = {
                'Scenario': f'{participation_multiplier:.2f}x',
                'Outcome': outcome
            }

            # Interpolate each year
            for year in year_columns:
                if year in lower_row.columns and year in upper_row.columns:
                    lower_val = float(lower_row[year].iloc[0])
                    upper_val = float(upper_row[year].iloc[0])

                    # Linear interpolation
                    interpolated_val = lower_val + weight * (upper_val - lower_val)
                    new_row[year] = interpolated_val

            interpolated_rows.append(new_row)

    # Convert to DataFrame
    if interpolated_rows:
        result_df = pd.DataFrame(interpolated_rows)
        # Ensure columns are in the right order
        column_order = ['Scenario', 'Outcome'] + year_columns
        result_df = result_df[column_order]
        return result_df
    else:
        return None

# @st.cache_data
def add_cost_calculations(df):  #Calculate costs
    """calculate_custom_costs
    Add cost calculation rows to the interpolated dataframe using custom unit costs
    Uses global variables: unit_costs, prob_incorrectCompleteFOBT, COL_complication_rate, treatment_multiplier

    Args:
        df: DataFrame with format [Scenario, Outcome, 2026, 2027, ..., 2099]

    Returns:
        DataFrame with additional cost outcome rows appended
    """
    # Get year columns and scenario
    year_range = list(range(start_year, lifetime_year + 1))

    year_columns = [col for col in year_range if col in df.columns]
    scenario = df['Scenario'].iloc[0]

    # Get outcome data
    persistent_never_screeners = df[df['Outcome'] == 'persistent_never_screeners'].iloc[0]
    invitation_sent_total = df[df['Outcome'] == 'Invitation_sent_total'].iloc[0]
    kits_returned_total = df[df['Outcome'] == 'kits_returned_total'].iloc[0]
    colonoscopies_total = df[df['Outcome'] == 'colonoscopies_total'].iloc[0]
    treatment_cost = df[df['Outcome'] == 'treatment_cost'].iloc[0]

    # Create new cost rows
    new_rows = []

    # 1. letters_only_cost
    letters_only_row = {'Scenario': scenario, 'Outcome': 'letters_only_cost'}
    for year in year_columns:
        letters_only_row[year] = persistent_never_screeners[year] * unit_costs['letter']
    new_rows.append(letters_only_row)

    # 2. sending_fobt_cost
    sending_fobt_row = {'Scenario': scenario, 'Outcome': 'sending_fobt_cost'}
    for year in year_columns:
        fobt_invites = invitation_sent_total[year] - persistent_never_screeners[year]
        sending_fobt_row[year] = fobt_invites * (unit_costs['kit_postage'] + unit_costs['FOBT_kit'])
    new_rows.append(sending_fobt_row)

    # 3. process_fobt_cost
    process_fobt_row = {'Scenario': scenario, 'Outcome': 'process_fobt_cost'}
    base_cost = unit_costs['kit_postage'] + unit_costs['process_kit']
    incorrect_cost = prob_incorrectCompleteFOBT * (
                2 * unit_costs['kit_postage'] + unit_costs['FOBT_kit'] + unit_costs['process_kit'])
    total_process_cost = base_cost + incorrect_cost

    for year in year_columns:
        process_fobt_row[year] = kits_returned_total[year] * total_process_cost
    new_rows.append(process_fobt_row)

    # 4. col_cost
    col_cost_row = {'Scenario': scenario, 'Outcome': 'col_cost'}
    no_comp_cost = unit_costs['gp_consult'] + unit_costs['colonoscopy_no_comp']
    comp_cost = unit_costs['gp_consult'] + unit_costs['colonoscopy_comp']

    for year in year_columns:
        no_comp_total = colonoscopies_total[year] * (1 - COL_complication_rate) * no_comp_cost
        comp_total = colonoscopies_total[year] * COL_complication_rate * comp_cost
        col_cost_row[year] = no_comp_total + comp_total
    new_rows.append(col_cost_row)

    # 5. new_treatment_cost
    new_treatment_cost_row = {'Scenario': scenario, 'Outcome': 'new_treatment_cost'}
    for year in year_columns:
        new_treatment_cost_row[year] = treatment_cost[year] * treatment_cost_multiplier
    new_rows.append(new_treatment_cost_row)

    # Convert new rows to DataFrame for easier calculation of composite costs
    temp_df = pd.DataFrame(new_rows)

    # 6. screening_cost = letters_only_cost + sending_fobt_cost + process_fobt_cost
    screening_cost_row = {'Scenario': scenario, 'Outcome': 'screening_cost'}
    letters_only_data = temp_df[temp_df['Outcome'] == 'letters_only_cost'].iloc[0]
    sending_fobt_data = temp_df[temp_df['Outcome'] == 'sending_fobt_cost'].iloc[0]
    process_fobt_data = temp_df[temp_df['Outcome'] == 'process_fobt_cost'].iloc[0]

    for year in year_columns:
        screening_cost_row[year] = letters_only_data[year] + sending_fobt_data[year] + process_fobt_data[year]
    new_rows.append(screening_cost_row)

    # 7. total_cost = screening_cost + col_cost + new_treatment_cost
    total_cost_row = {'Scenario': scenario, 'Outcome': 'total_cost'}
    col_cost_data = temp_df[temp_df['Outcome'] == 'col_cost'].iloc[0]
    new_treatment_data = temp_df[temp_df['Outcome'] == 'new_treatment_cost'].iloc[0]

    for year in year_columns:
        total_cost_row[year] = screening_cost_row[year] + col_cost_data[year] + new_treatment_data[year]
    new_rows.append(total_cost_row)

    # Append new rows to original dataframe
    # new_df = pd.DataFrame(new_rows)
    # result_df = pd.concat([df, new_df], ignore_index=True)
    result_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True, copy=False)
    return result_df

@st.cache_data
def calculate_totals(df):
    """
    Calculate totals for all outcomes with predefined time periods

    Args:
        df: DataFrame output from add_cost_calculations

    Returns:
        DataFrame with 4 additional total columns:
        - total_2026_2030: Sum for years 2026-2030
        - total_2026_2045: Sum for years 2026-2045
        - total_2026_2099: Sum for years 2026-2099
        # - discounted_total: Discounted total 2026-2075 (5% discount rate)
    """
    # Make a copy to avoid modifying original
    result_df = df.copy()

    # Define the time periods
    periods = {
        # 'total_2026_2030': (2026, 2030),
        'total_2026_2045': (2026, 2045),
        'total_2026_2099': (2026, 2099)
    }

    # Calculate totals for each period (for all rows)
    for column_name, (start_year, end_year) in periods.items():
        year_range = list(range(start_year, end_year + 1))
        available_years = [year for year in year_range if year in df.columns]

        if available_years:
            for idx in result_df.index:
                total = sum(result_df.loc[idx, year] for year in available_years)
                result_df.loc[idx, column_name] = total
        else:
            print(f"Warning: No year columns found in range {start_year}-{end_year}")
            # Set to 0 if no years available
            result_df[column_name] = 0

    return result_df


@st.cache_data
def interpolate_total_data(df, participation_multiplier):
    """
    Interpolate lifetime data and return as dictionary

    Args:
        df: DataFrame with format [Scenario, outcome1, outcome2, ..., outcomeN]
        participation_multiplier: Float between 0.5 and 1.5

    Returns:
        Dictionary with interpolated values for all outcomes
    """
    if df is None or df.empty:
        return None

    # Available scenarios
    scenarios = {
        0.5: '0.5x',
        0.75: '0.75x',
        1.0: '1.0x',
        1.25: '1.25x',
        1.5: '1.5x'
    }

    # Comparator
    if participation_multiplier == 0:
        result_row = df[df['Scenario'] == '1.0x']
        result_dict = result_row.iloc[0].to_dict()
        result_dict['Scenario'] = f'{participation_multiplier:.2f}x'
        # Convert all numpy types to regular Python types
        result_dict = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                       for k, v in result_dict.items()}
        result_dict['Letters_only'] = 0
        result_dict['Letters_only_discounted'] = 0
        return result_dict

    # Clamp to valid range
    if participation_multiplier < 0.5:
        print(f"Warning: Multiplier {participation_multiplier} below 0.5. Using 0.5")
        participation_multiplier = 0.5
    elif participation_multiplier > 1.5:
        print(f"Warning: Multiplier {participation_multiplier} above 1.5. Using 1.5")
        participation_multiplier = 1.5

    # If exact match, return that scenario as dict
    if participation_multiplier in scenarios:
        scenario_name = scenarios[participation_multiplier]
        result_row = df[df['Scenario'] == scenario_name]
        if not result_row.empty:
            result_dict = result_row.iloc[0].to_dict()
            result_dict['Scenario'] = f'{participation_multiplier:.2f}x'
            # Convert all numpy types to regular Python types
            result_dict = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in result_dict.items()}
            return result_dict
        else:
            print(f"Warning: Scenario {scenario_name} not found in data")
            return None

    # Find bounding scenarios for interpolation
    scenario_keys = sorted(scenarios.keys())

    lower_mult = None
    upper_mult = None

    for mult in scenario_keys:
        if mult <= participation_multiplier:
            lower_mult = mult
        if mult >= participation_multiplier and upper_mult is None:
            upper_mult = mult
            break

    if lower_mult is None or upper_mult is None:
        print("Error: Could not find bounding scenarios")
        return None

    # Get the actual data rows
    lower_scenario = scenarios[lower_mult]
    upper_scenario = scenarios[upper_mult]

    lower_row = df[df['Scenario'] == lower_scenario]
    upper_row = df[df['Scenario'] == upper_scenario]

    if lower_row.empty or upper_row.empty:
        print(f"Warning: Could not find scenarios {lower_scenario} or {upper_scenario}")
        return None

    # Calculate interpolation weight
    weight = (participation_multiplier - lower_mult) / (upper_mult - lower_mult)

    # Get outcome columns (everything except 'Scenario')
    outcome_columns = [col for col in df.columns if col != 'Scenario']

    # Create interpolated dictionary
    result_dict = {'Scenario': f'{participation_multiplier:.2f}x'}

    for outcome in outcome_columns:
        lower_val = lower_row[outcome].iloc[0]
        upper_val = upper_row[outcome].iloc[0]

        # Linear interpolation and convert to regular Python float
        interpolated_val = lower_val + weight * (upper_val - lower_val)
        result_dict[outcome] = float(interpolated_val)

    return result_dict


# @st.cache_data
def add_cost_calculations_dict(data_dict):
    """
    Add cost calculations to lifetime data dictionary

    Args:
        data_dict: Dictionary with outcome values from interpolate_lifetime_data_dict()
        Expected keys: Lifeyears, Lifeyears_discounted, Kits_sent, Kits_sent_discounted,
                      Kits_returned, Kits_returned_discounted, Colonscopies, Colonoscopies_discounted,
                      Treatment_costs, Treatment_costs_discounted, Letters_only, Letters_only_discounted

    Returns:
        Dictionary with additional cost values added (both regular and discounted versions)

    Uses global variables:
        unit_costs, prob_incorrectCompleteFOBT, COL_complication_rate, treatment_cost_multiplier
    """
    if data_dict is None:
        return None

    # Work with a copy
    result = data_dict.copy()

    # Extract base outcome values (regular versions)
    letters_only = data_dict.get('Letters_only', 0)
    kits_sent = data_dict.get('Kits_sent', 0)
    kits_returned = data_dict.get('Kits_returned', 0)
    colonoscopies = data_dict.get('Colonscopies', 0)
    treatment_costs = data_dict.get('Treatment_costs', 0)

    # Extract discounted versions
    letters_only_discounted = data_dict.get('Letters_only_discounted', 0)
    kits_sent_discounted = data_dict.get('Kits_sent_discounted', 0)
    kits_returned_discounted = data_dict.get('Kits_returned_discounted', 0)
    colonoscopies_discounted = data_dict.get('Colonoscopies_discounted', 0)
    treatment_costs_discounted = data_dict.get('Treatment_costs_discounted', 0)

    # Calculate regular costs

    # 1. Letters only cost
    letters_only_cost = letters_only * unit_costs['letter']

    # 2. Sending FOBT cost (kits sent minus letters only)
    fobt_recipients = kits_sent - letters_only
    sending_fobt_cost = fobt_recipients * (unit_costs['kit_postage'] + unit_costs['FOBT_kit'])

    # 3. Process FOBT cost
    base_process_cost = unit_costs['kit_postage'] + unit_costs['process_kit']
    incorrect_complete_cost = prob_incorrectCompleteFOBT * (
            2 * unit_costs['kit_postage'] + unit_costs['FOBT_kit'] + unit_costs['process_kit']
    )
    total_process_unit_cost = base_process_cost + incorrect_complete_cost
    process_fobt_cost = kits_returned * total_process_unit_cost

    # 4. Colonoscopy cost
    no_comp_cost = unit_costs['gp_consult'] + unit_costs['colonoscopy_no_comp']
    comp_cost = unit_costs['gp_consult'] + unit_costs['colonoscopy_comp']
    col_cost = colonoscopies * (
            (1 - COL_complication_rate) * no_comp_cost +
            COL_complication_rate * comp_cost
    )

    # 5. New treatment cost
    new_treatment_cost = treatment_costs * treatment_cost_multiplier

    # 6. Composite costs
    screening_cost = letters_only_cost + sending_fobt_cost + process_fobt_cost
    total_cost = screening_cost + col_cost + new_treatment_cost

    # Calculate discounted costs (same formulas but with discounted inputs)

    # 1. Letters only cost (discounted)
    letters_only_cost_discounted = letters_only_discounted * unit_costs['letter']

    # 2. Sending FOBT cost (discounted)
    fobt_recipients_discounted = kits_sent_discounted - letters_only_discounted
    sending_fobt_cost_discounted = fobt_recipients_discounted * (unit_costs['kit_postage'] + unit_costs['FOBT_kit'])

    # 3. Process FOBT cost (discounted)
    process_fobt_cost_discounted = kits_returned_discounted * total_process_unit_cost

    # 4. Colonoscopy cost (discounted)
    col_cost_discounted = colonoscopies_discounted * (
            (1 - COL_complication_rate) * no_comp_cost +
            COL_complication_rate * comp_cost
    )

    # 5. New treatment cost (discounted)
    new_treatment_cost_discounted = treatment_costs_discounted * treatment_cost_multiplier

    # 6. Composite costs (discounted)
    screening_cost_discounted = letters_only_cost_discounted + sending_fobt_cost_discounted + process_fobt_cost_discounted
    total_cost_discounted = screening_cost_discounted + col_cost_discounted + new_treatment_cost_discounted

    # Add all cost values to dictionary (regular versions)
    result['letters_only_cost'] = float(letters_only_cost)
    result['sending_fobt_cost'] = float(sending_fobt_cost)
    result['process_fobt_cost'] = float(process_fobt_cost)
    result['col_cost'] = float(col_cost)
    result['new_treatment_cost'] = float(new_treatment_cost)
    result['screening_cost'] = float(screening_cost)
    result['total_cost'] = float(total_cost)

    # Add discounted cost values to dictionary
    result['letters_only_cost_discounted'] = float(letters_only_cost_discounted)
    result['sending_fobt_cost_discounted'] = float(sending_fobt_cost_discounted)
    result['process_fobt_cost_discounted'] = float(process_fobt_cost_discounted)
    result['col_cost_discounted'] = float(col_cost_discounted)
    result['new_treatment_cost_discounted'] = float(new_treatment_cost_discounted)
    result['screening_cost_discounted'] = float(screening_cost_discounted)
    result['total_cost_discounted'] = float(total_cost_discounted)

    return result


# Combined function for even simpler workflow
def get_lifetime_results(df, participation_multiplier):
    """
    One-step function: interpolate lifetime data and calculate costs

    Args:
        df: DataFrame with lifetime data
        participation_multiplier: Target participation rate

    Returns:
        Dictionary with all interpolated outcomes and calculated costs
    """
    # Interpolate the data
    interpolated = interpolate_total_data(df, participation_multiplier)

    if interpolated is None:
        return None

    # Add cost calculations
    results = add_cost_calculations_dict(interpolated)

    return results

# Helper: return cumulative values
def get_cumulative_outcome(data, outcome, years):
    vals = get_outcome_data(data, outcome, years)
    return np.cumsum(vals)


# Title and description
st.title("NBCSP Cost-Effectiveness Dashboard")

# st.subheader(
#     # "Replacing iFOBT Kits with Letters for Persistent Never-Screeners"
#     "Sending Letters Instead of iFOBT Kits to Persistent Never-Screeners"
# )

st.markdown(
    # """
    # This dashboard models the **cost-effectiveness and health outcomes** associated with sending letters to persistent never-screeners instead of iFOBT kits.
    # Adjust participation and cost assumptions to see how outcomes change over time.
    # """
    """
    This dashboard models the **cost-effectiveness** and **health outcomes** of alternative outreach strategies for persistent never-screeners in the National Bowel Cancer Screening Program (NBCSP). **Persistent never-screeners** are those who have received at least three NBCSP kits but have not participated in screening.
    """
)
# st.markdown(
#     "For this analysis, **Persistent never-screeners** are defined as individuals who have received at least three NBCSP kits but have not participated in screening.  \n\n"
#     "The analysis models the cohort who are eligible for NBCSP screening between 2026 and 2045. "
#     "Results may be shown either for this period (2026–2045) or for the **lifetime of the modelled cohort**, "
#     "which follows the same individuals beyond 2045 until death to capture the longer-term impact of screening."
# )
st.markdown(
    "Using the left panel, adjust **participation** and **cost assumptions** to see how NBCSP outcomes change. The analysis models the cohort of people who are eligible for NBCSP screening between 2026 and 2045."
)
# Sidebar controls
st.sidebar.header("Analysis Configuration")

# # Participation rate slider with predefined scenarios
# participation_rate = st.sidebar.slider(
#     "Participation Rate Multiplier",
#     min_value=0.5,
#     max_value=1.5,
#     value=1.0,
#     step=0.01,
#     help="Choose participation level. Values will be interpolated between available scenarios."
# )
#
# # Participation rate slider with percentage display
# participation_percentage = st.sidebar.slider(
#     "Relative change in uptake probability (%) in persistent never-screeners",
#     min_value=-50.0,  # -50% decrease
#     max_value=50.0,   # +50% increase
#     value=0.0,        # 0% change (no change)
#     step=1.0,
#     format="%.2f",    # Display with 2 decimal place  "%.2f"
#     help="Choose participation level change. Negative values decrease uptake, positive values increase uptake. "
# )

participation_percentage = st.sidebar.slider(
    "Relative change in uptake rate (%) in persistent never-screeners",
    min_value=-50.0,
    max_value=50.0,
    value=0.0,
    step=0.1,
    format="%.1f",
    help="Choose participation level change. Negative values decrease uptake, positive values increase uptake."
)
st.sidebar.text(f"Selected: {participation_percentage:+.1f}%")

# Convert percentage to multiplier for your functions
participation_rate = 1.0 + (participation_percentage / 100.0)

# # Optional: Display the actual multiplier for reference
# st.sidebar.caption(f"Multiplier: {participation_rate:.2f}x")


start_year = 2026
end_year = 2045
lifetime_year = 2099


# Individual cost parameters
st.sidebar.subheader("Unit Cost Parameters (AUD)")

# Initialize defaults in session state if not present
if 'unit_costs_initialized' not in st.session_state:
    st.session_state['letter_postage_cost'] = cost_letter
    st.session_state['kit_postage_cost'] = postage
    st.session_state['fobt_kit_cost'] = cost_test_kit
    st.session_state['process_kit_cost'] = cost_process_kit
    st.session_state['gp_consult_cost'] = cost_GP_consultation
    st.session_state['colonoscopy_no_comp_cost'] = cost_COL_no_complication
    st.session_state['colonoscopy_comp_cost'] = cost_COL_w_complication
    st.session_state['unit_costs_initialized'] = True

# Add reset button
if st.sidebar.button("Reset to Default Values"):
    # Reset to default values by setting them explicitly
    st.session_state['letter_postage_cost'] = cost_letter
    st.session_state['kit_postage_cost'] = postage
    st.session_state['fobt_kit_cost'] = cost_test_kit
    st.session_state['process_kit_cost'] = cost_process_kit
    st.session_state['gp_consult_cost'] = cost_GP_consultation
    st.session_state['colonoscopy_no_comp_cost'] = cost_COL_no_complication
    st.session_state['colonoscopy_comp_cost'] = cost_COL_w_complication
    st.rerun()

unit_costs = {
    'letter': st.sidebar.number_input("Cost of mailing NBCSP letter",
                                     min_value=0.0, value=cost_letter, step=0.10, format="%.2f",
                                     key='letter_postage_cost'),
    'kit_postage': st.sidebar.number_input("Cost of mailing iFOBT kit",
                                     min_value=0.0, value=postage, step=0.10, format="%.2f",
                                     key='kit_postage_cost'),
    'FOBT_kit': st.sidebar.number_input("Cost of iFOBT kit",
                                       min_value=0.0, value=cost_test_kit, step=0.50, format="%.2f",
                                       key='fobt_kit_cost'),
    'process_kit': st.sidebar.number_input("Cost of iFOBT kit analysis",
                                          min_value=0.0, value=cost_process_kit, step=1.00, format="%.2f",
                                          key='process_kit_cost'),
    'gp_consult': st.sidebar.number_input("Cost of GP consultation",
                                         min_value=0.0, value=cost_GP_consultation, step=5.00, format="%.2f",
                                         key='gp_consult_cost'),
    'colonoscopy_no_comp': st.sidebar.number_input("Cost of colonoscopy (no complications)",
                                                   min_value=0.0, value=cost_COL_no_complication, step=50.00, format="%.2f",
                                                   key='colonoscopy_no_comp_cost'),
    'colonoscopy_comp': st.sidebar.number_input("Cost of colonoscopy (with complications)",
                                               min_value=0.0, value=cost_COL_w_complication, step=100.00, format="%.2f",
                                               key='colonoscopy_comp_cost')
}

# Treatment cost multiplier
# participation_percentage = st.sidebar.slider(
#     "Relative change in uptake probability (%) in persistent never-screeners",
#     min_value=-50.0,
#     max_value=50.0,
#     value=0.0,
#     step=1.0,
#     format="%.1f",
#     help="Choose participation level change. Negative values decrease uptake, positive values increase uptake."
# )
# st.sidebar.text(f"Selected: {participation_percentage:+.1f}%")
#
# # Convert percentage to multiplier for your functions
# participation_rate = 1.0 + (participation_percentage / 100.0)
st.sidebar.subheader("Treatment Cost Adjustment")
st.sidebar.markdown("*Adjust treatment costs to account for inflation, regional differences, or updated cost estimates.*")
treatment_cost_percentage = st.sidebar.slider(
    "Treatment Cost Change (%)",
    -50.0, 50.0, 0.0, 0.1,
    format="%.1f",
    help="Adjust CRC treatment costs"
)
treatment_cost_multiplier = 1.0 + (treatment_cost_percentage / 100.0)

# Generate scenario data with interpolation
comparator_data = interpolate_all_data(annual_df, 0)  # Baseline
# baseline = interpolate_all_data(annual_df,0)

scenario_data = interpolate_all_data(annual_df, participation_rate)  #1.5)#

comparator_data = add_cost_calculations(comparator_data)
# baseline = add_cost_calculations(baseline)
scenario_data = add_cost_calculations(scenario_data)

comparator_df = calculate_totals(comparator_data)
# print(comparator_df.to_markdown())
scenario_df = calculate_totals(scenario_data)
# baseline_df = calculate_totals(baseline)
# print(baseline_df.to_markdown())

lifetime_comparator_results = get_lifetime_results(total_df, 0)
# lifetime_baseline_results = get_lifetime_results(total_df,0)
lifetime_scenario_results = get_lifetime_results(total_df, participation_rate)
# print(lifetime_comparator_results)
# print(lifetime_baseline_results)
# Display current interpolation info
# if participation_rate not in [0.5, 0.75, 1.0, 1.25, 1.5]:
# st.info(f"Current analysis uses {participation_rate:.2f}x participation rate vs the baseline for persistent never-screeners.")
if participation_percentage > 0:
    st.info(f"Current analysis assumes all persistent never-screeners receive an NBCSP letter only, with only those who request a kit receiving one. It simulates a relative {participation_percentage:.1f}% participation increase vs the baseline for persistent never-screeners and a ${unit_costs['letter']:.2f} cost per letter sent to this group. The comparator is the current NBCSP status quo, with kits being sent to persistent never-screeners.")
elif participation_percentage < 0:
    st.info(f"Current analysis assumes all persistent never-screeners receive an NBCSP letter only, with only those who request a kit receiving one. It simulates a relative {-participation_percentage:.1f}% participation decrease vs the baseline for persistent never-screeners and a ${unit_costs['letter']:.2f} cost per letter sent to this group. The comparator is the current NBCSP status quo, with kits being sent to persistent never-screeners.")
else:
    st.info(f"Current analysis assumes all persistent never-screeners receive an NBCSP letter only, with only those who request a kit receiving one. It simulates no participation change vs the baseline for persistent never-screeners and a ${unit_costs['letter']:.2f} cost per letter sent to this group. The comparator is the current NBCSP status quo, with kits being sent to persistent never-screeners.")

st.markdown("### Key Metrics Overview")
# st.markdown("Showing **differences** between selected scenario and comparator (2026-2045), except where noted")
st.markdown("Showing **annual differences** between the selected scenario and the comparator (2026-2045), except where noted")

# Key metrics display

# st.markdown("#### Screening Activity")
col1, col2, col3 = st.columns(3)
# Letters, FOBT kits, Colonoscopies
# col1, col2, col3, col4 = st.columns(4)

with col1:
    total_letters_only = \
    scenario_df[scenario_df['Outcome'] == 'persistent_never_screeners'][f'total_{start_year}_{end_year}'].iloc[0]
    # comparator_letters_only = \
    # comparator_df[comparator_df['Outcome'] == 'persistent_never_screeners'][f'total_{start_year}_{end_year}'].iloc[0]
    # diff_letters = total_letters_only - comparator_letters_only
    # pct_letters = (diff_letters / comparator_letters_only * 100) if comparator_letters_only != 0 else 0
    scenario_total_invitation = \
        scenario_df[scenario_df['Outcome'] == 'Invitation_sent_total'][f'total_{start_year}_{end_year}'].iloc[
            0]

    pct_letters = (total_letters_only / scenario_total_invitation * 100)
    # Format with M for millions
    # total_letters_display = f"{total_letters_only / 1_000_000:+.1f}M"
    total_letters_display = f"{total_letters_only / 1_000_000 / 20:+.1f}M"
    total_invitation_display = f"{scenario_total_invitation / 1_000_000:.1f}M"

    st.metric(
        "Letters to Never-Screeners",
        total_letters_display
        # delta=f"{diff_letters:+,.0f} ({diff_letters / comparator_letters_only:+.4f}%)"
        # delta=f"{pct_letters:+.1f}%"
    )
    # st.caption(f"2026-2045 total: {total_invitation_display} ({pct_letters:.1f}%)")

with col2:
    total_fobt = scenario_df[scenario_df['Outcome'] == 'kits_returned_total'][f'total_{start_year}_{end_year}'].iloc[0]
    comparator_fobt = \
    comparator_df[comparator_df['Outcome'] == 'kits_returned_total'][f'total_{start_year}_{end_year}'].iloc[0]
    diff_fobt = total_fobt - comparator_fobt
    pct_fobt = (diff_fobt / comparator_fobt * 100) if comparator_fobt != 0 else 0

    st.metric(
        "iFOBT Kits Returned",
        f"{diff_fobt/20:+,.0f}",
        delta=f"{pct_fobt:+.1f}%"
    )
    # st.caption(f"2026-2045 total: {total_fobt:,.0f}")

with col3:
    total_incidence = scenario_df[scenario_df['Outcome'] == 'CRC_cases'][f'total_{start_year}_{end_year}'].iloc[0]
    comparator_incidence = \
    comparator_df[comparator_df['Outcome'] == 'CRC_cases'][f'total_{start_year}_{end_year}'].iloc[0]
    cases_prevented = comparator_incidence - total_incidence
    pct_cases = (cases_prevented / comparator_incidence * 100) if comparator_incidence != 0 else 0

    st.metric(
        # "CRC Cases Prevented Δ",
        # f"{cases_prevented:+,.0f}",
        "CRC Cases",
        f"{cases_prevented/20:+,.0f}",
        delta=f"{pct_cases:+.1f}%"
    )
    # st.caption(f"2026-2045 total: {total_incidence:,.0f}")

# st.markdown("#### Health & Cost Impact")
col4, col5, col6 = st.columns(3)
with col4:
    total_cost = scenario_df[scenario_df['Outcome'] == 'total_cost'][f'total_{start_year}_{end_year}'].iloc[0]
    comparator_total_cost = \
    comparator_df[comparator_df['Outcome'] == 'total_cost'][f'total_{start_year}_{end_year}'].iloc[0]
    cost_diff = total_cost - comparator_total_cost
    pct_cost = (cost_diff / comparator_total_cost * 100) if comparator_total_cost != 0 else 0

    st.metric(
        # "Total Cost Δ (2026-2045)",
        # f"${cost_diff / 1_000_000:+.1f}M",
        "Total Cost",
        f"${cost_diff / 1_000_000 / 20:+.1f}M",
        delta=f"{pct_cost:+.1f}%"
    )
    # st.caption(f"2026-2045 total: ${total_cost / 1_000_000:.1f}M")

with col5:
    # Lifetime costs
    total_cost_lifetime = \
    scenario_df[scenario_df['Outcome'] == 'total_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0]
    comparator_cost_lifetime = \
    comparator_df[comparator_df['Outcome'] == 'total_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0]
    cost_diff_lifetime = total_cost_lifetime - comparator_cost_lifetime
    pct_cost_lifetime = (cost_diff_lifetime / comparator_cost_lifetime * 100) if comparator_cost_lifetime != 0 else 0

    st.metric(
        "Lifetime Total Cost (Modelled Cohort)",
        f"${cost_diff_lifetime / 1_000_000:+.1f}M",
        delta=f"{pct_cost_lifetime:+.1f}%"
    )
    # st.caption(
    #     f"Lifetime total: ${total_cost_lifetime / 1_000_000:.1f}M")

# if cost_diff_lifetime > 0:
#     st.markdown(
#     "**Note:** Costs decrease over 2026–2045, but lifetime costs increase as short-term savings are offset by downstream costs later in life, such as treatment for cancers detected at a later stage or progression to more advanced disease."
#     )
# elif cost_diff_lifetime < 0:
#     st.markdown(
#         "**Note:** Costs increase over 2026–2045, but lifetime costs decrease as early investments are offset by avoided downstream costs, such as preventing advanced cancer treatments later in life."
#     )
# Tabs (rest of the code remains the same as original)
tab1, tab2, tab3, tab4 = st.tabs(
    ["Screening Activity", "Health Outcomes", "Cost Analysis", "Cost-Effectiveness"])

with tab1:
    st.subheader("Screening Activity Summary (2026–2045)")

    screening_metrics = [
        ('persistent_never_screeners', 'Letters Sent to Persistent Never-screeners'),
        ('kits_returned_total', 'Kits Returned'),
        ('positive_iFOBT', 'Positive iFOBT'),
        ('colonoscopies_total', 'Colonoscopies')
    ]

    # Check which metrics are available in the data
    available_metrics = []
    for metric, label in screening_metrics:
        if metric in comparator_data['Outcome'].values:
            available_metrics.append((metric, label))

    if available_metrics:
        # Create a 2x2 grid of columns for individual charts
        col1, col2 = st.columns(2)

        for i, (metric, title) in enumerate(available_metrics[:4]):
            scenario_values = get_outcome_data(scenario_data, metric, end_year)
            current_values = get_outcome_data(comparator_data, metric, end_year)

            if scenario_values and current_values:
                # Trim to analysis period
                start_idx = max(0, start_year - 2026)
                end_idx = min(len(scenario_values), end_year - 2026 + 1)

                scenario_subset = scenario_values[start_idx:end_idx]
                current_subset = current_values[start_idx:end_idx]
                year_ints = list(range(2026, lifetime_year))
                years_subset = year_ints[:len(scenario_subset)]

                # Create individual figure for each metric
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=years_subset,
                        y=scenario_subset,
                        name='Scenario',
                        line=dict(color='blue', width=3)
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=years_subset,
                        y=current_subset,
                        name='Comparator',
                        line=dict(color='red', dash='dash')
                    )
                )

                fig.update_layout(
                    title=title,
                    height=300,
                    showlegend=True,
                    legend=dict(
                        orientation="h",  # Horizontal legend
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Alternate between columns for 2x2 layout
                if i % 2 == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No screening activity metrics found in the data.")


    screening_metrics = [
        ('persistent_never_screeners', 'Letters Sent to Persistent Never-screeners'),
        ('kits_returned_total', 'Kits Returned'),
        ('positive_iFOBT', 'Positive iFOBT'),
        ('colonoscopies_total', 'Colonoscopies')
    ]

    screening_summary = []
    for metric, label in screening_metrics:
        # scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0]
        # comparator_total = comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0]
        # change to annual outcome
        scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 20
        comparator_total = comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 20
        difference = scenario_total - comparator_total

        # Add arrow for trend
        if difference > 0:
            diff_display = f"↑ {difference:,.0f}"
        elif difference < 0:
            diff_display = f"↓ {abs(difference):,.0f}"
        else:
            diff_display = f"→ {difference:,.0f}"

        # screening_summary.append({
        #     'Metric': label,
        #     'Comparator (2026-2045 Total)': comparator_total,
        #     'Selected Scenario (2026-2045 Total)': scenario_total,
        #     'Difference': diff_display
        # })
        screening_summary.append({
            # 'Resource (2026-2045 Total)': label,
            'Resources Utilised Annually (2026-2045)': label,
            'Comparator': comparator_total,
            'Selected Scenario': scenario_total,
            'Difference': diff_display
        })

    # Convert to DataFrame and remove index
    df_screening = pd.DataFrame(screening_summary).reset_index(drop=True)

    # Function to color arrows
    def color_difference(val):
        if '↑' in val:
            return 'color: green'
        elif '↓' in val:
            return 'color: red'
        else:
            return 'color: black'

    # # Apply formatting: add commas and color for difference column
    # styled_df = df_screening.style.format({
    #     'Comparator (2026-2045 Total)': "{:,.0f}",
    #     'Selected Scenario (2026-2045 Total)': "{:,.0f}"
    # }).applymap(color_difference, subset=['Difference'])
    #
    # # Display the styled dataframe
    # st.dataframe(styled_df, use_container_width=True)
   # Create styled DataFrame: format numbers and color differences
    styled_df = (
        df_screening.style
        .format({
            # 'Comparator (2026-2045 Total)': "{:,.0f}",
            # 'Selected Scenario (2026-2045 Total)': "{:,.0f}"
            'Comparator': "{:,.0f}",
            'Selected Scenario': "{:,.0f}"
        })
        .applymap(color_difference, subset=['Difference'])
        # Hide the index
        .hide(axis="index")
        .set_table_styles(
            [
                # Text columns left-aligned
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td.col0", "props": [("text-align", "left")]},  # assumes first column is text
                {"selector": "td.col1", "props": [("text-align", "right")]},  # numeric col
                {"selector": "td.col2", "props": [("text-align", "right")]},  # numeric col
                {"selector": "td.col3", "props": [("text-align", "right")]}  # numeric col
            ],
            overwrite=False
        )
    )

    # CSS for consistent look and no index
    st.markdown(
        """
        <style>
        # .custom-table-container {
        #     width: 100%;
        #     margin: auto;
        # }
        .custom-table-container table {
            width: 100% !important;
            table-layout: auto;  /* allow automatic column widths */
        }
        th, td {
            padding: 6px 10px;
            font-size: 14px;
            word-wrap: break-word;
        }
        thead th {
            background-color: #f9f9f9;
            font-weight: normal;  /* <-- remove bold in headers */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render without index
    st.markdown(
        f'<div class="custom-table-container">{styled_df.to_html()}</div>',
        unsafe_allow_html=True
    )


with tab2:
    st.subheader("Health Outcomes")
    col1, col2 = st.columns(2)

    # --- CRC Cases Prevented ---
    with col1:
    #     fig_inc = go.Figure()
    #
    #     # Calculate cases prevented (comparator - scenario)
    #     scenario_incidence = get_outcome_data(scenario_data, 'CRC_cases', end_year)
    #     current_incidence = get_outcome_data(comparator_data, 'CRC_cases', end_year)
    #     cases_prevented = [current - scenario for current, scenario in zip(current_incidence, scenario_incidence)]
    #
    #     fig_inc.add_trace(go.Scatter(
    #         x=year_ints,
    #         y=cases_prevented,
    #         mode='lines+markers',
    #         name='Cases Prevented',
    #         line=dict(color='green', width=3),
    #         fill='tonexty' if all(cp >= 0 for cp in cases_prevented) else None,
    #         fillcolor='rgba(0,128,0,0.2)',
    #         hovertemplate='%{x}: %{y:,.0f} cases'
    #     ))
    #
    #     fig_inc.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
    #
    #     fig_inc.update_layout(
    #         title='CRC Cases Prevented Over Time',
    #         xaxis_title='Year',
    #         yaxis=dict(title='Cases Prevented', tickformat=',.0f'),
    #         showlegend=False
    #     )
    #
    #     st.plotly_chart(fig_inc, use_container_width=True)
    #     if participation_percentage > 0:
    #         st.caption("The short-term increase in CRC cases is due to earlier detection.")
    #     elif participation_percentage < 0:
    #         st.caption("The short-term decrease in CRC cases is due to delayed detection.")

        fig_inc = go.Figure()

        # Calculate cases change (scenario - comparator)
        scenario_incidence = get_outcome_data(scenario_data, 'CRC_cases', end_year)
        current_incidence = get_outcome_data(comparator_data, 'CRC_cases', end_year)
        cases_change = [scenario - current for scenario, current in zip(scenario_incidence, current_incidence)]

        # # Create hover text dynamically
        # hover_text = [
        #     f"{int(val):,} more cases than comparator" if val >= 0 else f"{abs(int(val)):,} less cases than comparator"
        #     for val in cases_change
        # ]

        # Determine symmetric y-axis limits
        y_max = max(cases_change + [0])
        y_min = min(cases_change + [0])
        y_abs_max = max(abs(y_max), abs(y_min))

        # # Fill color: green for positive (increase), red for negative (decrease)
        # fill_colors = ['rgba(0,128,0,0.2)' if val >= 0 else 'rgba(255,0,0,0.2)' for val in cases_change]

        # Add the line and markers
        fig_inc.add_trace(go.Scatter(
            x=year_ints,
            y=cases_change,
            mode='lines+markers',
            name='Cases Change',
            line=dict(color='blue', width=3),
            # hoverinfo='text',
            # text=hover_text
            customdata=[[abs(c), "more" if c > 0 else "fewer"] for c in cases_change],
            hovertemplate='%{x}: %{customdata[0]:,.0f} cases %{customdata[1]} than comparator<extra></extra>'

        ))

        # Add zero line
        fig_inc.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

        # Update layout with symmetric y-axis and horizontal arrows
        fig_inc.update_layout(
            title='Change in CRC Cases Over Time',
            xaxis_title='Year',
            yaxis=dict(
                range=[-y_abs_max-5, y_abs_max+5],
                title='← Cases Decrease | Cases Increase →',
                tickformat=',.0f',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            ),
            showlegend=False
        )

        st.plotly_chart(fig_inc, use_container_width=True)

        # Optional caption for context
        if participation_percentage > 0:
            st.caption("The short-term increase in CRC cases is due to earlier detection.")
        elif participation_percentage < 0:
            st.caption("The short-term decrease in CRC cases is due to delayed detection.")

    # --- CRC Deaths Prevented ---
    with col2:
        fig_mort = go.Figure()

        scenario_mortality = get_outcome_data(scenario_data, 'CRC_deaths', end_year)
        current_mortality = get_outcome_data(comparator_data, 'CRC_deaths', end_year)
        deaths_change = [scenario - current for scenario, current in zip(scenario_mortality, current_mortality)]

        # Determine symmetric y-axis limits
        y_max = max(deaths_change + [0])
        y_min = min(deaths_change + [0])
        y_abs_max = max(abs(y_max), abs(y_min))

        fig_mort.add_trace(go.Scatter(
            x=year_ints,
            y=deaths_change,
            mode='lines+markers',
            name='Deaths Change',
            line=dict(color='crimson', width=3),
            customdata=[[abs(c), "more" if c > 0 else "fewer"] for c in deaths_change],
            hovertemplate='%{x}: %{customdata[0]:,.0f} deaths %{customdata[1]} than comparator<extra></extra>'

        ))

        # Add zero line
        fig_mort.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

        # Update layout with symmetric y-axis and horizontal arrows
        fig_mort.update_layout(
            title='Change in CRC Deaths Over Time',
            xaxis_title='Year',
            yaxis=dict(
                range=[-y_abs_max-5, y_abs_max+5],
                title='← Deaths Decrease | Deaths Increase →',
                tickformat=',.0f',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            ),
            showlegend=False
        )

        st.plotly_chart(fig_mort, use_container_width=True)
        # st.caption("Positive values indicate deaths prevented by the scenario compared with the comparator.")



    # Health outcomes summary
    # if end_year == 2045:
    #     st.subheader(f"Health Impact Summary (2026-{end_year})")
    # elif end_year == 2099:
    st.subheader("Health impact over the modelled cohort's lifetime")

    health_metrics = [
        ('stage1_cases', 'Stage 1 CRC Cases'),
        ('CRC_cases', 'Total CRC Cases'),
        ('CRC_deaths', 'CRC Deaths'),
        ('lifeyears', 'Life-years')
    ]

    health_summary = []
    for metric, label in health_metrics:
        scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{lifetime_year}'].iloc[0]
        comparator_total = \
        comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{lifetime_year}'].iloc[0]
        difference = scenario_total - comparator_total

        # Add arrow for trend
        if difference > 0:
            diff_display = f"↑ {difference:,.0f}"
        elif difference < 0:
            diff_display = f"↓ {abs(difference):,.0f}"
        else:
            diff_display = f"→ {difference:,.0f}"

        health_summary.append({
            'Outcome': label,
            'Comparator': comparator_total,
            'Selected Scenario': scenario_total,
            'Difference': diff_display
        })

    # Convert to DataFrame and remove index
    df_health = pd.DataFrame(health_summary)  # .reset_index(drop=True)


    # --- Function to color arrows ---
    def color_difference(val):
        if '↑' in val:
            return 'color: green'
        elif '↓' in val:
            return 'color: red'
        else:
            return 'color: black'


    # Create styled DataFrame: format numbers and color differences
    styled_df = (
        df_health.style
        .format({
            'Comparator': "{:,.0f}",
            'Selected Scenario': "{:,.0f}"
        })
        .applymap(color_difference, subset=['Difference'])
        # Hide the index
        .hide(axis="index")
        .set_table_styles(
            [
                # Text columns left-aligned
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td.col0", "props": [("text-align", "left")]},  # assumes first column is text
                {"selector": "td.col1", "props": [("text-align", "right")]},  # numeric col
                {"selector": "td.col2", "props": [("text-align", "right")]},  # numeric col
                {"selector": "td.col3", "props": [("text-align", "right")]}  # numeric col
            ],
            overwrite=False
        )
    )

    # CSS for consistent look and no index
    st.markdown(
        """
        <style>
        # .custom-table-container {
        #     width: 100%;
        #     margin: auto;
        # }
        .custom-table-container table {
            width: 100% !important;
            table-layout: auto;  /* allow automatic column widths */
        }
        th, td {
            padding: 6px 10px;
            font-size: 14px;
            word-wrap: break-word;
        }
        thead th {
            background-color: #f9f9f9;
            font-weight: normal;  /* <-- remove bold in headers */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render without index
    st.markdown(
        f'<div class="custom-table-container">{styled_df.to_html()}</div>',
        unsafe_allow_html=True
    )

    # --- Display trend description under table ---
    # stage1_diff = df_health.loc[df_health['Outcome'] == 'Stage 1 CRC Cases', 'Difference'].iloc[0]
    # total_cases_diff = df_health.loc[df_health['Outcome'] == 'Total CRC Cases', 'Difference'].iloc[0]
    # deaths_diff = df_health.loc[df_health['Outcome'] == 'CRC Deaths', 'Difference'].iloc[0]
    # life_years_diff = df_health.loc[df_health['Outcome'] == 'Life-years', 'Difference'].iloc[0]
    def display_crc_trend(participation_change):
        """
        Display a trend-focused markdown under the summary table in Streamlit.

        Positive trends: +Stage 1, -Total Cases, -Deaths, +Life-years
        Negative trends: opposite
        """
        # Determine if trend is beneficial or adverse
        if participation_change > 0:
            trend_text = f"""

            Across the modelled cohort’s lifetime, the scenario shifted **CRC cases** toward *earlier stages* due to the increase in uptake in persistent never-screeners,
            while reducing overall **CRC incidence** and **mortality**. These improvements translated into gains in **life-years**,
            reflecting the survival benefit of the scenario.
            """
        elif participation_change == 0:
            trend_text = f"""

            No health impact across the modelled cohort’s lifetime.
            """
        else:
            trend_text = f"""

            Across the modelled cohort’s lifetime, the scenario shifted **CRC cases** away from *earlier stages* due to the decrease in uptake in persistent never-screeners,
            resulting in an increase in overall **CRC incidence** and **mortality**. These changes led to a reduction in **life-years**,
            indicating a net negative health impact.
            """
        st.markdown(trend_text)

    display_crc_trend(participation_percentage)


with tab3:
#     st.subheader("Cumulative Cost Analysis")
#
    # --- Cumulative costs over time ---
    fig_costs = go.Figure()

    # Convert to millions of AUD
    cum_screening = get_cumulative_outcome(scenario_data, 'screening_cost', lifetime_year) / 1_000_000
    cum_colonoscopy = get_cumulative_outcome(scenario_data, 'col_cost', lifetime_year) / 1_000_000
    cum_treatment = get_cumulative_outcome(scenario_data, 'treatment_cost', lifetime_year) / 1_000_000

    # Screening costs
    fig_costs.add_trace(go.Scatter(
        x=year_ints,
        y=cum_screening,
        mode='lines+markers',
        name='Screening Costs',
        line=dict(color='green', width=2),
        stackgroup='one',
        fillcolor='rgba(0,128,0,0.2)',
        hovertemplate='%{x}: %{y:,.1f} M AUD'
    ))

    # Colonoscopy costs
    fig_costs.add_trace(go.Scatter(
        x=year_ints,
        y=cum_colonoscopy,
        mode='lines+markers',
        name='Colonoscopy Costs',
        line=dict(color='blue', width=2),
        stackgroup='one',
        fillcolor='rgba(0,0,255,0.2)',
        hovertemplate='%{x}: %{y:,.1f} M AUD'
    ))

    # Treatment costs
    fig_costs.add_trace(go.Scatter(
        x=year_ints,
        y=cum_treatment,
        mode='lines+markers',
        name='Treatment Costs',
        line=dict(color='orange', width=2),
        stackgroup='one',
        fillcolor='rgba(255,165,0,0.2)',
        hovertemplate='%{x}: %{y:,.1f} M AUD'
    ))

    fig_costs.update_layout(
        title='Cumulative cost in the modelled cohort, breakdown over time',
        xaxis_title='Year',
        yaxis=dict(title='Cumulative Cost (AUD, million)', tickformat=',.0f'),
        hovermode='x unified'
    )

    st.plotly_chart(fig_costs, use_container_width=True)
    # st.markdown(
    #     "**Note:** Annual costs decline sharply in later years as many in the modelled cohort gradually reach the maximum model age. To avoid this artefact, cumulative costs are shown instead of yearly values."
    # )
    st.subheader("Cost Differences")
    # --- Cost summary table ---
    cost_metrics = [
        ('screening_cost', 'Total Screening Costs'),
        ('col_cost', 'Colonoscopy Costs'),
        ('new_treatment_cost', 'Treatment Costs'),
        ('total_cost', 'Total Costs')
    ]


    # Build table as before
    cost_summary = []
    for metric, label in cost_metrics:
        lifetime_scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{lifetime_year}'].iloc[0] / 1000000
        lifetime_comparator_total = comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{lifetime_year}'].iloc[0] / 1000000
        lifetime_diff = lifetime_scenario_total - lifetime_comparator_total

        # short_scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 1000000
        # short_comparator_total = comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 1000000
        short_scenario_total = scenario_df[scenario_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 1000000 / 20
        short_comparator_total = comparator_df[comparator_df['Outcome'] == metric][f'total_{start_year}_{end_year}'].iloc[0] / 1000000 / 20
        short_diff = short_scenario_total - short_comparator_total


        # Add arrow for trend
        def add_arrow(diff):
            if diff > 0:
                return f"↑ {diff:,.1f}"
            elif diff < 0:
                return f"↓ {abs(diff):,.1f}"
            else:
                return f"→ {diff:,.1f}"

        cost_summary.append({
            'Cost Category': label,
            # 'Difference over 2026-2045 (AUD, million)': add_arrow(short_diff),
            'Annual Difference, 2026-2045 (AUD, million)': add_arrow(short_diff),
            'Lifetime Difference (AUD, million)': add_arrow(lifetime_diff)
        })

    df_cost_summary = pd.DataFrame(cost_summary)

    # Apply coloring to differences
    def color_diff(val):
        if '↑' in val:
            return 'color: green'
        elif '↓' in val:
            return 'color: red'
        else:
            return 'color: black'

    # styled_df = df_cost_summary.style.applymap(color_diff, subset=['Difference over 2026-2045 (AUD, million)','Difference over Lifetime (AUD, million)'])
    #
    # st.dataframe(styled_df, use_container_width=True)

    # Create styled DataFrame: format numbers and color differences
    styled_df = (
        df_cost_summary.style
        # .format({
        #     # 'Comparator (2026-2045 Total)': "{:,.0f}",
        #     # 'Selected Scenario (2026-2045 Total)': "{:,.0f}"
        #     'Comparator': "{:,.0f}",
        #     'Selected Scenario': "{:,.0f}"
        # })
        # .applymap(color_diff, subset=['Difference over 2026-2045 (AUD, million)','Difference over Lifetime (AUD, million)'])
        .applymap(color_diff, subset=['Annual Difference, 2026-2045 (AUD, million)','Lifetime Difference (AUD, million)'])
        # Hide the index
        .hide(axis="index")
        .set_table_styles(
            [
                # Text columns left-aligned
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td.col0", "props": [("text-align", "left")]},  # assumes first column is text
                {"selector": "td.col1", "props": [("text-align", "right")]},  # numeric col
                {"selector": "td.col2", "props": [("text-align", "right")]},  # numeric col
                # {"selector": "td.col3", "props": [("text-align", "right")]}  # numeric col
            ],
            overwrite=False
        )
    )

    # CSS for consistent look and no index
    st.markdown(
        """
        <style>
        # .custom-table-container {
        #     width: 100%;
        #     margin: auto;
        # }
        .custom-table-container table {
            width: 100% !important;
            table-layout: auto;  /* allow automatic column widths */
        }
        th, td {
            padding: 6px 10px;
            font-size: 14px;
            word-wrap: break-word;
        }
        thead th {
            background-color: #f9f9f9;
            font-weight: normal;  /* <-- remove bold in headers */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render without index
    st.markdown(
        f'<div class="custom-table-container">{styled_df.to_html()}</div>',
        unsafe_allow_html=True
    )


    # st.markdown(f"As uptake increases among persistent never-screeners, costs shift across the screening pathway. **Screening costs** fall as iFOBT kits are replaced with lower-cost targeted letters for the individuals unlikely to participate. **Colonoscopy costs** rise as more people engage with screening and require diagnostic follow-up. However, **treatment costs** decrease with higher uptake through earlier cancer detection and prevention. The result is a net decrease in **total costs**  through reduced screening expenditure and avoided treatment costs.")

    # Calculate all cost differences - both lifetime and short-term
    screening_lifetime = \
    scenario_df[scenario_df['Outcome'] == 'screening_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0] - \
    comparator_df[comparator_df['Outcome'] == 'screening_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0]

    colonoscopy_lifetime = scenario_df[scenario_df['Outcome'] == 'col_cost'][f'total_{start_year}_{lifetime_year}'].iloc[
                               0] - \
                           comparator_df[comparator_df['Outcome'] == 'col_cost'][
                               f'total_{start_year}_{lifetime_year}'].iloc[0]

    treatment_lifetime = \
    scenario_df[scenario_df['Outcome'] == 'new_treatment_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0] - \
    comparator_df[comparator_df['Outcome'] == 'new_treatment_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0]

    total_lifetime = scenario_df[scenario_df['Outcome'] == 'total_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0] - \
                     comparator_df[comparator_df['Outcome'] == 'total_cost'][f'total_{start_year}_{lifetime_year}'].iloc[0]

    total_short = scenario_df[scenario_df['Outcome'] == 'total_cost'][f'total_{start_year}_{end_year}'].iloc[0] - \
                  comparator_df[comparator_df['Outcome'] == 'total_cost'][f'total_{start_year}_{end_year}'].iloc[0]


    def generate_cost_summary_text(screening_diff, colonoscopy_diff, treatment_diff, total_lifetime_diff, total_short_diff, participation_diff):
        """
        Generate contextual summary based on cost differences
        Only total cost considers both short-term and lifetime perspectives

        Returns appropriate text for all possible combinations of cost changes
        """

        # Convert to millions for readability
        screening_m = screening_diff / 1e6
        colonoscopy_m = colonoscopy_diff / 1e6
        treatment_m = treatment_diff / 1e6
        total_lifetime_m = total_lifetime_diff / 1e6
        total_short_m = total_short_diff / 1e6

        if colonoscopy_m==0 and treatment_m==0:
            # Special case: No change in participation (comparator scenario)
            summary = "When participation rates remain unchanged (vs comparator scenario), "  #the intervention still yields cost savings by optimizing resource allocation. "
            summary += "**screening costs** fall as iFOBT kits are no longer sent to persistent never-screeners, replaced with lower-cost targeted letters. "
            summary += "**Colonoscopy costs** and **treatment costs** remain same as screening uptake is unchanged. "

            if total_lifetime_diff < 0:
                summary += f"The result is a net decrease in **total costs**. "
            else:
                summary += f"**Total costs** change by ${abs(total_lifetime_m):.1f}M."

        else:
            # Build the summary text
            summary = "As uptake changes among persistent never-screeners, costs shift across the screening pathway. Under current cost parameters, "

            # SCREENING COSTS (lifetime only)
            if screening_diff < 0:
                if participation_diff < 0 :
                    summary += "**screening costs** fall as iFOBT kits are replaced with lower-cost targeted letters for individuals unlikely to participate, and lower uptake reduces overall screening expenditure. "
                else:
                    summary += "**screening costs** fall as iFOBT kits are replaced with lower-cost targeted letters for individuals unlikely to participate. "
            elif screening_diff > 0:
                summary += "**screening costs** rise under the current cost parameters. "
            else:
                summary += "**screening costs** remain unchanged. "

            # COLONOSCOPY COSTS (lifetime only)
            if colonoscopy_diff > 0:
                summary += "**Colonoscopy costs** rise as more people engage with screening and require diagnostic follow-up. "
            elif colonoscopy_diff < 0:
                summary += "**Colonoscopy costs** fall as less people engage with screening and require diagnostic follow-up. "
            else:
                summary += "**Colonoscopy costs** remain same. "

            # TREATMENT COSTS (lifetime only)
            if treatment_diff < 0:
                summary += "**Treatment costs** decrease with higher uptake through earlier cancer detection and prevention. "
            elif treatment_diff > 0:
                summary += "However, **treatment costs** increase in this scenario. "
            else:
                summary += "**Treatment costs** remain unchanged. "

            # TOTAL COSTS - Consider both short-term and lifetime
            # Check if patterns differ between short and lifetime
            if (total_short_diff > 0 and total_lifetime_diff < 0):
                # Classic prevention pattern: short-term costs, lifetime savings
                summary += f"**total costs** increase over {start_year}–{end_year}, "
                summary += f"but the intervention achieves net lifetime savings in the modelled cohort. "
                summary += "This long-term return is achieved through "
                if screening_diff < 0 and treatment_diff < 0:
                    summary += "reduced screening expenditure and avoided treatment costs."
                elif treatment_diff < 0:
                    summary += "avoided treatment costs."
                else:
                    summary += "downstream savings."

            elif (total_short_diff < 0 and total_lifetime_diff < 0):
                # Cost-saving in both periods (ideal)
                summary += f"The result is a net decrease in **total costs**, "

                if screening_diff < 0 and treatment_diff < 0:
                    summary += " through reduced screening expenditure and avoided treatment costs."
                elif screening_diff < 0:
                    summary += " primarily through reduced screening expenditure."
                elif treatment_diff < 0:
                    summary += " primarily through avoided treatment costs."
                else:
                    summary += "."

            elif (total_short_diff > 0 and total_lifetime_diff > 0):
                # Cost-increasing in both periods
                # summary += f"**Total costs** increase. in both the short-term (${abs(total_short_m):.1f}M over {start_year}–{end_year}) "
                # summary += f"and lifetime (${abs(total_lifetime_m):.1f}M). "

                if colonoscopy_diff > abs(screening_diff) + abs(treatment_diff):
                    summary += f"the increase in colonoscopy costs outweighs other savings, resulting in the **total costs** increase. "
                elif screening_diff > 0 and treatment_diff > 0:
                    summary += "increases across screening, colonoscopy, and treatment costs drive the **total costs** increase. "
                else:
                    summary += f"**total costs** increase."
            elif (total_short_diff < 0 and total_lifetime_diff > 0):
                # Short-term savings, lifetime cost (unusual)
                summary += f"**Total costs** decrease over {start_year}–{end_year}, "
                summary += "but increase over the modelled cohort's lifetime. Short-term savings are offset by downstream costs later in life, such as treatment for cancers detected at a later stage or progression to more advanced disease."

            # elif abs(total_lifetime_diff) < 1e6 and abs(total_short_diff) < 1e6:
            #     # Cost-neutral in both periods
            #     summary += f"**total costs** remain approximately cost-neutral in both the short-term and lifetime (within ${max(abs(total_short_m), abs(total_lifetime_m)):.1f}M). "
            #     summary += "Changes across different cost categories largely offset each other."

            # else:
            #     # One period near zero, use the non-zero one
            #     if abs(total_lifetime_diff) > 1e6:
            #         if total_lifetime_diff < 0:
            #             summary += f"Overall, **total costs** decrease by ${abs(total_lifetime_m):.1f}M over the lifetime."
            #         else:
            #             summary += f"Under current cost parameters, **total costs** increase by ${abs(total_lifetime_m):.1f}M over the lifetime."
            #     else:
            #         if total_short_diff < 0:
            #             summary += f"Overall, **total costs** decrease by ${abs(total_short_m):.1f}M over {start_year}–{end_year}."
            #         else:
            #             summary += f"Under current cost parameters, **total costs** increase by ${abs(total_short_m):.1f}M over {start_year}–{end_year}."

        return summary


    # Generate and display the summary
    cost_summary_text = generate_cost_summary_text(screening_lifetime, colonoscopy_lifetime, treatment_lifetime,
                                                   total_lifetime, total_short, participation_percentage)
    st.markdown(cost_summary_text)


with tab4:
    st.subheader("Cost-Effectiveness Analysis")

    # Calculate key metrics
    discounted_scenario_total = lifetime_scenario_results['total_cost_discounted']
    discounted_comparator_total = lifetime_comparator_results['total_cost_discounted']
    total_cost_diff = discounted_scenario_total - discounted_comparator_total
    # Calculate percentage changes
    pct_cost_diff = total_cost_diff / discounted_comparator_total * 100

    discounted_scenario_lifeyears = lifetime_scenario_results['Lifeyears_discounted']
    discounted_comparator_lifeyears = lifetime_comparator_results['Lifeyears_discounted']
    lifeyears_gained = discounted_scenario_lifeyears - discounted_comparator_lifeyears
    # Calculate percentage changes
    pct_lifeyear_diff = lifeyears_gained / discounted_comparator_lifeyears * 100

    # Calculate ICER once at the top
    q = lifeyears_gained
    c = total_cost_diff
    icer = c / q if abs(lifeyears_gained) > 0 else 0

    # Determine ICER interpretation (move this up)
    if abs(lifeyears_gained) > 0:
        if lifeyears_gained > 0:  # Positive health outcomes
            if c < 0:
                icer_interpretation = "Dominant (Cost-saving)"
                status_color = "success"
            elif icer <= 30000:
                icer_interpretation = "Highly cost-effective"
                status_color = "success"
            elif icer <= 50000:
                icer_interpretation = "Cost-effective"
                status_color = "warning"
            else:
                icer_interpretation = "Not cost-effective"
                status_color = "error"
        else:  # Negative health outcomes
            if c > 0:
                icer_interpretation = "Dominated (Costly & harmful)"
                status_color = "error"
            else:
                icer_interpretation = "Cost-saving but harmful"
                status_color = "warning"
    else:
        icer_interpretation = "No health impact"
        status_color = "info"

    # Your existing metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total costs change in the modelled cohort (discounted)", f"${total_cost_diff/1000000:.2f} million")  #, delta=f"{pct_cost_diff:+.2f}%")
    with col2:
        st.metric("Life-years change in the modelled cohort (discounted)", f"{lifeyears_gained:,.0f}")  #, delta=f"{pct_lifeyear_diff:+.2f}%")

    if c==0:
        st.info(f"No cost change")
    elif c > 0 and q > 0:
        # if icer <= 30000:
        #     ce_str = "This falls below the $30k per life-year willingness-to-pay threshold, indicating that this scenario is considered cost-effective."
        # elif icer <= 50000:
        #     ce_str = "This falls below the $50k per life-year willingness-to-pay threshold, indicating that this scenario is considered cost-effective."
        # else:
        #     ce_str = "This falls above the $50k per life-year willingness-to-pay threshold, indicating that this scenario is considered not cost-effective."
        #
        message = f"The selected scenario is more effective but also more costly than the comparator, with an ICER of ${icer:,.0f} per life-year saved. "
        st.success(message)

    elif c>0 and q<0:
        st.error(f"The selected scenario is less effective and more costly than the comparator.")
    elif c<0 and q>0:
        st.success(f"The selected scenario is more effective and cost-saving than the comparator.")
    elif c<0 and q<0:
        st.warning(f"The selected scenario is less costly but also less effective than the comparator.")  #, with an ICER of ${icer:,.0f} per life-year lost. The acceptability of this trade-off depends on whether the savings are considered sufficient to justify the loss in health outcomes.")
    elif c<0 and q==0:
        st.success(f"The selected scenario has no health impact but cost-saving than the comparator.")
    elif c>0 and q==0:
        st.success(f"The selected scenario has no health impact but more costly than the comparator.")

    st.markdown("*All costs and life-years discounted by 5% annually from 2026.\
                        Cost-effectiveness ratios under the indicative $50,000 per life-year saved willingness-to-pay threshold are considered cost-effective.\
                            Change in costs includes screening-related costs, CRC diagnosis and treatment costs.*")

    # if lifeyears_gained != 0:  # Show plot for any non-zero life years
    # Dynamic scaling based on actual data point with padding
    padding_factor = 1.5  # 50% padding around the data point

    # X-axis: life years in thousands (ensure minimum range for visibility)
    x_abs = abs(q)
    x_range = max(x_abs * padding_factor, 500)  # Minimum range of 500 life-years
    x_max = x_range
    x_min = -x_range

    # Y-axis: costs in millions (convert from dollars to millions for display)
    c_millions = c / 1000000  # Convert dollars to millions for plotting
    y_abs = abs(c_millions)
    y_range = max(y_abs * padding_factor, 5)  # Minimum range for 5 million
    y_max = y_range
    y_min = -y_range

    # Create figure
    fig = go.Figure()

    # Always show cost-effectiveness regions in the positive quadrant (top-right)
    # Scale the threshold lines for the data range (costs in millions, life-years in thousands)
    x_fill = np.linspace(0, x_max, 100)

    # # Define boundary lines ($/life-year slopes converted to millions$/life-year)
    # y0 = np.zeros_like(x_fill)
    # y30 = (30000 / 1000000) * x_fill  # $30k per life-year in millions
    # y50 = (50000 / 1000000) * x_fill  # $50k per life-year in millions
    #
    # # 1️⃣ Shade between y=0 and y=30,000x (most cost-effective)
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y0, y30[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(0, 255, 0, 0.15)",  # Light green
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # # 2️⃣ Shade between y=30,000x and y=50,000x (cost-effective)
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y30, y50[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(255, 255, 0, 0.15)",  # Light yellow
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # # 3️⃣ Shade above y=50,000x (not cost-effective)
    # y_cap = np.full_like(x_fill, y_max)
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y50, y_cap[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(255, 0, 0, 0.15)",  # Light red
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # # Add threshold lines as references
    # fig.add_trace(go.Scatter(x=x_fill, y=y30, mode='lines',
    #                          line=dict(color='green', width=1, dash='dot'),
    #                          name='$30k/life-year', showlegend=False, hoverinfo="skip"))
    # fig.add_trace(go.Scatter(x=x_fill, y=y50, mode='lines',
    #                          line=dict(color='orange', width=1, dash='dot'),
    #                          name='$50k/life-year', showlegend=False, hoverinfo="skip"))
    #
    # # Add region labels if they fit
    # if x_max > 1000:  # Only if we have room for labels (adjusted for thousands scale)
    #     x_label = x_max * 0.9
    #     y_label_1 = (15000 / 1000000) * x_label  # Middle of first region
    #     y_label_2 = (40000 / 1000000) * x_label  # Middle of second region
    #
    #     if y_label_1 <= y_max:
    #         fig.add_annotation(x=x_label, y=y_label_1, text="Highly<br>cost-effective",
    #                            showarrow=False, font=dict(size=8, color="darkgreen"))
    #     if y_label_2 <= y_max:
    #         fig.add_annotation(x=x_label, y=y_label_2, text="Cost-effective",
    #                            showarrow=False, font=dict(size=8, color="darkorange"))

    # Show cost-effectiveness regions in the bottom-left quadrant
    # Scale the threshold lines for the data range (costs in millions, life-years in thousands)
    x_fill = np.linspace(x_min, 0, 100)  # Changed: from x_min to 0 instead of 0 to x_max

    # # Define boundary lines ($/life-year slopes converted to millions$/life-year)
    # y0 = np.zeros_like(x_fill)
    # y30 = (30000 / 1000000) * x_fill  # $30k per life-year in millions
    # y50 = (50000 / 1000000) * x_fill  # $50k per life-year in millions
    #
    # # 1️⃣ Shade between y=0 and y=30,000x (most cost-effective)
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y0, y30[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(255, 0, 0, 0.15)",  # Light red
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # # 2️⃣ Shade between y=30,000x and y=50,000x (cost-effective)
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y30, y50[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(255, 255, 0, 0.15)",  # Light yellow
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # # 3️⃣ Shade below y=50,000x (not cost-effective)
    # y_cap = np.full_like(x_fill, y_min)  # Changed: y_min instead of y_max
    # fig.add_trace(go.Scatter(
    #     x=np.concatenate([x_fill, x_fill[::-1]]),
    #     y=np.concatenate([y50, y_cap[::-1]]),
    #     fill="toself",
    #     fillcolor="rgba(0, 255, 0, 0.15)",  # Light green
    #     line=dict(color="rgba(0,0,0,0)"),
    #     hoverinfo="skip",
    #     showlegend=False
    # ))

    # # Add threshold lines as references
    # fig.add_trace(go.Scatter(x=x_fill, y=y30, mode='lines',
    #                          line=dict(color='green', width=1, dash='dot'),
    #                          name='$30k/life-year', showlegend=False, hoverinfo="skip"))
    # fig.add_trace(go.Scatter(x=x_fill, y=y50, mode='lines',
    #                          line=dict(color='orange', width=1, dash='dot'),
    #                          name='$50k/life-year', showlegend=False, hoverinfo="skip"))
    #
    # # Add region labels if they fit
    # if abs(x_min) > 1000:  # Changed: check x_min instead of x_max
    #     x_label = x_min * 0.9  # Changed: use x_min instead of x_max
    #     y_label_1 = (15000 / 1000000) * x_label  # Middle of first region
    #     y_label_2 = (40000 / 1000000) * x_label  # Middle of second region
    #
    #     if y_label_1 >= y_min:  # Changed: >= y_min instead of <= y_max
    #         fig.add_annotation(x=x_label, y=y_label_1, text="Highly<br>cost-effective",
    #                            showarrow=False, font=dict(size=8, color="darkgreen"))
    #     if y_label_2 >= y_min:  # Changed: >= y_min instead of <= y_max
    #         fig.add_annotation(x=x_label, y=y_label_2, text="Cost-effective",
    #                            showarrow=False, font=dict(size=8, color="darkorange"))


    # Add subtle quadrant background shading
    # Top-right: Cost increase, life years gained
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max,
                  fillcolor="rgba(0, 0, 255, 0.03)", line=dict(width=0))

    # Top-left: Cost increase, life years lost (worst case)
    fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max,
                  fillcolor="rgba(255, 0, 0, 0.03)", line=dict(width=0))

    # Bottom-left: Cost saving, life years lost (mixed outcome)
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0,
                  fillcolor="rgba(255, 255, 0, 0.03)", line=dict(width=0))

    # Bottom-right: Cost saving, life years gained (dominant - best case)
    fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0,
                  fillcolor="rgba(0, 255, 0, 0.03)", line=dict(width=0))

    # Add quadrant labels (adjusted positioning for new scale)
    label_size = 10
    fig.add_annotation(x=x_max * 0.85, y=y_max * 0.85, text="Potentially cost-effective<br>(WTP threshold-dependent)",
                       showarrow=False, font=dict(size=label_size, color="blue"))
    fig.add_annotation(x=x_min * 0.85, y=y_max * 0.85, text="Dominated<br>(More costly & Less effective)",
                       showarrow=False, font=dict(size=label_size, color="red"))
    fig.add_annotation(x=x_min * 0.85, y=y_min * 0.85, text="Trade-off<br>(Less costly & Less effective)",
                       showarrow=False, font=dict(size=label_size, color="orange"))
    fig.add_annotation(x=x_max * 0.85, y=y_min * 0.85, text="Dominant<br>(Less costly & More effective)",
                       showarrow=False, font=dict(size=label_size, color="green"))

    # Add axes lines
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_vline(x=0, line=dict(color="black", width=1))

    # Add the intervention point (using millions for y-axis)
    point_color = "black"
    point_size = 15  # Larger for better visibility on small scale

    if lifeyears_gained < 0:
        point_color = "red"  # Red for harmful interventions
    elif lifeyears_gained >= 0 and c < 0:
        point_color = "green"  # Green for dominant interventions

    fig.add_trace(go.Scatter(
        x=[q], y=[c_millions],  # Use c_millions for plotting
        mode="markers",
        marker=dict(size=point_size, color=point_color, symbol="circle",
                    line=dict(color="white", width=2)),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Calculate and display ICER prominently
    if abs(lifeyears_gained) > 0:
        icer = c / lifeyears_gained  # c is already in dollars
        icer_text = f"ICER: ${icer:,.0f}/life-year"

        # Color-code ICER based on both cost and health impact
        if lifeyears_gained > 0:  # Positive health outcomes
            if c < 0:  # Cost-saving AND health-improving
                icer_color = "green"
                icer_interpretation = "Dominant (Cost-saving)"
            elif icer <= 30000:
                icer_color = "darkgreen"
                icer_interpretation = "Highly cost-effective"
            elif icer <= 50000:
                icer_color = "orange"
                icer_interpretation = "Cost-effective"
            else:
                icer_color = "red"
                icer_interpretation = "Not cost-effective"
        else:  # Negative health outcomes (life years lost)
            if c > 0:  # Costs more AND harms health
                icer_color = "darkred"
                icer_interpretation = "Dominated (Costly & harmful)"
            else:  # Costs less but still harms health
                icer_color = "orange"
                icer_interpretation = "Cost-saving but harmful"
    else:
        icer_text = "ICER: Undefined"
        icer_color = "gray"
        icer_interpretation = "No health impact"

    # Enhanced dashed line from origin to point
    fig.add_trace(go.Scatter(
        x=[0, q], y=[0, c_millions],  # Use c_millions
        mode="lines",
        line=dict(dash="dash", color=point_color, width=3),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Layout optimized for cohort-level data
    fig.update_layout(
        title=f"Cost-Effectiveness Analysis",
        xaxis=dict(
            title="Change in total life-years (discounted)",  # Updated title
            showgrid=True, gridcolor="lightgray", gridwidth=0.5,
            zeroline=True, zerolinecolor="black", zerolinewidth=2,
            showline=True, linewidth=1, range=[x_min, x_max],
            tickformat=",.0f"  # Comma format for thousands
        ),
        yaxis=dict(
            title="Change in total costs (discounted, millions)",  # Updated title
            showgrid=True, gridcolor="lightgray", gridwidth=0.5,
            zeroline=True, zerolinecolor="black", zerolinewidth=2,
            showline=True, linewidth=1, range=[y_min, y_max],
            tickformat=",.0f"  # Millions format with 0 decimal place
        ),
        margin=dict(t=60, l=80, r=80, b=70),
        plot_bgcolor="white",
        font=dict(size=10)
    )

    st.plotly_chart(fig, use_container_width=True)


    # elif lifeyears_gained == 0:
    #     st.info("Cannot create cost-effectiveness plot: No change in life-years from the selected scenario")
    #     st.markdown(
    #         "The selected scenario has no measurable impact on life years, making cost-effectiveness analysis not applicable.")