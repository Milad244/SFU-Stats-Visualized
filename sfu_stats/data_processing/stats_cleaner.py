import os
import time
from enum import Enum
import pandas as pd
import functools
import pdfplumber
import concurrent.futures
import json
from .stats_structure import Stat, StatCategory
from pathlib import Path


def get_project_root() -> str:
    """
    Gets the project root folder.
    :return: the path of the project root
    """
    return str(Path(__file__).parent.parent)


def get_raw_path(file_name: str) -> str:
    """
    Gets the path for varying raw data files.
    :param file_name: the relative file path from raw_data
    :return: the full path for the given raw data file
    """
    raw_path = "raw_data"
    return os.path.join(get_project_root(), raw_path, file_name)


def get_output_data_path() -> str:
    """
    Gets the relative path for the cleaned output data
    :return: the relative path for the cleaned output data
    """
    return "sfu_stats/cleaned_data/all_stats.json"


def get_sfu_age_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up age distribution of SFU.
    :return: A Stat containing headcount by age.
    """
    stat_file = get_raw_path("headcount/age_distribution_ST20.xlsx")

    """ My original unconventional way of cleaning data.
    stats = pd.read_excel(stat_file, sheet_name="pivot table", header=8, usecols="A:B")
    x_lbl = "Age"
    y_lbl = "Count"
    x_values = []
    y_values = []

    for i, row in stats.iterrows():
        age = row["Age"]
        count = nan_to_int(row[2023])

        if age > 90:
            break

        x_values.append(age)
        y_values.append(count)

    df = pd.DataFrame({
        x_lbl: x_values,
        y_lbl: y_values
    })
    """
    # My improved way of cleaning data that I can now do after my experience with completing this project.
    df = pd.read_excel(stat_file, sheet_name="pivot table", header=8)
    df = df[["Age", 2023]]
    df = df.rename(columns={2023: "Count"})
    df = df[df.apply(lambda x: True if type(x["Age"]) is int and x["Age"] <= 90 else False, axis=1)]
    df = df.infer_objects(copy=False).fillna(0)

    stat = Stat(
        label="Age Distribution",
        graph_title="Total Count by Age (FALL 2023)",
        df=df
    )

    return {"age-count": stat}


def get_sfu_programs() -> pd.DataFrame:
    """
    Gets the cleaned up headcounts from 2023/24 for each SFU program.
    :return: A list of classes representing each program
    """
    stat_file = get_raw_path("headcount/program_distribution_ST04.xlsx")
    stats = pd.read_excel(stat_file, sheet_name="pivot table by gender", header=11, usecols="A:E")

    stats[["Faculty"]] = stats[["Faculty"]].ffill()
    stats.dropna(subset=["Men", "Women", "Not reported"], how="all", inplace=True)
    stats = stats[stats.apply(lambda x: "Total" not in x["Faculty"], axis=1)]
    stats = stats.fillna(0)
    stats.insert(2, "Total", stats["Men"] + stats["Women"] + stats["Not reported"])
    return stats.copy()


def insert_gender_percents(df: pd.DataFrame):
    df.insert(3, "Men percentage",
                         round(df["Men"] / df["Total"] * 100, 2))
    df.insert(5, "Women percentage",
                         round(df["Women"] / df["Total"] * 100, 2))
    df.insert(7, "Not reported percentage",
                         round(df["Not reported"] / df["Total"] * 100, 2))


def get_sfu_faculty_headcounts() -> dict[str: Stat]:
    """
    Gets the headcounts for each SFU faculty.
    :return: Stats of headcounts by faculty for three different measures: total-count, men-count, and women-count
    """
    program_stats = get_sfu_programs()
    program_stats = program_stats.groupby("Faculty")[["Total", "Men", "Women", "Not reported"]].sum()
    program_stats = program_stats.reset_index()
    insert_gender_percents(program_stats)
    total_df = program_stats.sort_values("Total")
    men_df = program_stats[["Faculty", "Men"]].sort_values("Men")
    women_df = program_stats[["Faculty", "Women"]].sort_values("Women")

    return {
        "total-count": Stat(
            label="Total Count",
            graph_title="Total Count by Faculty (2023/24)",
            df=total_df,
            extra_hover_columns=["Men percentage", "Women percentage", "Not reported percentage"]
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Faculty (2023/24)",
            df=men_df
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Faculty (2023/24)",
            df=women_df
        )
    }


def get_sfu_program_headcounts() -> dict[str: Stat]:
    """
    Gets the headcounts for each SFU program.
    :return: Stats of headcounts by program for three different measures: total-count, men-count, and women-count
    """
    program_stats = get_sfu_programs()
    program_stats = program_stats.drop("Faculty", axis=1)
    insert_gender_percents(program_stats)
    total_df = program_stats.sort_values("Total")
    men_df = program_stats[["Program", "Men"]].sort_values("Men")
    women_df = program_stats[["Program", "Women"]].sort_values("Women")
    return {
        "total-count": Stat(
            label="Total Count",
            graph_title="Total Count by Program (2023/24)",
            df=total_df,
            extra_hover_columns=["Men percentage", "Women percentage", "Not reported percentage"]
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Program (2023/24)",
            df=men_df
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Program (2023/24)",
            df=women_df
        )
    }


def get_sfu_new_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up new undergraduates distribution of SFU.
    :return: A Stat containing new undergraduates headcount by faculty for the latest year and a Stat
    to show new headcount growth in the last few years.
    """
    stat_file = get_raw_path("headcount/new_undergrads_distribution_ST12.xlsx")
    stats = pd.read_excel(stat_file, sheet_name="pivot table", header=8)

    # DF: Fiscal Year | Count | Faculty
    combined_df = pd.DataFrame()
    years_left = 10
    current_year = ""
    current_count, current_faculty = [], []

    for i, row in stats.iterrows():
        if years_left <= 0:
            break

        year = row["Fiscal Year"]
        if not pd.isna(year):
            if "Total" in year:
                continue
            current_year = year

        count = row[" Grand Total"]  # Header in Excel file has a space in front.
        faculty = row["Faculty"]

        if faculty == "Unspecified":
            year_arr = [current_year] * len(current_count)
            year_df = pd.DataFrame({
                "Fiscal Year": year_arr,
                "New Undergraduates Count": current_count,
                "Faculty": current_faculty
            })
            current_count, current_faculty = [], []
            combined_df = pd.concat([combined_df, year_df], ignore_index=True)
            years_left -= 1
            continue

        current_count.append(count)
        current_faculty.append(faculty)

    latest_year_df = combined_df[combined_df.apply(lambda r: "2024/25" in r["Fiscal Year"], axis=1)]
    latest_year_df = latest_year_df.sort_values("New Undergraduates Count")
    latest_year_df = latest_year_df.drop(["Fiscal Year"], axis=1)
    new_order = ["Faculty", "New Undergraduates Count"]
    latest_year_df = latest_year_df.reindex(columns=new_order)
    latest_stat = Stat(
        label="New Distribution",
        graph_title="New Undergraduates by Faculty (2024/25)",
        df=latest_year_df
    )

    combined_df["percent"] = (combined_df.groupby("Fiscal Year")["New Undergraduates Count"]
                                 .transform(lambda x: x/x.sum() * 100)).round(2)
    growth_stat = Stat(
        label="New Growth",
        graph_title="Growth of New Undergraduates by Faculty (2024/25)",
        df=combined_df,
        graph_type="grouped_bar",
        extra_hover_columns=["percent"]
    )

    return {
        "new-count": latest_stat,
        "new-growth": growth_stat,
    }


def get_sfu_units_taken() -> dict[str: Stat]:
    """
    Gets the units enrolled headcounts for each season.
    :return: Stats of headcounts by units enrolled for the three different seasons.
    """
    stat_file = get_raw_path("headcount/units_enrolled_distribution_ST32.pdf")

    x_lbl = "Units"
    y_lbl = "Count"
    x_values = []
    seasonal_y_values = {
        "summer-units": [],
        "fall-units": [],
        "spring-units": []
    }

    with pdfplumber.open(stat_file) as pdf_stats:
        for page in pdf_stats.pages:
            for table in page.extract_tables():
                row_count = 0
                for row in table:
                    seasons_row = []
                    for i in range(5, len(row), 6):  # Gets only the 3 most recent season values for each row
                        seasons_row.append(row[i])
                    season_count = 0
                    for season, y_values in seasonal_y_values.items():
                        if row_count == 21:
                            continue
                        if season == "summer-units":  # Only adding x-values once
                            if row_count == 20:
                                x_values.append("20+")
                            elif row_count == 22:
                                x_values.append("Total Students")
                            else:
                                x_values.append(str(row_count))

                        y_values.append(float(seasons_row[season_count].replace(",", "")))
                        season_count += 1
                    row_count += 1

    # Converting percent values of total students to student count
    for season, y_values in seasonal_y_values.items():
        for i in range(len(y_values)):
            y_values[i] *= 1 / 100 * y_values[-1]
        y_values.pop(-1)

    x_values.pop(-1)  # Only removing last x-value once

    stats = {
        "summer-units": Stat(
            label="Summer Units",
            graph_title="Units Taken Distribution (Summer 2024)",
            df=pd.DataFrame({
                x_lbl: x_values,
                y_lbl: seasonal_y_values["summer-units"]
            })
        ),
        "fall-units": Stat(
            label="Fall Units",
            graph_title="Units Taken Distribution (Fall 2024)",
            df=pd.DataFrame({
                x_lbl: x_values,
                y_lbl: seasonal_y_values["fall-units"]
            })
        ),
        "spring-units": Stat(
            label="Spring Units",
            graph_title="Units Taken Distribution (Spring 2025)",
            df=pd.DataFrame({
                x_lbl: x_values,
                y_lbl: seasonal_y_values["spring-units"]
            })
        )
    }

    return stats


def get_sfu_outcomes() -> dict[str: Stat]:
    """
    Gets the outcomes from each concentration in the outcome Baccalaureate Reports of 2024.
    :return: Stats of outcomes for each concentration.
    """
    start = time.time()
    print("Starting Outcome Parsing")

    outcome_dir = get_raw_path("outcome_BaccalaureateReports")
    paths = [os.path.join(outcome_dir, f) for f in os.listdir(outcome_dir)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        outcomes = list(executor.map(get_outcome, paths))
    combined_dataframes = combine_outcomes(outcomes)

    stats = {}
    for category, df in combined_dataframes.items():
        if df is not None and not df.empty:
            cleaned_category = category.title().replace('-', ' ')

            stats[category] = Stat(
                label=cleaned_category,
                graph_title=f"{cleaned_category} Across Concentrations (2024 Survey of 2022 Baccalaureate Graduates)",
                df=df,
                graph_type="grouped_bar"
            )

    end = time.time()
    print(f"Outcome Parsing Elapsed time: {end - start:.2f} seconds")

    return stats


def combine_outcomes(outcomes: dict[str: pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Creates a dictionary for each category that combines each outcome
    of that category from every concentration into a single dataframe.
    :param outcomes: dictionary of outcome dataframes
    :return: Dictionary of combined outcome dataframes.
    """
    category_map = {cat.name.lower(): [] for cat in SFUOutcomeCategories}

    for outcome in outcomes:
        for key, df in outcome.items():
            if df is not None:
                category_map[key].append(df)

    return {
        key.replace("_", "-"): pd.concat(dfs, ignore_index=True) for key, dfs in category_map.items()
    }


def get_outcome(outcome_path: str) -> dict[str: pd.DataFrame]:
    """
    Gets a dictionary of outcome dataframes where each key represents a category and each value it's dataframe.
    This is done by parsing the predictable outcome pdf files for each category.
    :param outcome_path: path to outcome pdf file
    :return: Dictionary of outcome dataframes.
    """
    found = set()
    concentration = ""
    result = {}

    category_triggers = {
        "B.C. Baccalaureate Outcomes": [
            SFUOutcomeCategories.RESPONSE_COUNT
        ],
        "Program Satisfaction": [
            SFUOutcomeCategories.SATISFACTION,
            SFUOutcomeCategories.USEFULNESS,
            SFUOutcomeCategories.REGRET
        ],
        "Employment": [
            SFUOutcomeCategories.EMPLOYMENT_RATE,
            SFUOutcomeCategories.JOB_RELATION,
            SFUOutcomeCategories.INCOME
        ]
    }

    with pdfplumber.open(outcome_path) as pdf_stats:
        for page in pdf_stats.pages:
            text = page.extract_text()
            tables = page.extract_tables()

            for trigger, categories in category_triggers.items():
                if trigger in text and trigger not in found:
                    if trigger == "B.C. Baccalaureate Outcomes":
                        concentration = get_concentration(page)
                    for cat in categories:
                        result[cat.name.lower()] = get_outcome_category(cat, tables, concentration)
                    found.add(trigger)

    return result


def search_list(lst: list[str], keyword) -> int:
    """
    Searches a list of strings against the keyword and returns the index of the first occurrence.
    :param lst: list of strings
    :param keyword: keyword to search for
    :return: Index of keyword. -1 if none found.
    """
    for i in range(len(lst)):
        if keyword == lst[i]:
            return i
    return -1


def get_concentration(page) -> str:
    """
    Gets the concentration from the outcome pdf file.
    :param page: pdf page to extract concentration from
    :return: Cleaned up concentration from the outcome pdf file.
    """
    lines = page.extract_text().split("\n")
    prev_line = search_list(lines, "Simon Fraser University")
    concentration_line = lines[prev_line + 1]
    # The following strips `14.0101: Engineering, general` into `Engineering`
    return concentration_line[concentration_line.find(": ") + 2:] \
        .replace(", general", "").replace(", other", "").title()


class PDFTableValue:
    """
    Represents a value/stat(s) that can be extracted from a pdf table.
    """
    def __init__(self, x_lbl: str, y_lbl: str, x_index: int, y_index: int, prev_value: str, row_count: int,
                 buffer: int = 0):
        self.x_lbl = x_lbl
        self.y_lbl = y_lbl
        self.x_index = x_index
        self.y_index = y_index
        self.prev_value = prev_value
        self.row_count = row_count
        self.buffer = buffer

    def get_values(self, tables, concentration: str) -> pd.DataFrame:
        """
        Gets the values as a dataframe from a pdf table.
        :param tables: pdf table to extract values from
        :param concentration: concentration of the values
        :return: dataframe of values extracted from the table
        """
        x_values = []
        y_values = []
        remaining_rows = self.row_count
        remaining_buffer = self.buffer

        for table in tables:
            started = False
            for row in table:
                if remaining_rows <= 0:
                    if self.x_lbl == SFUOutcomeCategories.INCOME.value.x_lbl:
                        for i in range(len(x_values)):
                            x_values[i] = x_values[i].split(" (full−time) ($)")[0]
                            y_values[i] = float(y_values[i].replace(",", ""))
                    elif self.x_lbl == SFUOutcomeCategories.RESPONSE_COUNT.value.x_lbl:
                        for i in range(len(x_values)):
                            x_values[i] = x_values[i].replace(" and Response Rate", "")
                            y_values[i] = float(y_values[i])

                    return pd.DataFrame({
                        "Concentration": concentration,
                        self.y_lbl: y_values,
                        self.x_lbl: x_values
                    })
                if started:
                    if remaining_buffer > 0:
                        remaining_buffer -= 1
                    else:
                        x_values.append(row[self.x_index])
                        y_values.append(row[self.y_index])
                        remaining_rows -= 1
                if self.prev_value in row:
                    started = True

        return pd.DataFrame()


class SFUOutcomeCategories(Enum):
    """
    Enum that represents SFU outcome categories.
    """
    RESPONSE_COUNT = PDFTableValue(
        x_lbl="Response",
        y_lbl="Count",
        x_index=0,
        y_index=1,
        prev_value="Survey Response Rate:",
        row_count=2
    )
    SATISFACTION = PDFTableValue(
        x_lbl="Satisfaction Level",
        y_lbl="Percentage",
        x_index=0,
        y_index=2,
        prev_value="Program Satisfaction:",
        row_count=4
    )
    USEFULNESS = PDFTableValue(
        x_lbl="Usefulness Level",
        y_lbl="Percentage",
        x_index=0,
        y_index=2,
        prev_value="Usefulness of Knowledge, Skills, and Abilities\nAcquired during Program in Work:",
        row_count=4
    )
    REGRET = PDFTableValue(
        x_lbl="Would Select the Same Program Again",
        y_lbl="Percentage",
        x_index=0,
        y_index=2,
        prev_value="Would select the same program again:",
        row_count=2
    )
    EMPLOYMENT_RATE = PDFTableValue(
        x_lbl="Employment Status",
        y_lbl="Percentage",
        x_index=0,
        y_index=2,
        prev_value="Employment:",
        row_count=2
    )
    JOB_RELATION = PDFTableValue(
        x_lbl="Job Relation",
        y_lbl="Percentage",
        x_index=0,
        y_index=2,
        prev_value="How related is your main job to your program?",
        row_count=4
    )
    INCOME = PDFTableValue(
        x_lbl="Income",
        y_lbl="Amount",
        x_index=0,
        y_index=1,
        prev_value="$100,000 and Above",
        row_count=2,
        buffer=1
    )


def get_outcome_category(category: SFUOutcomeCategories, tables, concentration: str) -> pd.DataFrame:
    return category.value.get_values(tables, concentration)


@functools.cache
def get_all_cleaned_stats() -> dict[str: StatCategory]:
    """
    Gets all the cleaned stats for the website.
    :return: a dict of stat categories
    """
    return {
        "faculty": StatCategory("Headcounts by Faculty", get_sfu_faculty_headcounts()),
        "programs": StatCategory("Headcounts by Programs", get_sfu_program_headcounts()),
        "age": StatCategory("Headcounts by Age", get_sfu_age_headcounts()),
        "new": StatCategory("New Headcounts by Faculty", get_sfu_new_headcounts()),
        "units": StatCategory("Units Enrolled by Season", get_sfu_units_taken()),
        "outcomes": StatCategory("Outcomes by Concentration", get_sfu_outcomes())
    }


def write_all_stats() -> None:
    """
    Writes all the stats into a json file. It first serializes all the stats into dicts
    and then dumps them in a json file named all_stats.json.
    :return: None
    """
    all_stats = get_all_cleaned_stats()
    serializable = {k: v.to_dict() for k, v in all_stats.items()}
    with open(get_output_data_path(), "w") as f:
        json.dump(serializable, f, indent=4)


def load_all_stats() -> dict[str: StatCategory]:
    """
    Loads all the stats from all_stats.json into a dict of stat categories.
    :return: a dict of stat categories
    """
    with open(get_output_data_path(), "r") as f:
        raw = json.load(f)

    result = {}
    for category_key, cat_val in raw.items():
        stats = {
            stat_key: Stat(
                label=stat_val["label"],
                graph_title=stat_val["graph_title"],
                df=pd.DataFrame(stat_val["df"]),
                graph_type=stat_val["graph_type"],
                extra_hover_columns=stat_val.get("extra_hover_columns", [])
            )
            for stat_key, stat_val in cat_val["stats"].items()
        }
        result[category_key] = StatCategory(cat_val["title"], stats)

    return result
