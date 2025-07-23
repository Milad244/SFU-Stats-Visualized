import os
import time
from enum import Enum
import pandas as pd
from flask import Flask, render_template, request
import plotly.express as px
import functools
import pdfplumber
import concurrent.futures

app = Flask(__name__)


def nan_to_int(num: int) -> int:
    """
    Converts a NaN value into 0. If not NaN, then leaves as is.
    :param num: the number to convert
    :return: the converted number
    """
    if pd.isna(num):
        return 0
    return num


def get_ordered_x_y(count_dict: dict[str: int]) -> (list, list):
    """
    Turns a dictionary into two lists that are in the same order.
    :param count_dict: the dictionary to convert into two lists
    :return: a tuple with the first value as x-values and the second value as y-values
    """
    ordered_x = []
    ordered_y = []
    for key, value in count_dict.items():
        ordered_x.append(key)
        ordered_y.append(value)

    n = len(ordered_y)
    for i in range(n):
        smallest_index = i
        for j in range(i + 1, n):
            if ordered_y[j] < ordered_y[smallest_index]:
                smallest_index = j
        ordered_y[i], ordered_y[smallest_index] = ordered_y[smallest_index], ordered_y[i]
        ordered_x[i], ordered_x[smallest_index] = ordered_x[smallest_index], ordered_x[i]

    return ordered_x, ordered_y


class Stat:
    """
    Represents a dataframe. Do not use cache or else keywords search won't work.
    """
    def __init__(self, label: str, graph_title: str, df: pd.DataFrame, graph_type: str = "bar"):
        self.label = label
        self.graph_title = graph_title
        self.df = df
        self.x_lbl = df.columns[0]
        self.y_lbl = df.columns[1]
        self.graph_type = graph_type
        self.keyword = None

    def set_keyword(self, keyword) -> None:
        """
        Sets the keyword for searching in dataframe.
        :param keyword: the keyword to search for
        :return: None
        """
        self.keyword = keyword

    def can_show_stats(self) -> bool:
        """
        Checks if the keyword can be searched in dataframe.
        :return: True if the keyword can be searched in dataframe.
        """
        if self.get_filtered_df().empty or self.graph_type != "bar":
            return False
        return True

    def get_filtered_df(self) -> pd.DataFrame:
        """
        Gets the filtered data frame based on the case-insensitive keyword in x-values.
        :return: the filtered data frame
        """
        if self.keyword is None:
            return self.df
        return self.df[self.df.apply(lambda row: self.keyword.lower() in str(row[self.x_lbl]).lower(), axis=1)]

    def get_graph(self) -> str:
        """
        Gets the graph based on the graph type and returns its html.
        :return: the graph html
        """
        if self.graph_type == "bar":
            return self.get_bar_graph()
        elif self.graph_type == "grouped_bar":
            return self.get_grouped_bar_graph()

        return ""

    def get_bar_graph(self) -> str:
        """
        Creates a bar graph.
        :return: the bar graph html
        """
        df = self.get_filtered_df()

        if df.empty:
            return '<p class="graph-msg">No Data Found<p>'

        y_total = df[self.y_lbl].sum()
        df['percentage'] = (df[self.y_lbl] / y_total * 100).round(2)

        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, title=self.graph_title)

        fig.update_traces(
            marker_color='rgb(179, 0, 0)',
            hovertemplate=(
                    f"{self.x_lbl}: %{{x}}<br>" +
                    f"{self.y_lbl}: %{{y}}<br>" +
                    "Percent: %{customdata[0]}%"
            ),
            customdata=df[['percentage']].values
        )

        fig.update_layout(
            paper_bgcolor='rgb(245, 245, 245)',
        )

        return fig.to_html(full_html=False)

    def get_grouped_bar_graph(self):
        """
        Creates a grouped bar graph.
        :return: the grouped bar graph html
        """
        df = self.get_filtered_df().copy()

        if df.empty:
            return '<p class="graph-msg">No Data Found<p>'

        color_col = df.columns[2]

        back = ""
        if df[self.y_lbl].dtype == object and df[self.y_lbl].str.contains('%').any():
            df[self.y_lbl] = df[self.y_lbl].str.rstrip('%').astype(float)
            back = "%"

        shades = [
            "rgb(179, 0, 0)",
            "rgb(143, 0, 25)",
            "rgb(107, 0, 50)",
            "rgb(79, 0, 75)",
            "rgb(50, 0, 100)"
        ]

        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, color=color_col, title=self.graph_title, custom_data=[color_col],
                     color_discrete_sequence=shades)

        fig.update_traces(
            hovertemplate=(
                    f"{self.x_lbl}: %{{x}}<br>" +
                    f"{color_col}: %{{customdata[0]}}<br>" +
                    f"{self.y_lbl}: %{{y}}{back}"
            ),
            hoverlabel=dict(namelength=0)
        )

        fig.update_layout(
            paper_bgcolor='rgb(245, 245, 245)',
        )

        return fig.to_html(full_html=False)

    def get_total(self) -> float:
        """
        Gets the total of all y-values added up.
        :return: the total
        """
        return self.get_filtered_df()[self.y_lbl].sum()

    def get_mean(self) -> float:
        """
        Gets the mean (average) of all y-values.
        :return: the mean
        """
        return self.get_filtered_df()[self.y_lbl].mean()

    def get_median(self) -> float:
        """
        Gets the median (middle) of all y-values in order.
        If even number of y-values then gets the average of the two in the middle.
        :return: the median
        """
        return self.get_filtered_df()[self.y_lbl].median()

    def get_mode_str(self) -> str:
        """
        Gets the mode (most common) of all y-values as a string of numbers in case more than 1 mode found.
        :return: the mode
        """
        df = self.get_filtered_df()
        mode_values = df[self.y_lbl].mode().tolist()
        if not mode_values or len(mode_values) == len(df):
            return "None"
        return ', '.join([f"{v:.2f}" for v in mode_values])


class StatCategory:
    """
    Represents a category of stats.
    """
    def __init__(self, title: str, stat: Stat):
        self.title = title
        self.stat = stat


def get_sfu_age_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up age distribution of SFU.
    :return: A Stat containing headcount by age.
    """
    stat_file = "data/headcount/age_distribution_ST20.xlsx"
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

    stat = Stat(
        label="Age Distribution",
        graph_title="Total Count by Age (FALL 2023)",
        df=df
    )

    return {"age-count": stat}


def get_sfu_new_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up new undergraduates distribution of SFU.
    :return: A Stat containing new undergraduates headcount by faculty.
    """
    stat_file = "data/headcount/new_undergrads_distribution_ST12.xlsx"
    stats = pd.read_excel(stat_file, sheet_name="pivot table", header=8)

    x_lbl = "Faculty"
    y_lbl = "New Undergraduates Count"
    x_y_dict = {}

    for i, row in stats.iterrows():
        faculty = row["Faculty"]
        count = row[" Grand Total"]  # Header in Excel file has a space in front.

        if faculty == "Unspecified":
            break

        x_y_dict[faculty] = count

    x_values, y_values = get_ordered_x_y(x_y_dict)
    df = pd.DataFrame({
        x_lbl: x_values,
        y_lbl: y_values
    })

    stat = Stat(
        label="New Undergraduates Distribution",
        graph_title="New Undergraduates by Faculty (2024/25)",
        df=df
    )

    return {"new-count": stat}


def get_sfu_units_taken() -> dict[str: Stat]:
    """
    Gets the units enrolled headcounts for each season.
    :return: Stats of headcounts by units enrolled for the three different seasons.
    """
    stat_file = "data/headcount/units_enrolled_distribution_ST32.pdf"

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

    outcome_dir = "data/outcome_BaccalaureateReports"
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
    return concentration_line[concentration_line.find(": ") + 2:]\
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


class SFUProgram:
    """
    Represents an SFU program.
    """
    def __init__(self, faculty: str, program: str, men_count: int, women_count: int, nr_count: int):
        self.faculty = faculty
        self.program = program
        self.men_count = nan_to_int(men_count)
        self.women_count = nan_to_int(women_count)
        self.nr_count = nan_to_int(nr_count)
        self.total_count = self.men_count + self.women_count + self.nr_count


def get_sfu_programs() -> list[SFUProgram]:
    """
    Gets the cleaned up headcounts from 2023/24 for each SFU program.
    :return: A list of classes representing each program
    """
    stat_file = "data/headcount/program_distribution_ST04.xlsx"
    stats = pd.read_excel(stat_file, sheet_name="pivot table by gender", header=11, usecols="A:E")

    programs = []
    last_faculty = ""
    for i, row in stats.iterrows():
        if pd.isna(row["Program"]):
            continue

        if pd.isna(row["Faculty"]):
            row["Faculty"] = last_faculty
        else:
            last_faculty = row["Faculty"]

        if pd.isna(row["Men"]) and pd.isna(row["Women"]) and pd.isna(row["Not reported"]):
            continue  # This needs to be after Faculty check to avoid wrong faculties in future

        programs.append(SFUProgram(row["Faculty"], row["Program"], row["Men"], row["Women"], row["Not reported"]))

    return programs


def get_sfu_faculty_headcounts(sfu_programs: list[SFUProgram]) -> dict[str: Stat]:
    """
    Gets the headcounts for each SFU faculty.
    :return: Stats of headcounts by faculty for three different measures: total-count, men-count, and women-count
    """
    total_fac = {}
    men_fac = {}
    women_fac = {}
    for program in sfu_programs:
        total_fac[program.faculty] = total_fac.get(program.faculty, 0) + program.total_count
        men_fac[program.faculty] = men_fac.get(program.faculty, 0) + program.men_count
        women_fac[program.faculty] = women_fac.get(program.faculty, 0) + program.women_count

    x_lbl = "Faculty"
    y_lbl = "Count"
    total_x, total_y = get_ordered_x_y(total_fac)
    men_x, men_y = get_ordered_x_y(men_fac)
    women_x, women_y = get_ordered_x_y(women_fac)

    return {
        "total-count": Stat(
            label="Total Count",
            graph_title="Total Count by Faculty (2023/24)",
            df=pd.DataFrame({
                x_lbl: total_x,
                y_lbl: total_y
            })
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Faculty (2023/24)",
            df=pd.DataFrame({
                x_lbl: men_x,
                y_lbl: men_y
            })
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Faculty (2023/24)",
            df=pd.DataFrame({
                x_lbl: women_x,
                y_lbl: women_y
            })
        )
    }


def get_sfu_program_headcounts(sfu_programs: list[SFUProgram]) -> dict[str: Stat]:
    """
    Gets the headcounts for each SFU program.
    :return: Stats of headcounts by program for three different measures: total-count, men-count, and women-count
    """
    x_lbl = "Program"
    y_lbl = "Count"
    total_ordered = sorted(sfu_programs, key=lambda p: p.total_count)
    total_x = [p.program for p in total_ordered]
    total_y = [p.total_count for p in total_ordered]
    men_ordered = sorted(sfu_programs, key=lambda p: p.men_count)
    men_x = [p.program for p in men_ordered]
    men_y = [p.men_count for p in men_ordered]
    women_ordered = sorted(sfu_programs, key=lambda p: p.women_count)
    women_x = [p.program for p in women_ordered]
    women_y = [p.women_count for p in women_ordered]
    return {
        "total-count": Stat(
            label="Total Count",
            graph_title="Total Count by Program (2023/24)",
            df=pd.DataFrame({
                x_lbl: total_x,
                y_lbl: total_y
            })
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Program (2023/24)",
            df=pd.DataFrame({
                x_lbl: men_x,
                y_lbl: men_y
            })
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Program (2023/24)",
            df=pd.DataFrame({
                x_lbl: women_x,
                y_lbl: women_y
            })
        )
    }


@functools.cache
def get_all_stats() -> dict[str: StatCategory]:
    """
    Gets all the stats for the website.
    :return: a dict of stat categories
    """
    sfu_programs = get_sfu_programs()
    return {
        "faculty": StatCategory("Headcounts by Faculty", get_sfu_faculty_headcounts(sfu_programs)),
        "programs": StatCategory("Headcounts by Programs", get_sfu_program_headcounts(sfu_programs)),
        "age": StatCategory("Headcounts by Age", get_sfu_age_headcounts()),
        "new": StatCategory("New Headcounts by Faculty", get_sfu_new_headcounts()),
        "units": StatCategory("Units Enrolled by Season", get_sfu_units_taken()),
        "outcomes": StatCategory("Outcomes by Concentration", get_sfu_outcomes())
    }


@app.route("/")
def main():
    """
    Creates our index page. This controls all the content shown in the site including the data selection options,
    the selected data, and all other navigation.
    :return: the rendering string
    """
    category = request.args.get("category")
    stat = request.args.get("view")
    keyword = request.args.get("keyword")
    graph_html = None
    graph_features = {}

    all_stats = get_all_stats()

    for cat_slug, stat_cat in all_stats.items():
        if category == cat_slug:
            try:
                gotten_stat = stat_cat.stat[stat]
                gotten_stat.set_keyword(keyword)
                graph_html = gotten_stat.get_graph()
                if gotten_stat.can_show_stats():
                    graph_features["Total"] = round(gotten_stat.get_total(), 2)
                    graph_features["Mean"] = round(gotten_stat.get_mean(), 2)
                    graph_features["Median"] = round(gotten_stat.get_median(), 2)
                    graph_features["Mode"] = gotten_stat.get_mode_str()
            except KeyError:
                print("Could not find view from the category:", category)
            break

    categories = {}
    for cat_slug, stat_cat in all_stats.items():
        categories[cat_slug] = {"title": stat_cat.title, "buttons": []}
        for stat_slug, stat in stat_cat.stat.items():
            categories[cat_slug]["buttons"].append({
                "label": stat.label,
                "query_url": f"/?category={cat_slug}&view={stat_slug}"
            })

    buttons = [
        {"label": "GitHub", "external_url": "https://github.com/Milad244/SFU-Stats-Visualized"},
        {"label": "Source", "external_url": "https://www.sfu.ca/irp/students.html"}
    ]
    return render_template("index.html", title="SFU Undergraduate Statistics Visualized", categories=categories,
                           buttons=buttons, graph_html=graph_html, graph_features=graph_features)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Runs the flask site on local host port 8000 in debug mode.
    # host='0.0.0.0' makes the server accessible from other devices on the same network
    # https://flask.palletsprojects.com/en/stable/quickstart/ to understand flask
