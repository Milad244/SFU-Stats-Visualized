import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import plotly.express as px
import functools

app = Flask(__name__)


"""Data from https://www.sfu.ca/irp/students.html"""


def nan_to_int(num: int) -> int:
    """
    Converts a NAN value into 0. If not NAN, then leaves as is.
    :param num: the number to convert
    :return: the converted number
    """
    if pd.isna(num):
        return 0
    return num


class Stat:
    """
    Represents an (x,y) graph. Do not use cache or keywords search won't work.
    """
    def __init__(self, label: str, graph_title: str, x_lbl: str, x_values: list, y_lbl: str, y_values: list):
        self.label = label
        self.graph_title = graph_title
        self.x_lbl = x_lbl
        self.x_values = x_values
        self.y_lbl = y_lbl
        self.y_values = y_values
        self.keyword = None

    def set_keyword(self, keyword) -> None:
        """
        Sets the keyword for searching in x values.
        :param keyword: the keyword
        :return: None
        """
        self.keyword = keyword

    def get_keyword_values(self) -> (list, list):
        """
        Gets the x (and corresponding y values) that are found within the keyword.
        :return: a tuple with the first value as x-values and the second value as y-values
        """
        if self.keyword is None:
            return self.x_values, self.y_values
        else:
            x_values = []
            y_values = []
            for i in range(len(self.x_values)):
                if self.keyword.lower() in str(self.x_values[i]).lower():
                    x_values.append(self.x_values[i])
                    y_values.append(self.y_values[i])
            return x_values, y_values

    def get_bar_graph(self) -> str:
        """
        Creates a bar graph.
        :return: The bar graph html
        """
        x_values, y_values = self.get_keyword_values()

        if len(x_values) == 0:
            return '<p class="graph-msg">No Data Found<p>'
        df = pd.DataFrame({
            self.x_lbl: x_values,
            self.y_lbl: y_values
        })
        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, title=self.graph_title)
        return fig.to_html(full_html=False)

    def get_total(self) -> float:
        """
        Gets the total of all y-values added up.
        :return: the total
        """
        x_values, y_values = self.get_keyword_values()

        total_values = 0
        for value in y_values:
            total_values += value

        return total_values

    def get_mean(self) -> float:
        """
        Gets the mean (average) of all y-values.
        :return: the mean
        """
        x_values, y_values = self.get_keyword_values()

        return self.get_total() / len(y_values)

    def get_median_str(self) -> str:
        """
        Gets the median (middle) of all y-values.
        If even number of y-values then gets the average of the two in the middle.
        :return: the mean
        """
        x_values, y_values = self.get_keyword_values()

        n = len(y_values)
        if n % 2 == 1:
            middle_n = n // 2  # Not adding 1 since index starts at 0
            median_x = x_values[middle_n]
            median_y = round(y_values[middle_n], 2)
            return f"{median_x} - {median_y}"
        else:
            middle_n1, middle_n2 = int(n / 2 - 1), int(n / 2)  # -1 Since index starts at 0
            median_x1, median_x2 = x_values[middle_n1], x_values[middle_n2]
            median_y = (y_values[middle_n1] + y_values[middle_n2]) / 2
            return f"{median_x1}, {median_x2} - {median_y}"

    def get_mode_str(self) -> str:
        """
        Gets the mode (most common) of all y-values as a string of numbers in case more than 1 mode found.
        :return: the mode
        """
        x_values, y_values = self.get_keyword_values()

        value_to_indexes = {}
        for i in range(len(y_values)):
            value = y_values[i]
            value_to_indexes.get(value, []).append(i)

        mode_indexes = []
        most_indexes = 0
        for value, indexes in value_to_indexes.items():
            if len(indexes) > most_indexes:
                most_indexes = len(indexes)
                mode_indexes = indexes
            elif len(indexes) == most_indexes:
                mode_indexes.extends(indexes)

        if most_indexes == 0:
            return "None"

        string_builder = ""
        for index in mode_indexes:
            string_builder += x_values[index] + ", "

        return string_builder + f"- {most_indexes}"


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
    :return: A Stat containing headcount by age
    """

    stat_file = "data/headcount/age_distribution_ST20.xlsx"
    stats = pd.read_excel(stat_file, sheet_name="pivot table", header=8, usecols="A:B")

    stat = Stat(
            label="Age Distribution",
            graph_title="Total Count by Age (FALL 2023)",
            x_lbl="Age",
            x_values=[],
            y_lbl="Count",
            y_values=[]
        )

    for i, s in stats.iterrows():
        age = s["Age"]
        count = nan_to_int(s[2023])

        if age > 90:
            break

        stat.x_values.append(age)
        stat.y_values.append(count)

    return {"age-count": stat}


def get_sfu_new_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up new undergraduates distribution of SFU.
    :return: A Stat containing new undergraduates headcount by faculty
    """

    stat_file = "data/headcount/new_undergrads_distribution_ST12.xlsx"
    stats = pd.read_excel(stat_file, sheet_name="pivot table", header=8)

    stat = Stat(
        label="New Undergraduates Distribution",
        graph_title="New Undergraduates by Faculty (2024/25)",
        x_lbl="Faculty",
        x_values=[],
        y_lbl="New Undergraduates Count",
        y_values=[]
    )

    x_y_dict = {}
    for i, s in stats.iterrows():
        faculty = s["Faculty"]
        count = s[" Grand Total"]

        if faculty == "Unspecified":
            break

        x_y_dict[faculty] = count

    stat.x_values, stat.y_values = get_ordered_x_y(x_y_dict)
    return {"new-count": stat}


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
    for i, s in stats.iterrows():
        if pd.isna(s["Program"]):
            continue

        if pd.isna(s["Faculty"]):
            s["Faculty"] = last_faculty
        else:
            last_faculty = s["Faculty"]

        if pd.isna(s["Men"]) and pd.isna(s["Women"]) and pd.isna(s["Not reported"]):
            continue  # This needs to be after Faculty check to avoid wrong faculties in future

        programs.append(SFUProgram(s["Faculty"], s["Program"], s["Men"], s["Women"], s["Not reported"]))

    return programs


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
            x_lbl=x_lbl,
            x_values=total_x,
            y_lbl=y_lbl,
            y_values=total_y
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Faculty (2023/24)",
            x_lbl=x_lbl,
            x_values=men_x,
            y_lbl=y_lbl,
            y_values=men_y
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Faculty (2023/24)",
            x_lbl=x_lbl,
            x_values=women_x,
            y_lbl=y_lbl,
            y_values=women_y
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
            x_lbl=x_lbl,
            x_values=total_x,
            y_lbl=y_lbl,
            y_values=total_y
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Program (2023/24)",
            x_lbl=x_lbl,
            x_values=men_x,
            y_lbl=y_lbl,
            y_values=men_y
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Program (2023/24)",
            x_lbl=x_lbl,
            x_values=women_x,
            y_lbl=y_lbl,
            y_values=women_y
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
        "new": StatCategory("New Undergraduates by Faculty", get_sfu_new_headcounts()),
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
                graph_html = gotten_stat.get_bar_graph()
                graph_features["Total"] = round(gotten_stat.get_total(), 2)
                graph_features["Mean"] = round(gotten_stat.get_mean(), 2)
                graph_features["Median"] = gotten_stat.get_median_str()
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

    buttons = []
    buttons.append({"label": "Source", "external_url": "https://www.sfu.ca/irp/students.html"})
    return render_template("index.html", title="SFU Statistics Visualized", categories=categories, buttons=buttons,
                           graph_html=graph_html, graph_features=graph_features)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Runs the flask site on local host port 8000 in debug mode.
    # host='0.0.0.0' makes the server accessible from other devices on the same network
    # https://flask.palletsprojects.com/en/stable/quickstart/ to understand flask
