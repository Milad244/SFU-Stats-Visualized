from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import plotly.express as px
import functools

matplotlib.use('TKAgg')
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
    Represents an (x,y) graph
    """
    def __init__(self, label: str, graph_title: str, x_lbl: str, x_values: list, y_lbl: str, y_values: list):
        self.label = label
        self.graph_title = graph_title
        self.x_lbl = x_lbl
        self.x_values = x_values
        self.y_lbl = y_lbl
        self.y_values = y_values

    @functools.cache
    def get_bar_graph(self) -> str:
        df = pd.DataFrame({
            self.x_lbl: self.x_values,
            self.y_lbl: self.y_values
        })
        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, title=self.graph_title)
        return fig.to_html(full_html=False)


def get_sfu_age_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up age distribution of SFU.
    :return: A Stat containing headcount by age
    """

    stat_file = "data/headcount/ST20.db.xlsx"
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


class ProgramHeadcount:
    """
    Represents the headcounts from each SFU program.
    """
    def __init__(self, faculty: str, program: str, men_count: int, women_count: int, nr_count: int):
        self.faculty = faculty
        self.program = program
        self.men_count = nan_to_int(men_count)
        self.women_count = nan_to_int(women_count)
        self.nr_count = nan_to_int(nr_count)
        self.total_count = self.men_count + self.women_count + self.nr_count


def get_sfu_program_headcounts() -> dict[str: Stat]:
    """
    Gets the cleaned up headcounts from 2024 for each SFU program.
    :return: Stats of headcounts by program for three different measures: total-count, men-count, and women-count
    """
    stat_file = "data/headcount/ST04.db.xlsx"
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

        programs.append(ProgramHeadcount(s["Faculty"], s["Program"], s["Men"], s["Women"], s["Not reported"]))
        # print(programs[-1].__dict__)

    # TODO: Make this much cleaner
    x_lbl = "Program"
    y_lbl = "Count"
    total_ordered = sorted(programs, key=lambda p: p.total_count)
    total_x = [p.program for p in total_ordered]
    total_y = [p.total_count for p in total_ordered]
    men_ordered = sorted(programs, key=lambda p: p.men_count)
    men_x = [p.program for p in men_ordered]
    men_y = [p.men_count for p in men_ordered]
    women_ordered = sorted(programs, key=lambda p: p.women_count)
    women_x = [p.program for p in women_ordered]
    women_y = [p.women_count for p in women_ordered]
    return {
        "total-count": Stat(
            label="Total Count",
            graph_title="Total Count by Program",
            x_lbl=x_lbl,
            x_values=total_x,
            y_lbl=y_lbl,
            y_values=total_y
        ),
        "men-count": Stat(
            label="Men Count",
            graph_title="Men Count by Program",
            x_lbl=x_lbl,
            x_values=men_x,
            y_lbl=y_lbl,
            y_values=men_y
        ),
        "women-count": Stat(
            label="Women Count",
            graph_title="Women Count by Program",
            x_lbl=x_lbl,
            x_values=women_x,
            y_lbl=y_lbl,
            y_values=women_y
        )
    }


@functools.cache
def get_all_stats() -> list[dict[str: Stat]]:  # TODO: Make it dict instead to group stats in main() easier
    """
    Gets all the stats for the website
    :return: a list of stat categories
    """
    headcount_by_program_stats = get_sfu_program_headcounts()
    headcount_by_age_stats = get_sfu_age_headcounts()
    return [headcount_by_program_stats, headcount_by_age_stats]


@app.route("/")
def main():
    """
    Creates our index(initial) page. This page will contain the selections for what data to see, the selected data, and
    all other navigation.
    :return: the rendering string
    """

    view = request.args.get("view")
    graph_html = None

    all_stats = get_all_stats()

    for stats in all_stats:
        if view in stats:
            stat = stats[view]
            graph_html = stat.get_bar_graph()

    buttons = []
    for stats in all_stats:
        for slug, stat in stats.items():
            buttons.append({
                "label": stat.label,
                "query_url": f"/?view={slug}"
            })

    buttons.append({"label": "Source", "external_url": "https://www.sfu.ca/irp/students.html"})

    return render_template("index.html", title="SFU Statistics Visualized", buttons=buttons, graph_html=graph_html)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Runs the flask site on local host port 8000 in debug mode.
    # https://flask.palletsprojects.com/en/stable/quickstart/ to understand flask
