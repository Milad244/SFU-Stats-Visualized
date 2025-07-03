import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, render_template
import plotly.express as px

matplotlib.use('TKAgg')
app = Flask(__name__)


"""Data from https://www.sfu.ca/irp/students.html#outcomes"""


def nan_to_int(num: int) -> int:
    if pd.isna(num):
        return 0
    return num


class Program:
    def __init__(self, faculty: str, program: str, men_count: int, women_count: int, nr_count: int):
        self.faculty = faculty
        self.program = program
        self.men_count = nan_to_int(men_count)
        self.women_count = nan_to_int(women_count)
        self.nr_count = nan_to_int(nr_count)
        self.total_count = self.men_count + self.women_count + self.nr_count


def create_bar_graph(x_values: list, y_values: list, x_lbl: str, y_lbl: str) -> None:
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.bar(x_values, y_values)
    plt.show()


def graph_programs(programs: list[Program]) -> None:
    # Total count graph
    programs.sort(key=lambda l: l.total_count)
    x = [p.program for p in programs]
    y = [p.total_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Total Count")
    # Man count graph
    programs.sort(key=lambda l: l.men_count)
    x = [p.program for p in programs]
    y = [p.men_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Men Count")
    # Women count graph
    programs.sort(key=lambda l: l.women_count)
    x = [p.program for p in programs]
    y = [p.women_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Women Count")


def get_sfu_programs() -> list[Program]:
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

        programs.append(Program(s["Faculty"], s["Program"], s["Men"], s["Women"], s["Not reported"]))
        print(programs[-1].__dict__)

    # graph_programs(programs)
    return programs


@app.route("/")
def main():
    programs = get_sfu_programs()
    programs.sort(key=lambda l: l.total_count)
    x = [p.program for p in programs]
    y = [p.total_count for p in programs]
    df = pd.DataFrame({
        "Program": x,
        "Headcount": y
    })

    fig = px.bar(df, x="Program", y="Headcount", title="Total Headcount by Program")

    graph_html = fig.to_html(full_html=False)
    return render_template("index.html", stat="Total Headcount by Program", graph_html=graph_html)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
