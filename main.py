import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
import pandas as pd
import numpy as np

"""Data from https://www.sfu.ca/irp/students.html#outcomes"""


def nan_to_int(num: int) -> int:
    if pd.isna(num):
        return 0
    return num


class Program:
    def __init__(self, faculty: str, program: str, man_count: int, women_count: int, na_count: int):
        self.faculty = faculty
        self.program = program
        self.man_count = nan_to_int(man_count)
        self.women_count = nan_to_int(women_count)
        self.na_count = nan_to_int(na_count)
        self.total_count = self.man_count + self.women_count + self.na_count


def create_bar_graph(x_values: list, y_values: list, x_lbl: str, y_lbl: str) -> None:
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.bar(x_values, y_values)
    plt.show()


def create_grouped_bar_graph(x_values: list, y_values: list, x_lbl: str, y_lbl: str) -> None:
    pass


def graph_programs(programs: list[Program]) -> None:
    # Total count graph
    programs.sort(key=lambda l: l.total_count)
    x = [p.program for p in programs]
    y = [p.total_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Total Count")
    # Man count graph
    programs.sort(key=lambda l: l.man_count)
    x = [p.program for p in programs]
    y = [p.man_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Man Count")
    # Women count graph
    programs.sort(key=lambda l: l.women_count)
    x = [p.program for p in programs]
    y = [p.women_count for p in programs]
    create_bar_graph(x, y, x_lbl="Program", y_lbl="Women Count")


def get_sfu_programs() -> None:
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

    graph_programs(programs)


if __name__ == '__main__':
    get_sfu_programs()

