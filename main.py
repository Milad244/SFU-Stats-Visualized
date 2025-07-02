import matplotlib.pyplot as plt
import pandas as pd

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


def create_graph(programs: list[Program]):
    programs.sort(key=lambda l: l.total_count)

    plt.xlabel("Program")
    plt.ylabel("Headcount")

    x = [p.program for p in programs]
    y = [p.total_count for p in programs]

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.bar(x, y)
    plt.show()


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

    return programs


if __name__ == '__main__':
    create_graph(get_sfu_programs())

