import plotly.express as px
import pandas as pd


class Stat:
    """
    Represents a dataframe. Do not use cache or else keywords search won't work.
    """
    def __init__(self, label: str, graph_title: str, df: pd.DataFrame, graph_type: str = "bar",
                 extra_hover_columns: list[str] = []):
        """
        :param label: stat label
        :param graph_title: graph title
        :param df: dataframe of stats
        :param graph_type: type of graph
        :param extra_hover_columns: list of column data to show on graph hover
        """
        self.label = label
        self.graph_title = graph_title
        self.df = df
        self.x_lbl = df.columns[0]
        self.y_lbl = df.columns[1]
        self.graph_type = graph_type
        self.extra_hover_columns = extra_hover_columns
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
        :return: True if the keyword can be searched in dataframe
        """
        if self.get_filtered_df().empty or self.graph_type != "bar":
            return False
        return True

    def get_filtered_df(self) -> pd.DataFrame:
        """
        Gets the filtered raw_data frame based on the case-insensitive keyword in x-values.
        :return: the filtered raw_data frame
        """
        if self.keyword is None:
            return self.df
        return self.df[self.df.apply(lambda row: self.keyword.lower() in str(row[self.x_lbl]).lower(), axis=1)].copy()

    def get_graph(self) -> str:
        """
        Gets the graph based on the graph type and returns its html.
        :return: the graph html
        """
        df = self.get_filtered_df()
        if df.empty:
            return '<p class="graph-msg">No Data Found<p>'

        if self.graph_type == "bar":
            return self.get_bar_graph(df)
        elif self.graph_type == "grouped_bar":
            return self.get_grouped_bar_graph(df)

        return ""

    def combine_hover_template(self, start: list[str], custom_data_index_start: int = 1) -> str:
        """
        Combines starting hover template and extra hover columns into one hover template.
        :param start: the starting hover template
        :param custom_data_index_start: put 0 when you have not used any custom data in start and add 1 for everytime
        you did
        :return: the combined hover template
        """
        hover_template_parts = start
        for i, col in enumerate(self.extra_hover_columns, start=custom_data_index_start):
            extra_back = ""
            if "percent" in col.lower():
                extra_back = "%"
            hover_template_parts.append(f"{col.title()}: %{{customdata[{i}]}}{extra_back}")
        return "<br>".join(hover_template_parts)

    def update_layout(self, fig, hover_template):
        """
        Updates the layout of the figure. Gives a background to the entire plot and graph.
        Also, applies the hover template.
        :param fig: the figure to update
        :param hover_template: the hover template
        :return: None
        """
        fig.update_layout(
            paper_bgcolor='#f8f3ff',
            plot_bgcolor='#f8f3ff'
        )

        fig.update_traces(
            hovertemplate=hover_template
        )

    def get_bar_graph(self, df: pd.DataFrame) -> str:
        """
        Creates a bar graph of the stat.
        :param df: the filtered dataframe
        :return: the bar graph html
        """
        y_total = df[self.y_lbl].sum()
        df['percentage'] = (df[self.y_lbl] / y_total * 100).round(2)

        custom_data_cols = ["percentage"] + self.extra_hover_columns
        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, title=self.graph_title, custom_data=custom_data_cols)

        hover_template_start = [
            f"{self.x_lbl}: %{{x}}",
            f"{self.y_lbl}: %{{y}}",
            "Percent: %{customdata[0]}%"
        ]
        hover_template = self.combine_hover_template(hover_template_start)

        self.update_layout(fig, hover_template)

        fig.update_traces(
            marker_color='rgb(179, 0, 0)'
        )

        fig.add_annotation(
            yref="paper",
            yanchor="bottom",
            y=1.025,
            text=self.get_stat_features(),
            xref="paper",
            xanchor="center",
            x=0.5,
            showarrow=False,
            font=dict(size=18)
        )

        return fig.to_html(full_html=False)

    def get_grouped_bar_graph(self, df: pd.DataFrame) -> str:
        """
        Creates a grouped bar graph of the stat.
        :param df: the filtered dataframe
        :return: the grouped bar graph html
        """
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
            "rgb(50, 0, 100)",
            "rgb(0, 0, 150)",
            "rgb(135, 0, 110)",
            "rgb(120, 45, 150)"
        ]

        custom_data_cols = [color_col] + self.extra_hover_columns

        fig = px.bar(df, x=self.x_lbl, y=self.y_lbl, color=color_col, title=self.graph_title,
                     custom_data=custom_data_cols, color_discrete_sequence=shades)

        hover_template_start = [
            f"{self.x_lbl}: %{{x}}",
            f"{color_col}: %{{customdata[0]}}",
            f"{self.y_lbl}: %{{y}}{back}"
        ]
        hover_template = self.combine_hover_template(hover_template_start)

        self.update_layout(fig, hover_template)

        fig.update_traces(
            hoverlabel=dict(namelength=0) # Removes default ugly plotly hover
        )

        return fig.to_html(full_html=False)

    def get_stat_features(self) -> str:
        """
        Gets the stat features containing the total, mean, median, and mode of the stat.
        :return: the stat features
        """
        return (f"Total: {self.get_total()} | Mean: {self.get_mean()} <br>"
                f"Median: {self.get_median()} | Mode: {self.get_mode_str()}")

    def get_total(self) -> float:
        """
        Gets the total of all y-values added up.
        :return: the total
        """
        return self.get_filtered_df()[self.y_lbl].sum().round(2)

    def get_mean(self) -> float:
        """
        Gets the mean (average) of all y-values.
        :return: the mean
        """
        return self.get_filtered_df()[self.y_lbl].mean().round(2)

    def get_median(self) -> float:
        """
        Gets the median (middle) of all y-values in order.
        If even number of y-values then gets the average of the two in the middle.
        :return: the median
        """
        return self.get_filtered_df()[self.y_lbl].median().round(2)

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

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "graph_title": self.graph_title,
            "df": self.df.to_dict("records"),
            "graph_type": self.graph_type,
            "extra_hover_columns": self.extra_hover_columns
        }


class StatCategory:
    """
    Represents a category of stats.
    """
    def __init__(self, title: str, stats: dict[str: Stat]):
        """
        :param title: the title of the category
        :param stats: dict of sub-stat-label: Stat
        """
        self.title = title
        self.stats = stats

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "stats": {k: v.to_dict() for k, v in self.stats.items()}
        }
