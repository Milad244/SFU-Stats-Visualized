from flask import Flask, render_template, request, redirect, url_for
from sfu_stats import get_all_stats

app = Flask(__name__)


@app.route("/")
def main():
    """
    Creates our index page. This controls all the content shown in the site including the cleaned_data selection options,
    the selected cleaned_data, and all other navigation.
    :return: the rendering string
    """
    category = request.args.get("category")
    stat = request.args.get("view")
    keyword = request.args.get("keyword")
    graph_html = None
    graph_features = {}

    all_stats = get_all_stats(new=True)

    none_cat = False
    if category is None:
        none_cat = True
        # return redirect(url_for("main", category="faculty"))
    else:
        for cat_slug, stat_cat in all_stats.items():
            if category == cat_slug:
                if stat is None:
                    first_stat = list(stat_cat.stats.items())[0][0]
                    return redirect(url_for("main", category=cat_slug, view=first_stat))
                try:
                    gotten_stat = stat_cat.stats[stat]
                except KeyError:
                    print("Could not find view from the category:", category)
                else:
                    gotten_stat.set_keyword(keyword)
                    graph_html = gotten_stat.get_graph()
                break

    categories = {}
    for cat_slug, stat_cat in all_stats.items():
        categories[cat_slug] = {"title": stat_cat.title, "query_url": f"/?category={cat_slug}", "buttons": []}
        for stat_slug, stat in stat_cat.stats.items():
            categories[cat_slug]["buttons"].append({
                "label": stat.label,
                "query_url": f"/?category={cat_slug}&view={stat_slug}",
                "stat_slug": stat_slug
            })

    buttons = [
        {"label": "GitHub", "external_url": "https://github.com/Milad244/SFU-Stats-Visualized"},
        {"label": "Source", "external_url": "https://www.sfu.ca/irp/students.html"}
    ]
    return render_template("index.html", title="SFU Undergraduate Statistics Visualized",
                           none_cat=none_cat, categories=categories, buttons=buttons,
                           graph_html=graph_html, graph_features=graph_features)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Runs the flask site on local host port 8000 in debug mode.
    # host='0.0.0.0' makes the server accessible from other devices on the same network
    # https://flask.palletsprojects.com/en/stable/quickstart/ to understand flask
