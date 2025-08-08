from sfu_stats.data_processing.stats_cleaner import write_all_stats, load_all_stats


def write_then_read_stats():
    write_all_stats()
    return load_all_stats()


def get_all_stats(new: bool = False):
    """
    Gets all the stats by parsing the all_stats json file.
    :param new: if true then cleans and writes the stats before reading them
    :return: a dict of stat categories
    """
    if new:
        return write_then_read_stats()

    try:
        return load_all_stats()
    except FileNotFoundError:
        try:
            return write_then_read_stats()
        except Exception as e:
            print(f"Failed to write/load stats after initial failure: {e}")
    except Exception as e:
        print(f"Failed to load stats: {e}")

    return None


if __name__ == "__main__":
    print(get_all_stats())
