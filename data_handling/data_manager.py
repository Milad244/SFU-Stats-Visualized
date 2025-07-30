from data_handling.data_cleaner import write_all_stats, load_all_stats

def read_then_write_stats():
    write_all_stats()
    return load_all_stats()

def get_all_stats(new: bool = False):
    if new:
        return read_then_write_stats()

    try:
        return load_all_stats()
    except FileNotFoundError:
        try:
            return read_then_write_stats()
        except Exception as e:
            print(f"Failed to write/load stats after initial failure: {e}")
    except Exception as e:
        print(f"Failed to load stats: {e}")

    return None

if __name__ == "__main__":
    print(get_all_stats())
