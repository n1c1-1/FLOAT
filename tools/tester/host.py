"""The process that holds the application's window for a usability session (started by session.py, never by hand):
what it does is in host_main.py. The guard below: a process the application spawns (a multiprocessing pool re-imports
the main script under another name) never starts a second window."""
if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).with_name("host_main.py")), run_name="__main__")
