# Driving the application for a usability session

You use the application as a person would: you see its window (a screenshot after each gesture) and you act with
the mouse and the keyboard. Every gesture is one shell command; its answer is the path of a new screenshot and the
words a person would read on screen at that moment.

Below, `PY` is the Python interpreter you were given, `TOOL` is this folder, `APP` the application's folder and
`S` your session folder (one per session, for example `<your folder>/session1`).

## Start a session

    PY TOOL/session.py --app APP --dir S --size 1920x1080 --lang en

- `--size 1250x685` gives a small laptop screen, `1920x1080` a desktop one; the window fills that screen.
- `--lang fr` starts the application in French.
- `--project <folder or project file>` starts with that project open (as when a project is opened from the
  file explorer). Otherwise you start on the Home screen, as after installing the application.

The command returns once the window is ready and prints the first screenshot. The application keeps running in the
background until you send `quit` (or after 30 minutes without any command). The session has its own settings: it
starts with no recent projects, and nothing you change reaches anyone else.

## Send a gesture

    PY TOOL/do.py S <command> [arguments]

The answer looks like this:

    S/shots/0012.png
    Window: FLOAT · My project *
    On top: dialog "Name the project"
    Tooltip: …
    Notice: 107 lights added to "Session 1" · Undo
    Status bar: 1 workflow · to run: 5 min  |  Log [button]
    Note: …

Open the PNG to see the screen. `Window` is the window's title, `On top` a dialog, menu or list open over the
window, `Notice` a message shown in the window, `Status bar` the bar at the bottom; `Note` lines come from the
session itself (a file dialog answered, a gesture that could not be done).

## Coordinates

All coordinates are pixels of the last screenshot, from its top-left corner (x to the right, y down). Read them off
the image: a button at x 1203, y 40 in the PNG is clicked with `click 1203 40`. Menus, lists and dialogs appear in
the screenshot where they are on screen and are clicked the same way. A dialog's title bar is drawn above it.

## Commands

| Command | What it does |
|---|---|
| `shot` | a new screenshot, nothing else |
| `move X Y` | moves the mouse there and stays still about a second (tooltips appear) |
| `click X Y` | left click; `click X Y right` right click (context menu); `click X Y left double` double-click; `middle` also exists |
| `drag X1 Y1 X2 Y2 [STEPS]` | press at X1 Y1, move slowly to X2 Y2 (15 steps unless STEPS), release: sliders, moving blocks, dragging a block from the palette |
| `press X Y` / `release X Y` | a drag in several commands: `press`, then any number of `move` (the button stays down, and you see where things would land), then `release` |
| `wheel X Y DELTA [ctrl]` | mouse wheel at that point; 120 is one notch up (or forward), -120 one notch down; `ctrl` holds Ctrl (zoom) |
| `key COMBO` | one key or shortcut: `enter`, `escape`, `tab`, `delete`, `backspace`, `f2`, `f5`, `ctrl+z`, `ctrl+shift+z`, `up`, `a`… |
| `type TEXT` | types the text where the keyboard focus is (click a field first) |
| `choose_file PATH [PATH…]` | prepares the answer of the NEXT file dialog the application opens (a file to open, files to add, a file to save) |
| `choose_folder PATH` | the same for a dialog that asks for a folder |
| `drop_files X Y PATH [PATH…]` | drops files or folders from the file explorer at that point of the window |
| `resize WxH` | resizes the window (not larger than the size the session started with) |
| `open PROJECT` | opens a project without going through the menus (use File > Open project… normally) |
| `mark "TEXT"` | writes a milestone in your journal ("project created", "first run started"…) |
| `wait SECONDS` | waits (up to 120 s), then a screenshot: for a run or a computation in progress |
| `quit` | closes the application and stops any run it started; the session is over |

### File dialogs

The application's file dialogs are not shown: you answer them beforehand. First send `choose_file` (or
`choose_folder`) with the path you would pick, then do the gesture that opens the dialog (a menu item, a "Browse…"
or "Add files…" button). The answer is a `Note:` line. If a dialog opens with no answer prepared, it is cancelled
and a `Note:` says so: prepare the answer and do the gesture again. Paths can be written with `/`.

## Your journal

Every command is written to `S/journal.jsonl` with its time and screenshot. To see your figures (number of commands
and clicks, time from the start to each mark):

    PY TOOL/stats.py S

Put a `mark` at each milestone of your task, so that the time each step took can be read afterwards.

## Folders

- `S/shots/`: your screenshots, numbered in order.
- `S/projects/`: where a new project is proposed by default when the application asks where to save it.
- `S/settings/`: the application's settings for this session only.

## Good to know

- A gesture answers once the screen has stopped changing (a few seconds at most). Something slow (a run, an image
  being computed) may still be going on: `wait 10` and look again.
- Hover with `move` to read a tooltip before you click something you do not understand.
- If the session does not answer any more, start it again with the same `--dir`: your screenshots and journal go on.
