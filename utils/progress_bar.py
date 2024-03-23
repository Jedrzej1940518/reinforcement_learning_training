
from rich.progress import Progress

progress = Progress()

def print_progress_bar(total=100, current_progress=None, task_description="Training..."):
    global progress
    if "task" not in globals():
        # Start the progress bar if it's not already started
        global task
        progress.start()
        task = progress.add_task(task_description, total=total)
    if current_progress is not None:
        progress.update(task, completed=current_progress)
    if progress.finished:
        progress.stop()