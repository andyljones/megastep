import os

def setup(app):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    app.add_html_theme(
        'doctheme', current_dir)

    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }