from matplotlib.figure import Figure
from base64 import b64encode
from io import BytesIO
import platform
import psutil

class RaportManager:
    """
    Class for creating and managing raports.
    It will create a raport from plots created by classes that inherit from AbstractPlot.
    """
    def __init__(self, figures):
        """Initializes the object with a list of Figure objects."""
        if not all(isinstance(fig, Figure) for fig in figures):
            raise TypeError("All elements must be of type matplotlib.figure.Figure")
        self.figures = figures

    def get_system_info(self):
        """Collects system and hardware information."""
        info = {
            "System": platform.system(),
            "Version": platform.version(),
            "Processor": platform.processor(),
            "Cores": psutil.cpu_count(logical=True),
            "RAM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB"
        }
        return info

    def generate_html(self, output_file: str):
        """Generates an HTML file containing system information and plots."""
        system_info = self.get_system_info()
        info_html = "<h1>System Information</h1><ul>"
        for key, value in system_info.items():
            info_html += f"<li><strong>{key}:</strong> {value}</li>"
        info_html += "</ul>"

        figures_html = []
        for fig in self.figures:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = b64encode(buf.read()).decode('utf-8')
            buf.close()

            img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="Figure" style="max-width:100%; height:auto; margin-bottom:20px;" />'
            figures_html.append(img_tag)

        content = info_html + "\n".join(figures_html)
        html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Figures with System Info</title>
            </head>
            <body>
                {content}
            </body>
            </html>
            """

        html_content = html_template.format(content=content)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
