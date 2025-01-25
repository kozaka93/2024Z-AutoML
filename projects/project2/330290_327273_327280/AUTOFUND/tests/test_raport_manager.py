from numerai_automl.raport_manager.raport_manager import RaportManager
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Creating sample plots
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [4, 5, 6])
    ax1.set_title("Plot 1")

    fig2, ax2 = plt.subplots()
    ax2.bar([1, 2, 3], [3, 4, 5])
    ax2.set_title("Plot 2")

    # Using the class
    figures = [fig1, fig2]
    exporter = RaportManager(figures)
    exporter.generate_html("figures_with_info.html")

    print("HTML file with system information and plots has been generated.")
