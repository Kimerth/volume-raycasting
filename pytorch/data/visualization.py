from torchio.data.subject import Subject
from torchio.datasets.fpg import GIF_COLORS
import os


def plot_subject(subject: Subject, save_plot_path: str):
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
    subject.plot(
        output_path=os.path.join(
            save_plot_path,
            subject['subject_id']
        )
    )