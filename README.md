# GAN: Enjoy Nice Interface Easy (GENIE)
![genie_interface](https://user-images.githubusercontent.com/105862211/173179551-0617b592-570c-4b79-bcee-25f66a1962d9.png)
Tool for monitoring and manage(in future) process of trainig GANs in Colab/Jupiter env

Live tutorial: https://colab.research.google.com/drive/1F8lwUuRlzVHYtKl_x0Lf5ZUIruHKHl5R?usp=sharing

# *Key features*
- Real-time monitoring for losses and metrics (stored each epoch):
  - text table, last 10 epochs
  - graph view with customizable epoch range and metric set

- Generator preview player:
  - update & show last preview
  - seek & show any previous previews
  - player for preview from any position right during training process and after
  - creating gif animation

- Store all data in .csv and png files, you can choose endpoint as local or google drive at once or transfer after training

- Secondary trainings do not overwrite the data of the previous one, but are added to them, the numbering of epochs continues. You can change any parameters except *metrics_for_monitoring*.

- Interface:
  - indicating whole training progress
  - button for manual interruption training process
