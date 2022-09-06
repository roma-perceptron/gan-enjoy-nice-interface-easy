# GAN: Enjoy Nice Interface Easy (GENIE)
![image](https://user-images.githubusercontent.com/105862211/188597941-2deb9561-cc43-4b20-84ab-44505b8cef66.png)

Tool for monitoring and manage process of trainig GANs in Colab/Jupiter env

Live tutorial: https://colab.research.google.com/drive/1F8lwUuRlzVHYtKl_x0Lf5ZUIruHKHl5R?usp=sharing

# *Key features*
- Real-time monitoring for losses and metrics (stored each epoch):
  - text table, last 10 epochs
  - graph view with customizable epoch range and metric set
  
- Managing hyper-parameters in real-time
  - write code for change learning_rate and other
  - wrap it in functions
  - make list/tuple with pairs of 'name' & function
  - add it to constructor of GENIE instance and your buttons will appear in interface

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
  - custom buttons with user code for real-time managing train process
