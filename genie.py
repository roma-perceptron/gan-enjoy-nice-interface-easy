# coding=utf-8

"""
GAN: Enjoy Nice Interface Easy или просто GENIE это небольшая библиотека предназанченная
для облегчения процесса мониторинга за обучением генеративно-состязательных сетей (GAN).
А в будущем, возможно появятся возможно по управлению гаперпараметрами.
"""

# imports
import os
import csv
import time
import shutil
import imageio
import datetime

import numpy as np
import pandas as pd

from PIL import Image

import ipywidgets as widgets
import matplotlib.pyplot as plt

from IPython.display import Javascript
from IPython.display import clear_output
from genie.genie_texts import GENIE_Texts



# self.df = pd.DataFrame({f: sorted([1 if abs(n) >=1 else abs(n) for n in np.random.normal(-0.5, 0.5, size=100)], reverse=True if 'loss' in f else False) for f in history_fields})


class GAN_Enjoy_Nice_Interface_Easy():
  # 
  def __init__(self, endpoint, history_fields, separator=';', preview_size=250, clear_generated=True, image_shape=(64, 64, 3), plt_style='default', lang='rus', silent=False):
    # class constants
    self.ENDPOINT = endpoint if endpoint.endswith('/') else endpoint + '/'
    self.HISTORY_FILE = self.ENDPOINT + 'history.csv'
    self.GENERATED = self.ENDPOINT + 'generated/'
    self._GENERATED = self.ENDPOINT + '_generated/'
    self.SEPARATOR = separator
    self.PREVIEW_SIZE = preview_size
    self.FIELDNAMES = history_fields
    self.CLEAR_GENERATED = clear_generated
    self.IMAGE_SHAPE = image_shape
    self.TXT = GENIE_Texts(lang=lang)

    # class variables
    self.allowed_refresh_graph = True
    self.allowed_refresh_image = True
    self.images_volume = 0
    self.df = pd.DataFrame({f: [None] for f in self.FIELDNAMES})
    self.control_command_code = ''
    self.epoch = 0
    self.epochs = 0
    self.last_epoch_of_previous_train = 0

    # cheking path for endpoint
    if not os.path.exists(self.ENDPOINT):
      os.mkdir(self.ENDPOINT)

    # initializing
    self._prepare_history_file()
    self._create_generated_path()
    self.images_volume = len(os.listdir(self.GENERATED))
    plt.style.use(plt_style)
    
    # show hello
    if not silent:
      print(self.TXT.hello())
    
    # create interface
    self.interface = self.get_interface()


  # eternal function and methods
  def _patch_fontawesome(self):
    """
    Patch ipywidgets bug with FontAwesome.
    It seems that ipywidgets upload remote font which don't exist now.
    So, I use my own version instead. 
    """
    with open('genie/utils/fontawesome.js') as f:
      js_code = f.read()
    display(Javascript(js_code))


  def _press_play_button(self, delay=3):
    """
    Emulate clicking play button to start updating.
    Await 'delay' second to ensure html was rendered.
    :param delay: int, seconds to await before execute
    """
    time.sleep(delay)
    js_code = "document.querySelector('.girl-update_buttons').children[0].click();"
    display(Javascript(js_code))


  def show_interface(self, epochs):
    """
    Render and display interface for GAN monitoring
    :param epochs: int, number of epoch to current train process
    """
    self.control_command_code = ''
    self.epochs = epochs
    self.last_epoch_of_previous_train = self._get_last_epoch_of_previous_train()
    display(self.interface)
    self._patch_fontawesome()
    self._press_play_button()


  def _prepare_history_file(self):
    """
    Make empty .csv file for history data (empty, but with headers).
    If file exists, it will be overwritten.
    """
    if os.path.exists(self.HISTORY_FILE):
      os.remove(self.HISTORY_FILE)

    if not os.path.exists(self.HISTORY_FILE):
      with open(self.HISTORY_FILE, mode='w') as f:
        f.write(';'.join(self.FIELDNAMES)+'\n')


  def _create_generated_path(self):
    """
    Create empty path for store generated images.
    If path exists, it will be overwritten.
    """
    if not os.path.exists(self.GENERATED):
      os.mkdir(self.GENERATED)
      self._create_zero_epoch_image()
    elif self.CLEAR_GENERATED:
      shutil.rmtree(self.GENERATED)
      os.mkdir(self.GENERATED)
      self._create_zero_epoch_image()


  def _create_zero_epoch_image(self):
    """
    If generated path are empty, create total black image as first pre-training
    result of model.
    """
    if not os.listdir(self.GENERATED):
      generated = np.zeros(self.IMAGE_SHAPE)
      pic = Image.fromarray(generated.astype('uint8'), mode='RGB')
      pic.format = 'png'

      next_num = len(os.listdir(self.GENERATED)) + 1
      pic.save('{}/0_e0.png'.format(self.GENERATED), format='png')


  def _get_filelist_sorted(self, pathname):
    """
    Get filenames by given path name and return sorted version.
    Sort by first number in name, chronologically, from first to last.
    :param pathname: str, path to find images to sort
    :return: list, names of image-files
    """
    return sorted(os.listdir(pathname),
                  key=lambda x: int(x.split('.')[0].split('_e')[0])
    )


  def _get_last_epoch_of_previous_train(self):
    """
    Return last epoch number from generated images names
    :return: int, number of last epoch
    """
    files = self._get_filelist_sorted(self.GENERATED)
    return int(files[-1].split('.')[0].split('_e')[-1])

  # read image in bytes
  def _get_genered_image(self, index=-1):
    """
    Get image with given index within generated pics
    :param index: int, idex of necessary image, default -1 (last)
    :return: tuple of image in bytes and str number of epoch from file name
    """
    if os.path.exists(self.GENERATED) and len(os.listdir(self.GENERATED)) > 0:
      path_for_generated = self.GENERATED
    else:
      self._create_pseudo_generated_path()
      path_for_generated = self._GENERATED
    # 
    files = self._get_filelist_sorted(path_for_generated)
    genenered_image_name = files[index]
    with open(path_for_generated+genenered_image_name, mode='rb') as f:
      genered_image_bytes = f.read()
    # 
    return genered_image_bytes, genenered_image_name.split('.')[0].split('_e')[-1]



  # draw plot for loss and metrics
  def _draw_history(self, history, only='', exclude='', start_x=0, final=None, show_trend=False, show_avg=False):
    """
    Draws double plot for metrics: accuracy-like and loss separately
    :param history: dict, metrics and losses data
    :param only: list of str, which metrics must be draws and ignored all other
    :param exclude: list of str, which metrcis will be ignored
    :param start_x: int, start value of x, it mean start epoch number
    :param final: float, if present, draws horisontal line for given value
    :param show_trend: bool, if True, draws straight line from half to end
    :param show_avg: bool, if True, draws mean average !NOT IMPLEMENTED NOW!
    :return: None, graph displayed in current output
    """
    # 
    labeling = {
        'accuracy': 'Доля верных ответов на обучающем наборе',
        'val_accuracy': 'Доля верных ответов на проверочном наборе',
        'loss': 'Ошибка на обучающем наборе',
        'val_loss': 'Ошибка на проверочном наборе',
        'final': 'Финальное значение метрики',
    } 

    fig, (m_plot, l_plot) = plt.subplots(nrows=1, ncols=2, figsize=(35, 6), facecolor='#f5f5f5')
    m_plot.set(title='График метрики (accuracy или иное)', xlabel='Эпоха обучения', ylabel='Значение')
    l_plot.set(title='График ошибки (loss)', xlabel='Эпоха обучения', ylabel='Значение')
    
    # отрисовка всех данных которые есть в history
    for param in history:
      ax = l_plot if 'loss' in param else m_plot
      if param not in exclude:
        # показываю примерный тренд
        if show_trend:
          # xdots, ydots это набор икс-координат, и y-координат по отдельности!
          xdots = len(history[param])//2, len(history[param])-1
          ydots = history[param][len(history[param])//2], history[param][-1]
          ax.plot(xdots, ydots, color='r', linestyle='--', linewidth=3)
          
        # # рисую скользящую среднюю
        # if show_avg:
        #   xLen = len(history[param]) // 10
        #   ma = lambda x: np.asarray(x).mean() # маленькая обертка для расчета среднего и возврата целого числа
        #   xyma = [ma(history[param][i:i+xLen+1]) if i > xLen else ma(history[param][:i+1]) for i in range(len(history[param]))]
        #   ax.plot(xyma, color='y', linewidth=6)
        # print(*history[param].items())
        if 'loss' in param:
          ax.plot(range(start_x, start_x+len(history[param])), history[param], label=labeling.get(param, param))
        else: #? это что? хм
          ax.plot(range(start_x, start_x+len(history[param])), history[param], label=labeling.get(param, param))

      # лениво узнаю сколько эпох было..
      epochs = len(history[param])

    if final:
      m_plot.plot([final for i in range(epochs)], label=labeling.get('final', final))

    m_plot.legend()
    l_plot.legend()

    # 
    plt.show()



  def get_interface(self):
    """
    General function for creating interface by ipywidgets and implement feedback
    loops logic for managing and autoupdating. 
    :return: ipywidgets widget, interface
    """

    # iternal functions and widget-callbacks
    # (with "self_widget" argument instead "self" reserved for class instance) #
    def change_image_preview(self_widget):
      img_bytes, epoch_num = self._get_genered_image(slider_image.value)
      img_preview.value = img_bytes
      epoch_label.value = f'from epoch: {epoch_num}'
      # 
      if slider_image.value == -1:
        self.allowed_refresh_image = True
      else:
        self.allowed_refresh_image = False
      

    def update_image_preview(self_widget):
      if self.allowed_refresh_image:
        img_bytes, epoch_num = self._get_genered_image(-1)
        img_preview.value = img_bytes
        epoch_label.value = f'from epoch: {epoch_num}'
      

    def reset_epochs_range_now(self_widget):
      epochs_range.options = range(self.df.shape[0])
      epochs_range.index = (0, self.df.shape[0]-1)
      self.allowed_refresh_graph = True


    def update_graph(self_widget):
      if self_widget.new != (0, 0):
        metrics_to_draw = [m.description for m in metrics if m.value]
        epochs_range_to_draw = epochs_range.index
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(self.df[metrics_to_draw][epochs_range_to_draw[0]: epochs_range_to_draw[1]].to_dict(orient='list'), start_x=epochs_range_to_draw[0])


    def update_all(self):
      # обновление базы со строками
      if epochs_range.index[0] != 0 or epochs_range.index[1] != self.df.shape[0]-1:
        self.allowed_refresh_graph = False

      if os.path.getsize(self.HISTORY_FILE) > 100:
        print('hey! change df now!', os.path.getsize(self.HISTORY_FILE))
        self.df = pd.read_csv(self.HISTORY_FILE, sep=';')

        # обновление графиков
        if self.allowed_refresh_graph:
          epochs_range.options = range(self.df.shape[0])
          epochs_range.index = (0, self.df.shape[0]-1)
      else:
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(self.df)

      # обновление картинки
      self.images_volume = len(os.listdir(self.GENERATED))
      slider_image.min = -self.images_volume
      update_image_preview(None)
      
      # обновление таблицы
      with output_for_table:
        clear_output(wait=True)
        pd.set_option('display.precision', 3)
        display(self.df.tail(10))


    # global updater by timer
    def update_data(self_widget):
      training_progress = round(100 * (self.epoch / self.epochs), 2)
      training_progress_bar.value = training_progress
      training_progress_lbl.value = f'Прогресс обучения: {training_progress}%'

      updating_progress = round(100 * slider_update.value / slider_update.max, 2)
      updating_progress_bar.value = updating_progress
      updating_progress_lbl.value = f'Обновление: {updating_progress}%'

      with output_for_props:
        clear_output(wait=True)
        if self.epoch > 0:
          print(f'Эпоха #{self.epoch} завершена')
          # print(slider_update.value, '/', slider_update.max)
        if slider_update.value == slider_update.max:
          print('Обновляюсь..')
          update_all(self)
          slider_update.value = 1

    def start_updating(self_widget):
      if self_widget.old == False and self_widget.new == True:
        update_all(self)

    def stop_training(self_widget):
      self.control_command_code = 'stop_training'
      update_widget.unlink()
      with output_for_props:
        print('Останавливаю обучение..')

    def change_delay_time(self_widget):
      slider_update.max = 2 * slider_update_delay.value
      update_buttons.max = 2 * slider_update_delay.value


    # body of code for bulding interface
    #
    # create outputs
    output_for_graph = widgets.Output()
    output_for_table = widgets.Output()
    output_for_props = widgets.Output()
    output_for_prevs = widgets.Output()

    # auto updating mechanism
    update_buttons = widgets.Play(
        value=15,
        min=1,
        max=16,
        step=1,
        interval=500,
        description="Press play",
        _repeat = True
    )
    slider_update = widgets.IntSlider(15, min=1, max=16, readout=True)
    update_widget = widgets.jslink((update_buttons, 'value'), (slider_update, 'value'))
    update_box = widgets.HBox([update_buttons, slider_update], layout=widgets.Layout(width='95%'))
    # 
    update_box.add_class('girl-update_box')
    update_buttons.add_class('girl-update_buttons')
    update_buttons.observe(start_updating, names='_playing')
    slider_update.observe(update_data, names='value')
    # 
    slider_update_delay = widgets.IntSlider(8, min=1, max=60, continuous_update=False, readout=True, layout=widgets.Layout())
    slider_update_delay.observe(change_delay_time)
    
    # groups of widget for preview generated images
    img_preview = widgets.Image(
        value=self._get_genered_image(-1)[0],
        format='png',
        width=self.PREVIEW_SIZE,
        layout=widgets.Layout(border='3px outset #e0e0e0')
    )
    # 
    slider_image = widgets.IntSlider(-1, min=-self.images_volume, max=-1, continuous_update=False, readout=False, layout=widgets.Layout())
    slider_image.observe(change_image_preview, names='value')
    # 
    play_buttons = widgets.Play(
        value=-1,
        min=-self.images_volume,
        max=-1,
        step=1,
        interval=200,
        description="Press play",
        disabled=False
    )
    play_widget = widgets.jslink((play_buttons, 'value'), (slider_image, 'value'))
    play_box = widgets.VBox([slider_image, play_buttons], layout=widgets.Layout(align_items='center'))
    epoch_label = widgets.Label(value='before training')

    # stop button
    stop_button = widgets.Button(description='завершить обучение')
    stop_button.on_click(stop_training)

    #progress bars
    training_progress_lbl = widgets.Label('Прогресс обучения: 0%')
    training_progress_bar = widgets.FloatProgress(value=0, style={'bar_color': '#00bbff'}, layout=widgets.Layout(margin='-10px 0px 0px 0px'))
    training_progress_box = widgets.VBox([training_progress_lbl, training_progress_bar], layout=widgets.Layout(margin='0px 0px 10px 0px'))
    # 
    updating_progress_lbl = widgets.Label('Обновление: 0%')
    updating_progress_bar = widgets.FloatProgress(value=0, style={'bar_color': '#1cd3a2'}, layout=widgets.Layout(margin='-10px 0px 0px 0px')) #1cd3a2
    updating_progress_box = widgets.VBox([updating_progress_lbl, updating_progress_bar])
    # 
    progress_bars = widgets.VBox([training_progress_box, updating_progress_box])
    
    # major parts
    table_box = widgets.Box([output_for_table], layout=widgets.Layout(border='2px solid #e0e0e0', width='40%'))
    props_box = widgets.VBox([progress_bars, slider_update_delay, output_for_props, stop_button, update_box], layout=widgets.Layout(border='2px solid #e0e0e0', width='24%', align_items='center'))
    graph_box = widgets.Box([output_for_graph], layout=widgets.Layout(border='2px solid #e0e0e0'))

    # buttons for toggle metrics of graph
    metrics = [
      widgets.ToggleButton(
          value=True,
          description=metric,
          button_style='',
          tooltip=f'Включить/исключить метрику {metric}',
      )
      for metric in sorted(self.FIELDNAMES, key=lambda x: 'loss' in x)
    ]
    for m in metrics:
      m.observe(update_graph, names='value')
    metrics_box = widgets.HBox(metrics, layout=widgets.Layout(justify_content='space-around', margin='25px 0px 10px 0px'))

    # complex widget for manage epoch range for graph
    epochs_range = widgets.SelectionRangeSlider(
        options=range(self.df.shape[0]),
        index=(0, self.df.shape[0]-1),
        description='Диапазон эпох:',
        continuous_update=False,
        layout=widgets.Layout(width='90%'),
        style={'description_width': 'initial'}
    )
    epochs_range.observe(update_graph, names='value')
    # 
    reset_epochs_range = widgets.Button(description='Весь диапазон')
    reset_epochs_range.on_click(reset_epochs_range_now)
    epochs_range_box = widgets.HBox([epochs_range, reset_epochs_range], layout=widgets.Layout(justify_content='space-between', margin='10px 0px 0px 0px'))
    
    # build all together
    preview_box = widgets.VBox([img_preview, epoch_label, play_box], layout=widgets.Layout(border='2px solid #e0e0e0', width='35%', align_items='center'))
    upper_box = widgets.HBox([preview_box, table_box, props_box], layout=widgets.Layout(justify_content='space-between'))
    lower_box = widgets.VBox([metrics_box, graph_box, epochs_range_box])
    main_box = widgets.VBox([upper_box, lower_box, output_for_props], layout=widgets.Layout(align_content='space-around', border='10px solid transparent', width='100%'))
    # 
    upper_box.add_class('upper_box')
    lower_box.add_class('lower_box')

    # upload stylesheet file
    with open('genie/utils/styles.css', mode='r') as f:
      data_input_style = f.read()

    # apply styles
    interface = widgets.Box([widgets.HTML(data_input_style), main_box])
    interface.add_class('main')
    
    # 
    return interface


  # отрисовква генерации
  def show_gen_cat(self, generator, noise, epoch_number=0, verbose=0, path_for_generated=''):
  # 
    """
    Надо добавить self.last_epoch_number и к нему добавлять текущий номер эпохи.
    Определять этот last по номеру эпохи из последнего файла в generated!
    """
    # self.last_epoch_of_previous_train
    mode = None
    generated = generator.predict(np.asarray([noise]))[0]
    if generated.ndim == 3 and generated.shape[-1] == 3:
      mode = 'RGB'
    else:
      generated = generated.reshape(generated.shape[:2])
    # 
    pic = Image.fromarray((255 * generated).astype('uint8'), mode=mode)
    pic.format = 'png'

    if not path_for_generated:
      path_for_generated = self.GENERATED
    next_num = len(os.listdir(path_for_generated)) + 1
    epoch_number = self.last_epoch_of_previous_train + epoch_number
    pic.save('{}/{}_e{}.png'.format(path_for_generated, next_num, epoch_number), format='png')

    if verbose:
      display(pic)


  def get_animation(self, first=0, last=-1, each=1, easy_count=100, easy_in=True, easy_out=True, silent=False, path_for_generated=''):
    if not path_for_generated:
      path_for_generated = self.GENERATED

    files = self._get_filelist_sorted(path_for_generated)
    # 
    easy_in_count = easy_count if easy_in else 0
    easy_out_count = -easy_count if easy_out else 0
    # 
    first_part = files[first:first+easy_in_count]
    mid_part = files[first+easy_in_count:easy_out_count+last:each]
    last_part = files[easy_out_count+last:last]
    selected_files = first_part + mid_part + last_part
    # 
    generated_images = [imageio.imread('{}/{}'.format(path_for_generated, file)) for file in selected_files]
    next_num = len([f for f in os.listdir(self.ENDPOINT) if 'GAN' in f and f.endswith('.gif')]) + 1
    imageio.mimsave(self.ENDPOINT + f'GAN_{next_num}.gif', generated_images)
    # 
    if not silent:
      print(f'Анимация GAN_{next_num}.gif из {len(generated_images)}-ти кадров создана!')


  def write_to_history_file(self, lines):
    """
    Method for appending text to .csv data file
    :param lines: str, text for adding, line breaks must be prepared!
    """
    with open(self.HISTORY_FILE, mode='a') as f:
      f.writelines(lines)


  def help(self):
    print(self.TXT.help())

  def example():
    print(self.TXT.example())

