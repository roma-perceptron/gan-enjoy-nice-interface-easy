# coding=utf-8

"""
GAN: Enjoy Nice Interface Easy или просто GENIE это небольшая библиотека предназанченная
для облегчения процесса мониторинга за обучением генеративно-состязательных сетей (GAN).
А в будущем, возможно появятся и управление гаперпараметрами.
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
from traitlets import TraitError

from IPython.display import Javascript
from IPython.display import clear_output
from genie.genie_texts import GENIE_Texts


class GAN_Enjoy_Nice_Interface_Easy():
  # 
  def __init__(self, endpoint, history_fields, generator=None, constant_noise=None, separator=';', preview_size=250, clear_generated=True, image_shape=(64, 64, 3), plt_style='default', lang='rus', silent=False, update_delay=1, preview_every=None, methods=None):
    # class constants
    self.ENDPOINT = endpoint if endpoint.endswith('/') else endpoint + '/'
    self.HISTORY_FILE = self.ENDPOINT + 'history.csv'
    self.GENERATED = self.ENDPOINT + 'generated/'
    self.SEPARATOR = separator
    self.PREVIEW_SIZE = preview_size
    self.FIELDNAMES = history_fields
    self.CLEAR_GENERATED = clear_generated
    self.IMAGE_SHAPE = image_shape
    self.TXT = GENIE_Texts(lang=lang)
    self.GENERATOR = generator
    self.NOISE = constant_noise
    self.METHODS = methods

    # class variables
    self.preview_every = preview_every
    self.initial_update_delay = update_delay
    self.allowed_refresh_graph = True
    self.allowed_refresh_image = True
    self.allowed_last100_steps = False
    self.images_volume = 0
    # self.df = pd.DataFrame(columns=self.FIELDNAMES)
    self.df = pd.DataFrame(columns=self.FIELDNAMES) if self.CLEAR_GENERATED or not os.path.exists(self.HISTORY_FILE) else pd.read_csv(self.HISTORY_FILE, sep=';')
    self.control_command_code = ''
    self.step = -1
    self.steps = 0
    self.last_step_of_previous_train = 0
    self.timepoint = None
    self.steptime = float('+inf')
    self._steptimes = []
    self.training_progress = 0
    self.estimated_time = float('+inf')

    # cheking path for endpoint
    if not os.path.exists(self.ENDPOINT):
      os.mkdir(self.ENDPOINT)

    # initializing
    self._prepare_history_file()
    self._create_generated_path()
    self.images_volume = len(os.listdir(self.GENERATED))
    plt.style.use(plt_style)
    pd.set_option('display.precision', 3)
    
    # show hello
    if not silent:
      print(self.TXT.hello())
    
    # create interface
    self.interface = self.get_interface()
    # with self.output_for_console:
    #   print(self.TXT.console())


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
    js_code = "document.querySelector('.genie--update_buttons').children[0].click();"
    display(Javascript(js_code))


  def show_interface(self, steps):
    """
    Render and display interface for GAN monitoring
    :param steps: int, number of step to current train process
    """
    self.control_command_code = ''
    self.step = -1
    self.steps = steps
    self.timepoint = datetime.datetime.now()
    self.last_step_of_previous_train = self._get_last_step_of_previous_train()
    self.preview_every = self.steps // 100 if not self.preview_every else self.preview_every
    self.preview_every = 1 if self.preview_every == 0 else self.preview_every
    # 
    display(self.interface)
    self.stop_button.layout.display = 'block'
    self._patch_fontawesome()
    self._press_play_button()
    with self.output_for_props:
      clear_output()


  def _prepare_history_file(self):
    """
    Make empty .csv file for history data (empty, but with headers).
    If file exists, it will be overwritten.
    """
    if self.CLEAR_GENERATED:
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
      self._create_zero_step_image()
    elif self.CLEAR_GENERATED:
      shutil.rmtree(self.GENERATED)
      os.mkdir(self.GENERATED)
      self._create_zero_step_image()


  def _create_zero_step_image(self):
    """
    If generated path are empty, create black image as first pre-training
    result of model.
    """
    if not os.listdir(self.GENERATED):
      generated = np.zeros(self.IMAGE_SHAPE)
      pic = Image.fromarray(generated.astype('uint8'), mode='RGB')
      pic.format = 'png'

      pic.save('{}/before_the_times.png'.format(self.ENDPOINT), format='png')


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


  def _get_last_step_of_previous_train(self):
    """
    Return last step number from generated images names
    :return: int, number of last step
    """
    if os.path.exists(self.GENERATED) and len(os.listdir(self.GENERATED)) > 0:
      files = self._get_filelist_sorted(self.GENERATED)
      return int(files[-1].split('.')[0].split('_e')[-1]) + 1
    else:
      return 0


  # read image in bytes
  def _get_genered_image(self, index=-1):
    """
    Get image with given index within generated pics. If images not genered yet,
    return black 'before the times' square.
    :param index: int, idex of necessary image, default -1 (last)
    :return: tuple of image in bytes and str number of step from file name
    """
    if os.path.exists(self.GENERATED) and len(os.listdir(self.GENERATED)) > 0:
      # 
      files = self._get_filelist_sorted(self.GENERATED)
      genenered_image_name = files[index]
      with open(self.GENERATED+genenered_image_name, mode='rb') as f:
        genered_image_bytes = f.read()
      # 
      return genered_image_bytes, genenered_image_name.split('.')[0].split('_e')[-1]
    else:
      with open(self.ENDPOINT + '/before_the_times.png', mode='rb') as f:
        genered_image_bytes = f.read()
      return genered_image_bytes, 'before the times'


  # draw plot for loss and metrics
  def _draw_history(self, history, only='', exclude='', start_x=0):
    """
    Draws double plot for metrics: accuracy-like and loss separately
    :param history: dict, metrics and losses data
    :param only: list of str, which metrics must be draws and ignored all other
    :param exclude: list of str, which metrcis will be ignored
    :param start_x: int, start value of x, it mean start epoch number
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

    fig, (m_plot, l_plot) = plt.subplots(nrows=1, ncols=2, figsize=(35, 7), facecolor='#f5f5f5')
    m_plot.set(title='График метрики (accuracy или иное)', xlabel='Шаг обучения', ylabel='Значение')
    l_plot.set(title='График ошибки (loss)', xlabel='Шаг обучения', ylabel='Значение')
    
    for param in history:
      ax = l_plot if 'loss' in param else m_plot
      if param not in exclude:
        ax.plot(range(start_x, start_x+len(history[param])), history[param], label=labeling.get(param, param))
    
    m_plot.legend()
    l_plot.legend()
    
    plt.show()


  def get_interface(self):
    """
    General function for creating interface by ipywidgets and implement feedback
    loops logic for managing and autoupdating. 
    :return: ipywidgets widget, interface
    """

    # iternal functions and widget-callbacks
    # (use "self_widget" argument instead "self" reserved for class instance) #
    def change_image_preview(self_widget):
      img_bytes, step_num = self._get_genered_image(slider_image.value)
      img_preview.value = img_bytes
      step_label.value = f'from step: {step_num}'
      # 
      if slider_image.value == -1:
        self.allowed_refresh_image = True
      else:
        self.allowed_refresh_image = False
      

    def update_image_preview(self_widget):
      if self.allowed_refresh_image:
        img_bytes, step_num = self._get_genered_image(-1)
        img_preview.value = img_bytes
        step_label.value = f'from step: {step_num}'

    def update_graph(self_widget):
      if type(self_widget.owner) == widgets.widget_bool.ToggleButton:
        self.allowed_refresh_graph = True
      if self_widget.new != (0, 0) and self.allowed_refresh_graph:
        metrics_to_draw = [m.description for m in metrics if m.value]
        indx = steps_range.index
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(self.df[metrics_to_draw][indx[0]: indx[1]].to_dict(orient='list'), start_x=indx[0])


    def last100_steps_range_now(self_widget):
      self.allowed_last100_steps = True
      self.allowed_refresh_graph = True
      start_index = 0 if len(steps_range.options) < 101 else steps_range.options[-101]
      steps_range.index = (start_index, steps_range.options[-1])
        

    def reset_steps_range_now(self_widget):
      if self.df.index.size > 1:
        steps_range.options = range(self.df.shape[0])
        steps_range.index = (0, steps_range.options[-1])
        self.allowed_refresh_graph = True
        self.allowed_last100_steps = False


    def update_all(self): 
      # обновление таблицы
      with output_for_table:
        clear_output(wait=True)
        display(self.df.tail(10))

      # обновление картинки
      self.images_volume = len(os.listdir(self.GENERATED))
      # with output_for_props:
      #   print(self.images_volume, slider_image.min, play_buttons.min)
      if self.images_volume:
        slider_image.min = -self.images_volume
        play_buttons.min = -self.images_volume
        update_image_preview(None)

      # обновление графиков
      if self.step == 1:
        reset_steps_range_now(None)

      if steps_range.index[0] != 0 or steps_range.index[1] != steps_range.options[-1]:
        self.allowed_refresh_graph = False

      if self.allowed_last100_steps:
        self.allowed_refresh_graph = True
        steps_range.options = range(self.df.shape[0])
        start_index = 0 if len(steps_range.options) < 101 else steps_range.options[-101]
        steps_range.index = (start_index, steps_range.options[-1])
      elif self.df.index.size > 1 and self.allowed_refresh_graph:
        try:
          steps_range.options = range(self.df.shape[0])
          steps_range.index = (0, self.df.shape[0]-1)
        except TraitError:
          # print('Ошибка из-за какой-то рассинхронизации, не успевают обновиться значения в виджете. Просто пробую еще разок.')
          steps_range.options = range(self.df.shape[0])
          steps_range.index = (0, self.df.shape[0]-1)

    # global updater by timer
    def update_data(self_widget):
      training_progress_bar.value = self.training_progress
      training_progress_lbl.value = f'Прогресс обучения: {self.training_progress}%, еще {self.estimated_time}'

      with output_for_props:
        clear_output(wait=True)
        if self.step > 0:
          print(f'Шаг #{self.step} завершен.\nВ среднем {self.steptime} сек. на шаг.')
        if self.step == self.steps-1:
          clear_output()
          update_buttons._playing = False
          stop_button.layout.display = 'none'
          print(f'Обучение завершено! {self.steps} шагов обучения!')
          update_all(self)
          return None
        if slider_update.value == slider_update.max:
          # print('Обновляюсь..')
          update_all(self)
          slider_update.value = 1


    def start_updating(self_widget):
      if self_widget.old == False and self_widget.new == True:
        update_all(self)

    
    def update_button_click(self_widget):
      with output_for_props:
        clear_output(wait=True)
        print('Обновляюсь досрочно..')
      update_all(self)
      update_data(self_widget)
      

    def stop_training(self_widget):
      self.control_command_code = 'stop_training'
      stop_button.layout.display = 'none'
      update_buttons._playing = False
      with output_for_props:
        print('Останавливаю обучение..')
        update_all(self)

    def get_animation_now(self_widget):
      if self.images_volume > 0:
        self.get_animation(silent=True)
        msg = f'Анимация готова, в папке {self.ENDPOINT}\nБольше возможностей по созданию анимации при явном использовании genie.get_animation()'
      else:
        msg = 'Примеров работы генератора нет'
      with output_for_props:
        clear_output()
        print(msg)
        time.sleep(5)
        clear_output()


    # body of code for bulding interface
    #
    # create outputs
    output_for_graph = widgets.Output()
    output_for_table = widgets.Output()
    output_for_props = widgets.Output(layout=widgets.Layout(height='75px'))
    output_for_prevs = widgets.Output()
    
    self.output_for_props = output_for_props
    self.output_for_console = widgets.Output(layout=widgets.Layout(margin='3px 0px 0px 0px'))

    # new auto updating mechanism
    update_buttons = widgets.Play(
        value=1,
        min=1,
        max=self.initial_update_delay * 4,
        step=1,
        interval=250,
        description="Press play",
        _repeat = True
    )
    slider_update = widgets.IntSlider(1, min=1, max=update_buttons.max, readout=True)
    update_widget = widgets.link((update_buttons, 'value'), (slider_update, 'value'))
    update_box = widgets.HBox([update_buttons, slider_update], layout=widgets.Layout(width='95%'))
    self.update_box = update_box
    # 
    update_box.add_class('genie--update_box')
    update_buttons.add_class('genie--update_buttons')
    update_buttons.observe(start_updating, names='_playing')
    slider_update.observe(update_data, names='value')

    # groups of widget for preview generated images
    img_preview = widgets.Image(
        value=self._get_genered_image(-1)[0],
        format='png',
        width=self.PREVIEW_SIZE,
        layout=widgets.Layout(border='3px outset #e0e0e0')
    )
    # 
    slider_image = widgets.IntSlider(-1, min=-2, max=-1, continuous_update=False, readout=False)
    slider_image.observe(change_image_preview, names='value')
    # 
    play_buttons = widgets.Play(
        value=-1,
        min=-2,
        max=-1,
        step=1,
        interval=500,
        description="Press play",
        disabled=False
    )
    play_widget = widgets.jslink((play_buttons, 'value'), (slider_image, 'value'))
    make_animation = widgets.Button(description='Сделать анимацию')
    make_animation.on_click(get_animation_now)
    play_box = widgets.VBox([slider_image, widgets.HBox([play_buttons, make_animation])], layout=widgets.Layout(align_items='center'))
    step_label = widgets.Label(value='before training')

    # stop button
    stop_button = widgets.Button(description='завершить обучение', icon='cancel')
    stop_button.layout.width = 'initial'
    stop_button.on_click(stop_training)
    self.stop_button = stop_button

    #progress bars
    training_progress_lbl = widgets.Label('Прогресс обучения: 0%')
    training_progress_bar = widgets.FloatProgress(value=0, style={'bar_color': '#1cd3a2'}, layout=widgets.Layout(margin='-10px 0px 0px 0px')) #1cd3a2 00bbff
    training_progress_box = widgets.VBox([training_progress_lbl, training_progress_bar], layout=widgets.Layout(margin='0px 0px 10px 0px'))
    # 
    progress_bars = widgets.VBox([training_progress_box])

    update_button = widgets.Button(description='обновить сейчас', icon='refresh', button_style='info')
    update_button.layout.width = 'initial'
    update_button.style.button_color = '#1cd3a2'
    update_button.on_click(update_button_click)
    # self.update_button = update_button
    update_button.layout.display = 'none' # отключение отображения кнопки


    # major parts
    table_box = widgets.Box([output_for_table], layout=widgets.Layout(border='2px solid #e0e0e0', width='40%'))
    props_box = widgets.VBox([progress_bars, update_button, output_for_props, stop_button, update_box], layout=widgets.Layout(border='2px solid #e0e0e0', width='24%', align_items='center'))
    graph_box = widgets.Box([output_for_graph], layout=widgets.Layout(border='2px solid #e0e0e0'))
    graph_box.add_class('graph_box')

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
    metrics_box.add_class('metrics_box')

    # complex widget for manage step range for graph
    steps_range = widgets.SelectionRangeSlider(
        options=range(2),
        index=(0, 1),
        description='Диапазон:',
        continuous_update=False,
        layout=widgets.Layout(width='90%'),
        style={'description_width': 'initial'}
    )
    steps_range.observe(update_graph, names='value')
    # 
    last100_steps_range = widgets.Button(description='Последние 100')
    last100_steps_range.on_click(last100_steps_range_now)
    # 
    reset_steps_range = widgets.Button(description='Весь диапазон')
    reset_steps_range.on_click(reset_steps_range_now)
    steps_range_box = widgets.HBox([steps_range, last100_steps_range, reset_steps_range], layout=widgets.Layout(justify_content='space-between', margin='10px 0px 0px 0px'))
    steps_range_box.add_class('steps_range_box')

    # buttons for user-methods
    if self.METHODS:
      buttons = [widgets.Button(description=method[0]) for method in self.METHODS]
      for i, user_button in enumerate(buttons):
        user_button.on_click(self.METHODS[i][1])
    user_buttons = widgets.HBox(buttons) if self.METHODS else widgets.HBox()

    # build all together
    preview_box = widgets.VBox([img_preview, step_label, play_box], layout=widgets.Layout(border='2px solid #e0e0e0', width='35%', align_items='center'))
    upper_box = widgets.HBox([preview_box, table_box, props_box], layout=widgets.Layout(justify_content='space-between'))
    hor_line = widgets.Box()
    lower_box = widgets.VBox([metrics_box, graph_box, steps_range_box, hor_line, user_buttons, self.output_for_console])
    main_box = widgets.VBox([upper_box, lower_box], layout=widgets.Layout(align_content='space-around', border='10px solid transparent', width='100%'))
    # 
    upper_box.add_class('upper_box')
    lower_box.add_class('lower_box')
    hor_line.add_class('horizontal_line')

    # upload stylesheet file
    with open('genie/utils/styles.css', mode='r') as f:
      data_input_style = f.read()

    # apply styles
    interface = widgets.Box([widgets.HTML(data_input_style), main_box])
    interface.add_class('main')
    
    with output_for_graph:
      clear_output(wait=True)
      self._draw_history({f: [None] for f in self.FIELDNAMES})
    # 
    return interface

  
  def change_delay_time(self, value=1):
    value = np.clip(float(value), 1, 60).astype(int)
    self.update_box.children[0].max = 4 * value
    self.update_box.children[1].max = 4 * value
    with self.output_for_console:
      print(f'Время задержки перед обновлением изменено, теперь: {value} сек.')

  # 
  def compose_generated(self, generated, shift=0):
    
    l = len(generated)

    if l == 1:
      rows, cols = 1, 1
    else:
      cols = int(l ** 0.5)
      rows = l // cols

    if generated[0].ndim == 3 and generated[0].shape[-1] == 3:
      mode = 'RGB'
    else:
      mode = None

    width = generated[0].shape[0]
    height = generated[0].shape[1]

    combined_img = Image.new('RGB', (width * rows + shift * (rows-1), height * cols + shift * (cols-1)))
    combined_img.format = 'png'
    index = 0
    for r in range(cols):
      for c in range(rows):
        if mode:
          next_img = Image.fromarray(255 * generated[index], mode=mode)
        else:
          next_img = Image.fromarray(255 * generated[index].reshape(generated[index].shape[:2]), mode=None)
        combined_img.paste(next_img, (width * c + shift * c, height * r + shift * r))
        index += 1

    return combined_img


  # отрисовква генерации
  def make_gen_preview(self, generator, noise, step_number=False, verbose=0, path_for_generated=None, shift=0, resize_to=None):
  # 
    """
    """
    generated = generator.predict(noise)
    pic = self.compose_generated(generated, shift=shift)
    
    if resize_to:
      pic = pic.resize(resize_to)

    if not path_for_generated:
      path_for_generated = self.GENERATED
    next_num = len(os.listdir(path_for_generated))

    if not step_number:
      step_number = self.last_step_of_previous_train + self.step
    pic.save('{}/{}_e{}.png'.format(path_for_generated, next_num, step_number), format='png')

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

  
  def compute_time(self):
    """
    Compute time duration of one step and etc
    """
    steptime = (datetime.datetime.now() - self.timepoint).total_seconds()
    time_samples = 100
    if len(self._steptimes) == time_samples:
      self._steptimes.pop(0)
    self._steptimes.append(steptime)
    self.steptime = round(sum(self._steptimes) / time_samples, 2)
    self.timepoint = datetime.datetime.now()

    self.training_progress = round(100 * (self.step / (self.steps-1)), 2)
    self.estimated_seconds = (self.steps - self.step - 1) * self.steptime
    if self.estimated_seconds > 24 * 60 * 60 - 1:
      self.estimated_time = 'infinity'
    else:
      self.estimated_time = time.strftime("%H:%M:%S", time.gmtime(self.estimated_seconds))


  def update_data(self, params):
    """
    Interface for transfer history params from your training function to GENIE
    :param params: dict, key-value and keys must be same as 'metrics_for_monitoring'
    """
    # update internal dataframe object
    self.df = self.df.append(params, ignore_index=True)

    # preparation data and write to csv
    string_for_append = ';'.join([str(params[param]) for param in self.FIELDNAMES]) + '\n'
    self.write_to_history_file(string_for_append)

    # increment for step counter here
    self.step += 1

    # compute time duration of one step and etc
    self.compute_time()

    # making preview of generator
    if self.GENERATOR and type(self.NOISE) != type(None) and self.step % self.preview_every == 0:
      self.make_gen_preview(self.GENERATOR, self.NOISE)


  def help(self):
    print(self.TXT.help())

  def example(self):
    print(self.TXT.example())

