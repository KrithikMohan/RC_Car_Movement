import math
import os
import random
import threading
import time
import cv2
import numpy as np
import pygame
import requests

PI_IP = '10.42.0.1'  # Update to active Raspberry Pi IP
BASE_URL = f'http://{PI_IP}:5000'
DOWNLOAD_DIR = os.path.expanduser('~/Downloads')

telemetry = {
    'pos_x': 0.0,
    'pos_y': 0.0,
    'distance_home_m': 0.0,
    'heading_home_deg': 0.0,
    'speed_mode': 'low',
    'battery_pct': 100,
    'voltage': 12.6,
    'is_recording': False,
    'rec_file': '',
    'free_storage_gb': 0.0,
}

current_cmd = 's'
last_send_time = 0
last_downloaded_file = ''
transfer_status = 'IDLE'  # 'IDLE', 'DOWNLOADING', 'SUCCESS', 'FAILED'
transfer_progress_mb = 0.0

drive_mode = 'manual'
auto_state = 'FORWARD'
recovery_timer = 0
turn_dir = 'l'

last_stuck_check_time = time.time()
last_stuck_x = 0.0
last_stuck_y = 0.0


def send_post(endpoint, json_data=None):
  def _post():
    try:
      requests.post(f'{BASE_URL}/{endpoint}', json=json_data, timeout=0.3)
    except Exception as e:
      print(f'[NET ERROR] {endpoint}: {e}')

  threading.Thread(target=_post, daemon=True).start()


def send_cmd(cmd):
  global current_cmd, last_send_time
  now = time.time()
  if cmd != current_cmd or (cmd != 's' and (now - last_send_time > 0.15)):
    current_cmd = cmd
    last_send_time = now
    send_post('control', {'command': cmd})


# NON-BLOCKING STREAMING VIDEO DOWNLOAD
def download_video_async(filename):
  global transfer_status, transfer_progress_mb

  if transfer_status == 'DOWNLOADING':
    print('[TRANSFER] Download already in progress.')
    return

  def _worker():
    global transfer_status, transfer_progress_mb
    transfer_status = 'DOWNLOADING'
    transfer_progress_mb = 0.0
    print(f'[TRANSFER] Starting stream download: {filename}...')

    try:
      url = f'{BASE_URL}/download/{filename}'
      save_path = os.path.join(DOWNLOAD_DIR, filename)

      with requests.get(url, stream=True, timeout=120) as res:
        res.raise_for_status()
        with open(save_path, 'wb') as f:
          for chunk in res.iter_content(chunk_size=16384):
            if chunk:
              f.write(chunk)
              transfer_progress_mb += len(chunk) / (1024 * 1024)

      transfer_status = 'SUCCESS'
      print(f'[TRANSFER SUCCESS] {filename} ({transfer_progress_mb:.1f} MB) -> {save_path}')
    except Exception as e:
      transfer_status = 'FAILED'
      print(f'[TRANSFER FAILED] {e}')

  threading.Thread(target=_worker, daemon=True).start()


def poll_telemetry_loop():
  global telemetry, last_downloaded_file
  while True:
    try:
      res = requests.get(f'{BASE_URL}/telemetry', timeout=0.8)
      if res.status_code == 200:
        telemetry = res.json()

        # Auto-stop and trigger download if storage runs low
        if telemetry['is_recording'] and telemetry['free_storage_gb'] < 1.0:
          rec_file = telemetry['rec_file']
          send_post('recording', {'action': 'stop'})
          if rec_file and rec_file != last_downloaded_file:
            last_downloaded_file = rec_file
            download_video_async(rec_file)
    except Exception:
      pass
    time.sleep(0.5)


threading.Thread(target=poll_telemetry_loop, daemon=True).start()


def process_indoor_auto(frame, telemetry_data):
  global auto_state, recovery_timer, turn_dir
  global last_stuck_check_time, last_stuck_x, last_stuck_y

  now = time.time()

  if auto_state == 'RECOVER_BACK':
    if now - recovery_timer < 1.0:
      return False, False, True, False
    else:
      auto_state = 'RECOVER_TURN'
      recovery_timer = now

  if auto_state == 'RECOVER_TURN':
    if now - recovery_timer < 0.7:
      return (False, True, False, False) if turn_dir == 'l' else (False, False, False, True)
    else:
      auto_state = 'FORWARD_CLEAR'
      recovery_timer = now

  if auto_state == 'FORWARD_CLEAR':
    if now - recovery_timer < 0.6:
      return True, False, False, False
    else:
      auto_state = 'FORWARD'

  if auto_state == 'FORWARD':
    if now - last_stuck_check_time > 1.5:
      dx = telemetry_data['pos_x'] - last_stuck_x
      dy = telemetry_data['pos_y'] - last_stuck_y
      dist_moved = math.sqrt(dx**2 + dy**2)

      last_stuck_check_time = now
      last_stuck_x = telemetry_data['pos_x']
      last_stuck_y = telemetry_data['pos_y']

      if dist_moved < 0.05 and current_cmd == 'b':
        auto_state = 'RECOVER_BACK'
        recovery_timer = now
        turn_dir = 'l' if random.random() > 0.5 else 'r'
        return False, False, True, False

  small = cv2.resize(frame, (320, 240))
  gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  thresh = cv2.adaptiveThreshold(
      blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
  )

  kernel = np.ones((5, 5), np.uint8)
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  roi = closed[130:235, :]

  left_roi = roi[:, 0:106]
  center_roi = roi[:, 106:213]
  right_roi = roi[:, 213:320]

  def get_max_contour_y(roi_block):
    contours, _ = cv2.findContours(
        roi_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    max_y = 0
    for cnt in contours:
      if cv2.contourArea(cnt) > 120:
        for pt in cnt:
          y_val = pt[0][1]
          if y_val > max_y:
            max_y = y_val
    return max_y + 130 if max_y > 0 else 0

  c_y = get_max_contour_y(center_roi)
  l_y = get_max_contour_y(left_roi)
  r_y = get_max_contour_y(right_roi)

  DANGER_Y_THRESHOLD = 205

  if c_y > DANGER_Y_THRESHOLD:
    if l_y < r_y and l_y < DANGER_Y_THRESHOLD:
      return False, True, False, False
    elif r_y < DANGER_Y_THRESHOLD:
      return False, False, False, True
    else:
      auto_state = 'RECOVER_BACK'
      recovery_timer = now
      turn_dir = 'l' if l_y < r_y else 'r'
      return False, False, True, False

  return True, False, False, False


pygame.init()
info = pygame.display.Info()
SCREEN_W, SCREEN_H = info.current_w, info.current_h
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN)
pygame.display.set_caption('UGV02 Control HUD')
font_sm = pygame.font.SysFont('Arial', 18, bold=True)

cap = cv2.VideoCapture(f'{BASE_URL}/video_feed')

BTN_SIZE = 55
SPACING = 62
CENTER_X = SCREEN_W // 2
BOTTOM_Y = SCREEN_H - 80

RECT_W = pygame.Rect(CENTER_X - (BTN_SIZE // 2), BOTTOM_Y - SPACING, BTN_SIZE, BTN_SIZE)
RECT_A = pygame.Rect(CENTER_X - (BTN_SIZE // 2) - SPACING, BOTTOM_Y, BTN_SIZE, BTN_SIZE)
RECT_S = pygame.Rect(CENTER_X - (BTN_SIZE // 2), BOTTOM_Y, BTN_SIZE, BTN_SIZE)
RECT_D = pygame.Rect(CENTER_X - (BTN_SIZE // 2) + SPACING, BOTTOM_Y, BTN_SIZE, BTN_SIZE)

RECT_REC = pygame.Rect(20, 20, 100, 40)
RECT_TRANSFER = pygame.Rect(130, 20, 140, 40)
RECT_SPEED = pygame.Rect(280, 20, 90, 40)
RECT_RESET_HOME = pygame.Rect(380, 20, 100, 40)
RECT_MODE = pygame.Rect(490, 20, 150, 40)
RECT_SHUTDOWN = pygame.Rect(SCREEN_W - 140, 20, 120, 40)


def draw_button(rect, text, is_active, bg_color=None, text_color=(255, 255, 255)):
  if bg_color is None:
    bg_color = (0, 180, 70, 220) if is_active else (20, 20, 20, 180)
  border_color = (255, 255, 255) if is_active else (120, 120, 120)

  surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
  pygame.draw.rect(surf, bg_color, (0, 0, rect.width, rect.height), border_radius=8)
  pygame.draw.rect(surf, border_color, (0, 0, rect.width, rect.height), width=2, border_radius=8)

  txt_surf = font_sm.render(text, True, text_color)
  surf.blit(txt_surf, txt_surf.get_rect(center=(rect.width // 2, rect.height // 2)))
  screen.blit(surf, rect.topleft)


def draw_hud_overlay():
  telemetry_bg = pygame.Surface((340, 130), pygame.SRCALPHA)
  pygame.draw.rect(telemetry_bg, (10, 10, 10, 190), (0, 0, 340, 130), border_radius=8)
  screen.blit(telemetry_bg, (20, 70))

  x_m, y_m = telemetry['pos_x'], telemetry['pos_y']
  dist = telemetry['distance_home_m']
  head = telemetry['heading_home_deg']

  display_mode = drive_mode.upper().replace('_', ' ')
  if drive_mode == 'auto_indoor':
    display_mode = f'AUTO ({auto_state})'

  t0 = font_sm.render(f'DRIVE MODE: {display_mode}', True, (255, 255, 0))
  t1 = font_sm.render(f'POSITION: X: {x_m:.2f}m  Y: {y_m:.2f}m', True, (240, 240, 240))
  t2 = font_sm.render(f'DIST TO HOME: {dist:.2f} meters', True, (0, 255, 180))
  t3 = font_sm.render(f'RETURN HEADING: {head:.1f}°', True, (255, 215, 0))
  rec_display = telemetry['rec_file'] if telemetry['is_recording'] else 'OFF'
  t4 = font_sm.render(f'REC FILE: {rec_display}', True, (200, 200, 200))

  screen.blit(t0, (30, 75))
  screen.blit(t1, (30, 95))
  screen.blit(t2, (30, 115))
  screen.blit(t3, (30, 135))
  screen.blit(t4, (30, 155))

  bat_pct = telemetry['battery_pct']
  bat_col = (220, 50, 50, 220) if bat_pct < 20 else ((220, 140, 0, 220) if bat_pct < 50 else (30, 180, 60, 220))

  draw_button(pygame.Rect(SCREEN_W - 320, 20, 170, 40), f"BAT: {bat_pct}% ({telemetry['voltage']}V)", False, bg_color=bat_col)
  draw_button(pygame.Rect(SCREEN_W - 320, 70, 170, 40), f"SD FREE: {telemetry['free_storage_gb']}GB", False, bg_color=(30, 30, 30, 190))


running = True
clock = pygame.time.Clock()

while running:
  mouse_pos = pygame.mouse.get_pos()

  for event in pygame.event.get():
    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
      running = False
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
      if RECT_REC.collidepoint(mouse_pos):
        if telemetry['is_recording']:
          send_post('recording', {'action': 'stop'})
        else:
          send_post('recording', {'action': 'start'})
      elif RECT_TRANSFER.collidepoint(mouse_pos):
        if telemetry['rec_file']:
          download_video_async(telemetry['rec_file'])
      elif RECT_SPEED.collidepoint(mouse_pos):
        new_mode = 'high' if telemetry['speed_mode'] == 'low' else 'low'
        send_post('speed', {'mode': new_mode})
      elif RECT_RESET_HOME.collidepoint(mouse_pos):
        send_post('reset_origin')
      elif RECT_MODE.collidepoint(mouse_pos):
        drive_mode = 'auto_indoor' if drive_mode == 'manual' else 'manual'
        auto_state = 'FORWARD'
      elif RECT_SHUTDOWN.collidepoint(mouse_pos):
        send_post('shutdown')
        running = False

  ret, raw_frame = cap.read()
  if ret:
    frame_resized = cv2.resize(raw_frame, (SCREEN_W, SCREEN_H))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(frame_surf, (0, 0))
  else:
    screen.fill((15, 15, 15))
    raw_frame = None

  mouse_pressed = pygame.mouse.get_pressed()[0]
  keys = pygame.key.get_pressed()

  manual_w = keys[pygame.K_w] or (mouse_pressed and RECT_W.collidepoint(mouse_pos))
  manual_a = keys[pygame.K_a] or (mouse_pressed and RECT_A.collidepoint(mouse_pos))
  manual_s = keys[pygame.K_s] or (mouse_pressed and RECT_S.collidepoint(mouse_pos))
  manual_d = keys[pygame.K_d] or (mouse_pressed and RECT_D.collidepoint(mouse_pos))

  if manual_w or manual_a or manual_s or manual_d:
    drive_mode = 'manual'
    auto_state = 'FORWARD'

  if drive_mode == 'auto_indoor' and raw_frame is not None:
    auto_w, auto_a, auto_s, auto_d = process_indoor_auto(raw_frame, telemetry)
  else:
    auto_w, auto_a, auto_s, auto_d = False, False, False, False

  w_act = manual_w or auto_w
  a_act = manual_a or auto_a
  s_act = manual_s or auto_s
  d_act = manual_d or auto_d

  if w_act:
    send_cmd('b')
  elif s_act:
    send_cmd('f')
  elif a_act:
    send_cmd('l')
  elif d_act:
    send_cmd('r')
  else:
    send_cmd('s')

  draw_button(RECT_W, 'W', w_act)
  draw_button(RECT_A, 'A', a_act)
  draw_button(RECT_S, 'S', s_act)
  draw_button(RECT_D, 'D', d_act)

  rec_text = 'STOP REC' if telemetry['is_recording'] else 'START REC'
  rec_color = (220, 40, 40, 220) if telemetry['is_recording'] else (40, 40, 40, 190)
  draw_button(RECT_REC, rec_text, False, bg_color=rec_color)

  # Transfer button status feedback
  if transfer_status == 'DOWNLOADING':
    transfer_label = f'{transfer_progress_mb:.1f}MB...'
    transfer_bg = (200, 140, 0, 220)
  elif transfer_status == 'SUCCESS':
    transfer_label = 'DONE!'
    transfer_bg = (30, 180, 60, 220)
  elif transfer_status == 'FAILED':
    transfer_label = 'FAILED'
    transfer_bg = (220, 40, 40, 220)
  else:
    transfer_label = 'TRANSFER'
    transfer_bg = (40, 40, 40, 190)

  draw_button(RECT_TRANSFER, transfer_label, False, bg_color=transfer_bg)

  spd_mode = telemetry['speed_mode'].upper()
  draw_button(RECT_SPEED, f'SPD: {spd_mode}', False, bg_color=(40, 100, 200, 190))
  draw_button(RECT_RESET_HOME, 'SET HOME', False, bg_color=(40, 40, 40, 190))

  mode_button_text = 'AUTO INDOOR' if drive_mode == 'manual' else 'SET MANUAL'
  mode_bg_color = (160, 40, 200, 220) if drive_mode == 'auto_indoor' else (40, 40, 40, 190)
  draw_button(RECT_MODE, mode_button_text, False, bg_color=mode_bg_color)

  draw_button(RECT_SHUTDOWN, 'OFF PI', False, bg_color=(180, 30, 30, 220))

  draw_hud_overlay()

  pygame.display.flip()
  clock.tick(30)

send_cmd('s')
cap.release()
pygame.quit()