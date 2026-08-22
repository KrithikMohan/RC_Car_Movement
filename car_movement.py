import os
import threading
import time
import cv2
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
    'drive_mode': 'manual',
    'rccar_available': True,
    'nearest_obs_cm': None,
    'curb_side': None,
    'curb_offset_cm': None,
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


auto_err_time = 0.0


def send_post(endpoint, json_data=None, on_success=None, on_error=None):
    def _post():
        try:
            res = requests.post(f'{BASE_URL}/{endpoint}', json=json_data, timeout=1.0)
            if res.status_code == 200:
                if on_success:
                    on_success(res.json())
            else:
                if on_error:
                    on_error(res)
                print(f'[NET ERROR] {endpoint} returned {res.status_code}: {res.text}')
        except Exception as e:
            if on_error:
                on_error(e)
            print(f'[NET ERROR] {endpoint}: {e}')

    threading.Thread(target=_post, daemon=True).start()


def toggle_auto_mode():
    global telemetry, auto_err_time
    target_mode = 'manual' if telemetry.get('drive_mode') == 'auto' else 'auto'

    def _on_success(data):
        server_mode = data.get('mode', target_mode)
        telemetry['drive_mode'] = server_mode

    def _on_error(err):
        global auto_err_time
        auto_err_time = time.time()
        print(f'[AUTO TOGGLE FAILED] Could not switch to {target_mode}: {err}')

    send_post('mode', {'mode': target_mode}, on_success=_on_success, on_error=_on_error)


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
                data = res.json()
                telemetry.update(data)

                # Auto-stop and trigger download if storage runs low
                if telemetry.get('is_recording') and telemetry.get('free_storage_gb', 10.0) < 1.0:
                    rec_file = telemetry.get('rec_file', '')
                    send_post('recording', {'action': 'stop'})
                    if rec_file and rec_file != last_downloaded_file:
                        last_downloaded_file = rec_file
                        download_video_async(rec_file)
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=poll_telemetry_loop, daemon=True).start()


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
RECT_AUTO = pygame.Rect(380, 20, 110, 40)
RECT_RESET_HOME = pygame.Rect(500, 20, 100, 40)
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
    telemetry_bg = pygame.Surface((360, 135), pygame.SRCALPHA)
    pygame.draw.rect(telemetry_bg, (10, 10, 10, 190), (0, 0, 360, 135), border_radius=8)
    screen.blit(telemetry_bg, (20, 70))

    x_m, y_m = telemetry.get('pos_x', 0.0), telemetry.get('pos_y', 0.0)
    dist = telemetry.get('distance_home_m', 0.0)
    head = telemetry.get('heading_home_deg', 0.0)
    mode = telemetry.get('drive_mode', 'manual').upper()
    curb = telemetry.get('curb_side')
    curb_off = telemetry.get('curb_offset_cm')
    obs_dist = telemetry.get('nearest_obs_cm')

    mode_color = (0, 255, 120) if mode == 'AUTO' else (200, 200, 200)
    t0 = font_sm.render(f'DRIVE MODE: {mode} (Press M to toggle)', True, mode_color)
    t1 = font_sm.render(f'POSITION: X: {x_m:.2f}m  Y: {y_m:.2f}m', True, (240, 240, 240))
    t2 = font_sm.render(f'DIST TO HOME: {dist:.2f} meters', True, (0, 255, 180))
    t3 = font_sm.render(f'RETURN HEADING: {head:.1f}°', True, (255, 215, 0))

    curb_str = f"{curb.upper()} ({curb_off:.0f}cm)" if (curb and curb_off is not None) else ("ACTIVE" if curb else "SEARCHING")
    obs_str = f"{obs_dist:.0f}cm" if obs_dist is not None else "CLEAR"
    t4 = font_sm.render(f'CURB: {curb_str} | OBS: {obs_str}', True, (120, 220, 255))

    screen.blit(t0, (30, 75))
    screen.blit(t1, (30, 95))
    screen.blit(t2, (30, 115))
    screen.blit(t3, (30, 135))
    screen.blit(t4, (30, 155))

    bat_pct = telemetry.get('battery_pct', 100)
    voltage = telemetry.get('voltage', 12.6)
    bat_col = (220, 50, 50, 220) if bat_pct < 20 else ((220, 140, 0, 220) if bat_pct < 50 else (30, 180, 60, 220))

    v_str = f"{voltage:.2f}V" if isinstance(voltage, (int, float)) else f"{voltage}V"
    draw_button(pygame.Rect(SCREEN_W - 330, 20, 180, 40), f"BAT: {bat_pct}% ({v_str})", False, bg_color=bat_col)

    sd_free = telemetry.get('free_storage_gb', 0.0)
    draw_button(pygame.Rect(SCREEN_W - 330, 70, 180, 40), f"SD FREE: {sd_free:.1f}GB" if isinstance(sd_free, (int, float)) else f"SD FREE: {sd_free}GB", False, bg_color=(30, 30, 30, 190))


running = True
clock = pygame.time.Clock()

while running:
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                # Toggle Autonomous Driving Mode
                toggle_auto_mode()
            elif event.key == pygame.K_SPACE:
                # Emergency Stop
                send_post('mode', {'mode': 'manual'})
                telemetry['drive_mode'] = 'manual'
                send_cmd('s')
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if RECT_REC.collidepoint(mouse_pos):
                if telemetry.get('is_recording'):
                    send_post('recording', {'action': 'stop'})
                else:
                    send_post('recording', {'action': 'start'})
            elif RECT_TRANSFER.collidepoint(mouse_pos):
                if telemetry.get('rec_file'):
                    download_video_async(telemetry['rec_file'])
            elif RECT_SPEED.collidepoint(mouse_pos):
                new_mode = 'high' if telemetry.get('speed_mode') == 'low' else 'low'
                send_post('speed', {'mode': new_mode})
            elif RECT_AUTO.collidepoint(mouse_pos):
                toggle_auto_mode()
            elif RECT_RESET_HOME.collidepoint(mouse_pos):
                send_post('reset_origin')
                telemetry['pos_x'] = 0.0
                telemetry['pos_y'] = 0.0
                telemetry['distance_home_m'] = 0.0
                telemetry['heading_home_deg'] = 0.0
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

    mouse_pressed = pygame.mouse.get_pressed()[0]
    keys = pygame.key.get_pressed()

    w_act = keys[pygame.K_w] or (mouse_pressed and RECT_W.collidepoint(mouse_pos))
    a_act = keys[pygame.K_a] or (mouse_pressed and RECT_A.collidepoint(mouse_pos))
    s_act = keys[pygame.K_s] or (mouse_pressed and RECT_S.collidepoint(mouse_pos))
    d_act = keys[pygame.K_d] or (mouse_pressed and RECT_D.collidepoint(mouse_pos))

    # Any manual keypress immediately overrides auto mode
    if (w_act or a_act or s_act or d_act) and telemetry.get('drive_mode') == 'auto':
        telemetry['drive_mode'] = 'manual'
        send_post('mode', {'mode': 'manual'})

    # Smooth combined driving and steering commands
    if w_act and a_act:
        send_cmd('bl')
    elif w_act and d_act:
        send_cmd('br')
    elif s_act and a_act:
        send_cmd('fl')
    elif s_act and d_act:
        send_cmd('fr')
    elif w_act:
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

    rec_text = 'STOP REC' if telemetry.get('is_recording') else 'START REC'
    rec_color = (220, 40, 40, 220) if telemetry.get('is_recording') else (40, 40, 40, 190)
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

    spd_mode = telemetry.get('speed_mode', 'low').upper()
    draw_button(RECT_SPEED, f'SPD: {spd_mode}', False, bg_color=(40, 100, 200, 190))

    # Auto Mode Button
    is_auto = telemetry.get('drive_mode') == 'auto'
    if time.time() - auto_err_time < 2.0:
        auto_label = 'AUTO: ERR'
        auto_bg = (220, 40, 40, 220)
    else:
        auto_label = 'AUTO: ON' if is_auto else 'AUTO: OFF'
        auto_bg = (0, 180, 70, 220) if is_auto else (40, 40, 40, 190)
    draw_button(RECT_AUTO, auto_label, is_auto, bg_color=auto_bg)

    draw_button(RECT_RESET_HOME, 'SET HOME', False, bg_color=(40, 40, 40, 190))
    draw_button(RECT_SHUTDOWN, 'OFF PI', False, bg_color=(180, 30, 30, 220))

    draw_hud_overlay()

    pygame.display.flip()
    clock.tick(30)

send_cmd('s')
cap.release()
pygame.quit()