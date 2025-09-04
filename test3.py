import pygame
import math
import sys
from pygame.locals import *

# test3.py
# Simple first-person pseudo-3D racing game using pygame
# Run: pip install pygame
# Then: python test3.py


# Initialize
pygame.init()
WIDTH, HEIGHT = 900, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 24)

# Game params
HORIZON = HEIGHT // 3
CAMERA_HEIGHT = 100
DRAW_STEP = 2  # pixel step for road scanlines
MAX_SPEED = 400.0
ACC = 600.0
FRIC = 300.0
STEER_SPEED = 3.0  # how fast player moves left/right
PLAYER_WIDTH = 60
PLAYER_HEIGHT = 100

# Track / world state
speed = 0.0     # forward speed (units/s)
dist = 0.0      # distance along track (units)
player_pos = 0.0  # player lateral position (-1..1)
steer_vel = 0.0
lap_time = 0.0
crashed = False

def road_curve_at(z):
    # A procedurally curved track: sum of sines
    return 200 * math.sin(z * 0.002) + 150 * math.sin(z * 0.0007 + 1.5) + 80 * math.sin(z * 0.0017 + 3.0)

def lerp(a, b, t):
    return a + (b - a) * t

def draw_hud():
    txt = FONT.render(f"Speed: {int(speed)}  Pos: {player_pos:.2f}  Dist: {int(dist)}", True, (255,255,255))
    SCREEN.blit(txt, (10, 10))
    if crashed:
        t2 = FONT.render("CRASH! Press R to restart", True, (255, 50, 50))
        SCREEN.blit(t2, (WIDTH//2 - t2.get_width()//2, 50))

def draw_road():
    # We'll draw slices from horizon to bottom using polygons
    screen_cx = WIDTH // 2
    prev_left = prev_right = screen_cx
    prev_y = HORIZON
    # Precompute curve at current forward distance to simulate upcoming road
    for y in range(HORIZON, HEIGHT, DRAW_STEP):
        perspective = (y - HORIZON) / (HEIGHT - HORIZON)  # 0..1
        z = dist + perspective * 2000  # lookahead distance for this slice
        curve = road_curve_at(z)
        # Road width decreases with distance
        road_w = lerp(200, WIDTH * 1.3, 1 - perspective)  # near big, far small
        # Center offset: track curve and player's offset affect center
        # curve shifts camera relative to world; player_pos shifts camera opposite
        center_x = screen_cx + curve * (1 - perspective) + (-player_pos * screen_cx) * (1 - perspective) * 0.6
        left = center_x - road_w / 2
        right = center_x + road_w / 2
        # Sky/ground
        if y == HORIZON:
            prev_left, prev_right = left, right
            prev_y = y
            continue
        # draw grass
        grass_color = (14, 120, 14) if (int(z//100) % 2 == 0) else (10, 95, 10)
        pygame.draw.polygon(SCREEN, grass_color, [(0, prev_y), (WIDTH, prev_y), (WIDTH, y), (0, y)])
        # draw road segment
        road_color = (90, 90, 90)
        pygame.draw.polygon(SCREEN, road_color, [(prev_left, prev_y), (prev_right, prev_y), (right, y), (left, y)])
        # draw road edges
        pygame.draw.line(SCREEN, (255,255,255), (prev_left, prev_y), (left, y), max(1, int(2 * (1 - perspective))))
        pygame.draw.line(SCREEN, (255,255,255), (prev_right, prev_y), (right, y), max(1, int(2 * (1 - perspective))))
        # center stripe (dashed)
        stripe_w = max(2, int(8 * (1 - perspective)))
        pattern = 40
        if (int(z) // pattern) % 2 == 0:
            cx1 = lerp(prev_left, prev_right, 0.5)
            cx2 = lerp(left, right, 0.5)
            pygame.draw.polygon(SCREEN, (255, 240, 60), [(cx1 - stripe_w, prev_y), (cx1 + stripe_w, prev_y), (cx2 + stripe_w, y), (cx2 - stripe_w, y)])
        prev_left, prev_right = left, right
        prev_y = y

    # draw bottom road edges highlight (near)
    pygame.draw.rect(SCREEN, (40,40,40), (0, HEIGHT-80, WIDTH, 80))

def draw_player():
    # compute player's screen x based on bottom road width
    # approximate bottom road center
    bottom_curve = road_curve_at(dist + 1.0 * 2000)
    bottom_center = WIDTH//2 + bottom_curve * 0  # curve near bottom negligible in our scheme
    # map player_pos (-1..1) to screen x inside road edges
    bottom_road_width = lerp(200, WIDTH * 1.3, 0)  # near param = 0 => near width
    car_x = WIDTH//2 + player_pos * (bottom_road_width / 2 - 60)
    car_y = HEIGHT - PLAYER_HEIGHT - 20
    # car rotation based on steer velocity
    angle = -steer_vel * 6
    # draw simple car
    car_rect = pygame.Rect(0,0, PLAYER_WIDTH, PLAYER_HEIGHT)
    car_rect.center = (car_x, car_y)
    car_surf = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), SRCALPHA)
    pygame.draw.polygon(car_surf, (200, 30, 30), [(PLAYER_WIDTH//2, 0), (PLAYER_WIDTH, PLAYER_HEIGHT//2), (PLAYER_WIDTH//4*3, PLAYER_HEIGHT), (PLAYER_WIDTH//4, PLAYER_HEIGHT), (0, PLAYER_HEIGHT//2)])
    # wheels
    pygame.draw.rect(car_surf, (20,20,20), (6, PLAYER_HEIGHT-18, 18, 8))
    pygame.draw.rect(car_surf, (20,20,20), (PLAYER_WIDTH-24, PLAYER_HEIGHT-18, 18, 8))
    rotated = pygame.transform.rotate(car_surf, angle)
    rrect = rotated.get_rect(center=car_rect.center)
    SCREEN.blit(rotated, rrect.topleft)

def reset():
    global speed, dist, player_pos, steer_vel, crashed, lap_time
    speed = 0.0
    dist = 0.0
    player_pos = 0.0
    steer_vel = 0.0
    crashed = False
    lap_time = 0.0

reset()

# Main loop
while True:
    dt = CLOCK.tick(60) / 1000.0
    lap_time += dt
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_r:
                reset()

    keys = pygame.key.get_pressed()
    if not crashed:
        # acceleration/brake
        if keys[K_UP]:
            speed += ACC * dt
        elif keys[K_DOWN]:
            speed -= ACC * dt
        else:
            # natural friction
            if speed > 0:
                speed -= FRIC * dt
            else:
                speed += FRIC * dt
        speed = max(0.0, min(MAX_SPEED, speed))

        # steering
        steer = 0.0
        if keys[K_LEFT]:
            steer = -1.0
        if keys[K_RIGHT]:
            steer = 1.0
        # steer velocity smooth
        steer_vel = lerp(steer_vel, steer * STEER_SPEED, min(1.0, dt * 8))
        player_pos += steer_vel * dt * (0.7 + speed / MAX_SPEED)  # faster => more effect
        player_pos = max(-1.2, min(1.2, player_pos))

        # advance distance
        dist += speed * dt * 10  # scale so curves change at a nice rate

        # simple collision when player outside road bounds at bottom
        # compute bottom road left/right like in draw_road for bottom slice
        z = dist + 1.0 * 2000
        curve = road_curve_at(z)
        perspective = 1.0
        road_w = lerp(200, WIDTH * 1.3, 1 - perspective)
        center_x = WIDTH//2 + curve * (1 - perspective) + (-player_pos * WIDTH//2) * (1 - perspective) * 0.6
        left = center_x - road_w / 2
        right = center_x + road_w / 2
        car_screen_x = WIDTH//2 + player_pos * (road_w / 2 - 60)
        if car_screen_x < left + 10 or car_screen_x > right - 10:
            crashed = True
            speed = 0.0

    # draw
    SCREEN.fill((135, 206, 235))  # sky
    # distant mountains / horizon
    pygame.draw.rect(SCREEN, (120, 120, 120), (0, HORIZON-60, WIDTH, 60))
    draw_road()
    draw_player()
    draw_hud()

    pygame.display.flip()