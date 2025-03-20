import pygame
import math
import random




# Initialize Pygame
pygame.init()




# Screen dimensions
w = 1200
h = 700
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Bouncing Balls")




# Colors
bg_color = (255, 255, 255)
ball_color = (0, 255, 255)




# # Number of balls
num_ball = int(input("Enter the desired number of balls): "))




# Ball properties




x = [random.randint(0, w) for _ in range(num_ball)]            
y = [random.randint(0, h) for _ in range(num_ball)]
speed_x = [random.uniform(1, 5) for _ in range(num_ball)]
speed_y = [random.uniform(1, 5) for _ in range(num_ball)]
r = [random.randint(10, 30) for _ in range(num_ball)]




# Ensure no initial overlap
for i in range(num_ball):
    for j in range(i + 1, num_ball):
        while math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) < r[i] + r[j]:
            x[j] = random.randint(50, w-50)
            y[j] = random.randint(50, h-50)




# Main loop
run = True




while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False




    screen.fill(bg_color)




    for i in range(num_ball):
        x[i] += speed_x[i]
        y[i] += speed_y[i]




        # Bounce off the walls
        if x[i] - r[i] < 0 or x[i] + r[i] > w:
            speed_x[i] *= -1
        if y[i] - r[i] < 0 or y[i] + r[i] > h:
            speed_y[i] *= -1




        # Collision between balls
        for j in range(i + 1, num_ball):
            distance = math.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            if distance < r[i] + r[j]:
                speed_x[i], speed_x[j] = speed_x[j], speed_x[i]
                speed_y[i], speed_y[j] = speed_y[j], speed_y[i]




        pygame.draw.circle(screen, ball_color, (int(x[i]), int(y[i])), r[i], 5)




    pygame.display.update()
    pygame.time.Clock().tick(60)




pygame.quit()
