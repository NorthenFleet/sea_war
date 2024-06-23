import threading
import pygame
from render.render_thread import RenderThread


class RenderManager:
    def __init__(self, env_config):
        self.env_config = env_config
        self.screen = None
        self.clock = None
        self.render_thread = None

    def initialize_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))  # Window size
        pygame.display.set_caption(self.env_config["name"])
        self.clock = pygame.time.Clock()

    def start_render_thread(self):
        self.render_thread = RenderThread(self.screen, self.env_config)
        self.render_thread.start()

    def stop_render_thread(self):
        if self.render_thread:
            self.render_thread.stop()
            self.render_thread.join()

    def run(self):
        self.initialize_pygame()
        self.start_render_thread()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.clock.tick(60)  # FPS
        self.stop_render_thread()
        pygame.quit()
