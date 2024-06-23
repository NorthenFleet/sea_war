import threading
import pygame


class RenderThread(threading.Thread):
    def __init__(self, screen, env_config):
        threading.Thread.__init__(self)
        self.screen = screen
        self.env_config = env_config
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.render_frame()

    def render_frame(self):
        self.screen.fill((0, 0, 0))  # Clear screen with black
        # Render entities
        for player in self.env_config['players']:
            for entity in player['entities']:
                self.draw_entity(entity)
        pygame.display.flip()

    def draw_entity(self, entity):
        if entity.entity_type == "flight":
            color = (255, 0, 0)  # Red for flights
        elif entity.entity_type == "ship":
            color = (0, 255, 0)  # Green for ships
        elif entity.entity_type == "submarine":
            color = (0, 0, 255)  # Blue for submarines
        else:
            color = (255, 255, 255)  # White for unknown types

        pygame.draw.circle(self.screen, color,
                           (int(entity.x), int(entity.y)), 5)
