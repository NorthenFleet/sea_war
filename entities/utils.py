
import math

ma = 340


def global_move(entity, target_x, target_y, steps):
    path = []
    current_x, current_y = entity.x, entity.y
    step_x = (target_x - current_x) / steps
    step_y = (target_y - current_y) / steps

    for _ in range(steps):
        current_x += step_x
        current_y += step_y
        path.append((current_x, current_y))

    return path


def local_move(entity, course, speed):
    current_x, current_y = entity.x, entity.y
    step_x = speed * math.sin(course)
    step_y = speed * math.cos(course)

    return step_x, step_y


def detect_targets(sensor, targets):
    detected_targets = []
    sensor_x, sensor_y = sensor.x, sensor.y
    for target in targets:
        target_x, target_y = target.x, target.y
        distance = math.sqrt((sensor_x - target_x)**2 +
                             (sensor_y - target_y)**2)
        if distance <= sensor.range:
            detected_targets.append(target)
    return detected_targets
