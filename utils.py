import math

ma = 340

DD = 2


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


def find_weapon_type(weapon_name, weapon_data):
    # Helper function to determine the type of weapon (missiles, guns, etc.)
    for weapon_type, weapons in weapon_data["weapons"].items():
        if weapon_name in weapons:
            return weapon_type
    return None


def calculate_distance(unit1_id, unit2_id, game_data):
    # 获取 unit1 和 unit2 的坐标
    unit1_pos = game_data.units[unit1_id].position
    unit2_pos = game_data.units[unit2_id].position

    # 计算距离
    distance = math.sqrt((unit1_pos['x'] - unit2_pos['x']) ** 2 +
                         (unit1_pos['y'] - unit2_pos['y']) ** 2 +
                         (unit1_pos['z'] - unit2_pos['z']) ** 2)

    return distance
