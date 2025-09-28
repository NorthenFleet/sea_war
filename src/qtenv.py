import sys
import math
import numpy as np
import gym
from gym import spaces
from PyQt5 import QtWidgets, QtCore, QtGui


class MissileSimEnv(gym.Env):
    """
    导弹-舰船拦截仿真环境，手动键盘控制导弹航向。
    State: [mx, my, mvx, mvy, tx, ty, tvx, tvy]
    """

    def __init__(self):
        super().__init__()
        # 时间步长 & 最大步数
        self.dt = 0.05
        self.max_steps = 10000
        # 导弹飞行速度（m/s）及速度范围
        self.V_m = 300.0
        self.V_m_min = -60000  # 最小速度150
        self.V_m_max = 60000.0  # 最大速度600
        self.V_m_step = 30.0  # 每次加减速的步长

        # 键盘手动控制航向变化速率 (rad/s)
        self.manual_heading_rate = 0.0
        
        # 爆炸效果相关参数
        self.explosion_active = False
        self.explosion_time = 0
        self.explosion_duration = 2.0  # 爆炸持续2秒
        self.explosion_radius = 5.0    # 初始爆炸半径
        self.explosion_max_radius = 50.0  # 最大爆炸半径
        
        # 导弹搜索扇区参数
        self.search_angle = math.pi/3  # 搜索扇区角度（60度）
        self.search_range = 5000.0     # 搜索范围（米）

        # 动作和状态空间（占位，但主要使用手动控制）
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([10.0], dtype=np.float32),
            shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.reset()

    def init_simulation(self):
        # 导弹初始位置与目标设置
        self.m_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        # 生成目标位置（限制在屏幕可见范围内）
        # 考虑到缩放因子和偏移量，计算合适的距离范围
        max_visible_distance = 12000.0  # 最大可见距离
        min_visible_distance = 3000.0   # 最小可见距离（避免太近）
        
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(min_visible_distance, max_visible_distance)
        self.target_pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ], dtype=np.float32)
        
        # 目标速度（可以设置为缓慢移动）
        speed = np.random.uniform(5.0, 15.0)  # 5-15节
        move_angle = np.random.uniform(0, 2 * np.pi)
        self.target_vel = np.array([
            speed * np.cos(move_angle),
            speed * np.sin(move_angle)
        ], dtype=np.float32)
        
        # 目标舰艇属性
        self.target_length = 150.0  # 舰艇长度(m)
        self.target_width = 30.0    # 舰艇宽度(m)
        self.target_heading = move_angle  # 舰艇朝向与移动方向一致
        
        # 重置爆炸状态
        self.explosion_active = False
        self.explosion_time = 0

        # 初始航向指向目标
        los = self.target_pos - self.m_pos
        self.heading = math.atan2(los[1], los[0])

        self.steps = 0
        self.hit = False

    def reset(self):
        self.init_simulation()
        return self._get_state()

    def _get_state(self):
        # 状态向量: 导弹位置、速度，目标位置、速度
        m_vel = self.V_m * \
            np.array([math.cos(self.heading), math.sin(self.heading)])
        state = np.concatenate(
            [self.m_pos, m_vel, self.target_pos, self.target_vel])
        return state.astype(np.float32)

    def step(self, action):
        # 环境的 step 依然可用手动更新，但不在 GUI 模式下使用
        N = float(action[0])
        self._update_missile(N)
        self.target_pos += self.target_vel * self.dt

        dist = np.linalg.norm(self.target_pos - self.m_pos)
        reward = -0.01 * dist
        done = dist < 10.0 or self.steps >= self.max_steps
        if dist < 10.0:
            reward += 100.0
        self.steps += 1

        return self._get_state(), reward, done, {}

    def _update_missile_manual(self):
        # 只根据手动输入调整航向并更新位置
        if not self.explosion_active:
            self.heading += self.manual_heading_rate * self.dt
            self.m_pos += self.V_m * \
                np.array([math.cos(self.heading), math.sin(self.heading)]) * self.dt
            
            # 检测碰撞
            self._check_collision()
        else:
            # 更新爆炸效果
            self.explosion_time += self.dt
            if self.explosion_time >= self.explosion_duration:
                self.reset()  # 爆炸结束后重置游戏

    def _check_collision(self):
        # 简化的碰撞检测：检查导弹是否在舰艇的矩形范围内
        # 计算导弹到目标的距离
        dist = np.linalg.norm(self.target_pos - self.m_pos)
        
        # 更精确的碰撞检测可以考虑舰艇的朝向和形状
        # 这里使用简化版：如果导弹在舰艇长宽范围内，则视为命中
        if dist < max(self.target_length, self.target_width) / 2:
            self.hit = True
            self.explosion_active = True
            self.explosion_time = 0
            
    def render(self, mode='human'):
        # GUI 渲染在 SimulationWindow 中完成
        pass


class SimulationWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.env = MissileSimEnv()
        self.env.reset()
        self.init_ui()

        # 定时器驱动仿真
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(int(self.env.dt * 1000))
        
        # 爆炸效果颜色
        self.explosion_colors = [
            QtGui.QColor(255, 200, 0),   # 黄色
            QtGui.QColor(255, 100, 0),   # 橙色
            QtGui.QColor(255, 0, 0),     # 红色
        ]

    def init_ui(self):
        self.setWindowTitle("导弹制导仿真 - 手动控制")
        self.resize(2400, 1800)
        self.show()

    def on_timer(self):
        # 更新导弹位置
        self.env._update_missile_manual()
        # 目标保持静止或可添加移动逻辑
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # 启用抗锯齿
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 背景
        painter.fillRect(self.rect(), QtGui.QColor(10, 10, 80))

        # 坐标映射参数
        scale = 0.06
        x_offset = self.width() / 4
        y_offset = self.height() / 2

        # 绘制导弹搜索扇区（半透明）
        if not self.env.explosion_active:
            mx, my = self.env.m_pos
            hx = float(mx * scale + x_offset)
            hy = float(y_offset - my * scale)
            hd = -self.env.heading
            search_radius = float(self.env.search_range * scale)
            
            # 创建扇形路径
            search_path = QtGui.QPainterPath()
            search_path.moveTo(hx, hy)
            search_path.arcTo(
                hx - search_radius, 
                hy - search_radius,
                search_radius * 2,
                search_radius * 2,
                math.degrees(-hd - self.env.search_angle/2),
                math.degrees(self.env.search_angle)
            )
            search_path.closeSubpath()
            
            # 设置半透明填充
            painter.setBrush(QtGui.QColor(0, 255, 0, 40))  # 绿色，透明度40/255
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 80), 1))
            painter.drawPath(search_path)

        # 绘制目标舰船（北约军标样式）
        tx, ty = self.env.target_pos
        cx = float(tx * scale + x_offset)
        cy = float(y_offset - ty * scale)
        
        # 舰艇尺寸 - 保持放大倍数
        ship_scale_factor = 10.0  # 放大倍数
        ship_size = float(self.env.target_length * scale * ship_scale_factor * 0.8)  # 调整大小以适应军标
        
        # 保存当前状态
        painter.save()
        
        # 移动到舰艇中心并旋转
        painter.translate(cx, cy)
        painter.rotate(-math.degrees(self.env.target_heading))
        
        # 设置舰艇颜色（灰色）
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 2))
        painter.setBrush(QtGui.QColor(150, 150, 150))
        
        # 绘制舰艇北约军标（水面舰艇符号）
        ship_path = QtGui.QPainterPath()
        
        # 舰艇主体（矩形）
        ship_path.addRect(-ship_size/2, -ship_size/6, ship_size, ship_size/3)
        
        # 舰艇前部（三角形）
        bow_points = [
            QtCore.QPointF(ship_size/2, 0),
            QtCore.QPointF(ship_size/2 + ship_size/4, -ship_size/6),
            QtCore.QPointF(ship_size/2 + ship_size/4, ship_size/6)
        ]
        ship_path.moveTo(ship_size/2, 0)
        for point in bow_points:
            ship_path.lineTo(point)
        ship_path.closeSubpath()
        
        # 绘制舰艇符号
        painter.drawPath(ship_path)
        
        # 绘制舰艇类型标识（水面作战单位）
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 1.5))
        # 使用QPointF替代直接传入浮点数
        painter.drawLine(
            QtCore.QPointF(-ship_size/3, -ship_size/3), 
            QtCore.QPointF(ship_size/3, ship_size/3)
        )
        painter.drawLine(
            QtCore.QPointF(-ship_size/3, ship_size/3), 
            QtCore.QPointF(ship_size/3, -ship_size/3)
        )
        
        # 恢复状态
        painter.restore()

        # 如果爆炸激活，绘制爆炸效果
        if self.env.explosion_active:
            explosion_progress = self.env.explosion_time / self.env.explosion_duration
            explosion_radius = self.env.explosion_max_radius * explosion_progress * scale
            
            # 创建径向渐变
            gradient = QtGui.QRadialGradient(cx, cy, explosion_radius)
            gradient.setColorAt(0, QtGui.QColor(255, 255, 200, 200))
            gradient.setColorAt(0.3, QtGui.QColor(255, 150, 0, 180))
            gradient.setColorAt(0.7, QtGui.QColor(255, 50, 0, 150))
            gradient.setColorAt(1, QtGui.QColor(100, 0, 0, 100))
            
            painter.setBrush(QtGui.QBrush(gradient))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(cx, cy), explosion_radius, explosion_radius)
        else:
            # 绘制导弹（北约军标样式）
            mx, my = self.env.m_pos
            hx = float(mx * scale + x_offset)
            hy = float(y_offset - my * scale)
            hd = self.env.heading
            
            # 保存当前状态
            painter.save()
            
            # 移动到导弹位置并旋转
            painter.translate(hx, hy)
            painter.rotate(-math.degrees(hd))
            
            # 设置画笔和画刷
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 2))
            painter.setBrush(QtGui.QColor(255, 50, 50))
            
            # 绘制导弹北约军标（箭头形状）
            size = 15
            missile_path = QtGui.QPainterPath()
            
            # 主体部分（矩形）
            missile_path.addRect(-size/2, -size/4, size, size/2)
            
            # 箭头部分（三角形）
            arrow_points = [
                QtCore.QPointF(size/2, 0),
                QtCore.QPointF(size, -size/2),
                QtCore.QPointF(size, size/2)
            ]
            missile_path.moveTo(size/2, 0)
            for point in arrow_points:
                missile_path.lineTo(point)
            missile_path.closeSubpath()
            
            # 绘制导弹符号
            painter.drawPath(missile_path)
            
            # 绘制导弹周围的圆圈（表示移动中的单位）
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 1, QtCore.Qt.DashLine))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(0, 0), size*1.2, size*1.2)
            
            # 恢复状态
            painter.restore()
        
        # 绘制信息文本
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 10))
        
        # 显示距离信息
        dist = np.linalg.norm(self.env.target_pos - self.env.m_pos)
        painter.drawText(10, 20, f"距离目标: {dist:.1f}米")
        painter.drawText(10, 40, f"导弹航向: {math.degrees(self.env.heading):.1f}°")
        painter.drawText(10, 60, f"导弹速度: {self.env.V_m:.1f}米/秒")
        
        # 显示操作提示
        painter.drawText(10, self.height() - 40, "使用左右方向键控制导弹航向")
        painter.drawText(10, self.height() - 20, "使用上下方向键控制导弹速度")

    def keyPressEvent(self, event):
        # 方向键控制航向变化率和速度
        rate = 1.0  # rad/s
        if event.key() == QtCore.Qt.Key_Left:
            self.env.manual_heading_rate = rate
        elif event.key() == QtCore.Qt.Key_Right:
            self.env.manual_heading_rate = -rate
        elif event.key() == QtCore.Qt.Key_Up:
            # 加速导弹（限制最大速度）
            self.env.V_m = min(self.env.V_m + self.env.V_m_step, self.env.V_m_max)
        elif event.key() == QtCore.Qt.Key_Down:
            # 减速导弹（限制最小速度）
            self.env.V_m = max(self.env.V_m - self.env.V_m_step, self.env.V_m_min)
        elif event.key() == QtCore.Qt.Key_R:  # 添加重置键
            self.env.reset()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            self.env.manual_heading_rate = 0.0
        else:
            super().keyReleaseEvent(event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = SimulationWindow()
    sys.exit(app.exec_())
