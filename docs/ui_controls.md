# RTS 交互与快捷键（单人模式 渲染器）

以下为渲染器中的基础交互说明，覆盖单位选中、移动、攻击、停止与UI控制按钮；适用于 `src/core/game_single_play.py` 单人模式入口。

## 单位选择
- 左键单击单位：将其加入当前选中集合；重复点击可切换选择。
- 左键拖拽：在主视图中拖出选择框，框选单位加入选中集合。
- 选中高亮：被选中单位在主视图绘制高亮框与血量条。

## 移动与攻击
- 右键移动（默认模式）：
  - 在主视图中右键点击地面，选中单位生成 `MoveCommand(actor, target_position)`，由环境的路径规划与移动系统执行。
- 攻击模式：
  - 底部按钮 `攻击模式` 切换开/关（`attack_mode`）。
  - 开启后，右键点击一个目标实体，为每个选中单位生成 `AttackCommand(actor, target, weapon=weapon_type)`。
  - `weapon_type` 依据单位的 `LauncherComponent` 的 `weapon_type` 字段；若不存在，则发起无武器类型的攻击指令（当前环境作简化处理）。
- 停止：
  - 底部按钮 `停止`，为所有选中单位生成 `StopCommand(actor)`，环境将清除移动目标并重置路径规划。

## UI 控制按钮
- `暂停/继续`：切换环境的暂停状态。
- `加速` / `减速`：调整仿真速度系数 `speed_scale`（范围约 `0.25`–`4.0`）。
- `攻击模式`：切换右键行为为攻击（点击目标实体触发）。
- `停止`：取消选中单位的当前移动目标。

## 新增界面与功能
- 小地图（右上角）：显示单位与视窗范围，红蓝表示所属阵营；位置随窗口大小自适应。
- 绿色障碍叠层：基于 `Map.compressed_map` 在主视窗与小地图绘制半透明绿色方块与网格线。
  - 默认行为：当环境已绑定 `default_map_json`（例如 `core/data/map.json`）时开启；仅指定地形图片时默认关闭。
  - 切换方式：
    - 顶部按钮“叠层开关”；
    - 快捷键 `O`（字母 O）。
- 右侧单位控制面板（位于小地图下方）：
  - “停止”：为所有选中单位发出 `StopCommand`。
  - “速度+ / 速度-”：为选中单位发送 `SetSpeedCommand`，速度在 `0~100` 之间约束。
  - “左转10° / 右转10°”：发送 `RotateHeadingCommand`，按二维向量旋转航向。
  - “传感器切换”：发送 `ToggleSensorCommand`，若首次则开启，再次点击在开/关之间切换。
  - “就近攻击”：发送 `AttackNearestCommand`，环境根据 `GameData.unit_owner` 自动寻找最近敌方单位并造成固定演示伤害。

## 状态与渲染
- 单位血量条：基于 `HealthComponent(current_health, max_health)` 绘制；攻击后即时反映。
- 小地图：按比例绘制单位位置与地形；不支持攻击/移动交互，仅作态势参考。
- 右下状态面板：选中单位的基础信息展示。

### 障碍叠层绘制说明
- 主视窗：`render/single_process.py:draw_terrain` 从 `Map.compressed_map` 读取网格，按缩放比在屏幕上叠加半透明绿色矩形与网格线。
- 小地图：`draw_minimap` 使用更轻的绿色色块显示障碍与可通行区域。
- 关闭叠层时仅显示背景地形图片，不再覆盖整屏绿色。

## 顶部菜单栏
- 位置与外观：位于屏幕上方，包含 `视图`、`控制`、`单位`、`帮助` 四个菜单，点击显示下拉项。
- 视图：`叠层开关`（同 `O` 键）、`重置视窗`（将视窗偏移重置为 `[0,0]`）。
- 控制：`暂停/继续`、`加速`、`减速`（通过 UI 动作交由外层处理）。
- 单位：`攻击模式`（右键攻击/移动切换）、`停止`、`速度+/-`、`左转10°/右转10°`、`传感器切换`、`就近攻击`。
- 帮助：`快捷键说明`（或按 `F1` 弹出/隐藏说明面板）。

### 快捷键
- `方向键`：移动视窗。
- `O`：切换障碍叠层显示。
- `F1`：打开/关闭快捷键说明面板。

## 环境指令处理（简化版）
- `MoveCommand`：更新单位 `MovementComponent.target_position`，触发 `PathfindingSystem` 规划路径并由 `MovementSystem` 执行移动。
- `AttackCommand`：直接对目标 `HealthComponent` 施加固定伤害（示例为 `10`）；后续将接入 `DeviceTable` 的射程、冷却与概率模型。
- `StopCommand`：清除 `MovementComponent.target_position`，并重置 `PathfindingComponent.current_goal`。

- `SetSpeedCommand`：约束并设置 `MovementComponent.speed`，范围 `0~100`（`sea_war_env.process_commands`）。
- `RotateHeadingCommand`：按度数旋转二维航向向量，并写回 `MovementComponent.heading`。
- `ToggleSensorCommand`：切换或设置 `SensorComponent.enabled`。
- `AttackNearestCommand`：依据 `GameData.unit_owner` 查找最近敌方单位并施加固定演示伤害。

### 菜单栏动作派发
- 渲染层统一使用 `process_ui_action(act)` 处理来自按钮与菜单的动作。
- 其中单位相关动作直接转换为相应环境指令（如 `StopCommand`、`SetSpeedCommand`、`RotateHeadingCommand`）。
- 其它如 `pause_toggle`、`speed_up/speed_down` 作为 UI 动作经外层消费并调节游戏速度与暂停状态。

## 已知限制与后续计划
- 主动攻击逻辑尚未实现：`AttackSystem.update` 当前不自动寻找目标；攻击由玩家指令触发。
- 武器与射程模型：暂未使用 `init.DeviceTableDict` 提供的 `range_min/range_max/cooldown` 等参数，后续将接入。
- 友军/敌军识别：环境将基于 `GameData.unit_owner` 扩展敌我判定，以避免对友军造成伤害。
- 快捷键：当前仅提供按钮交互，后续增加键盘快捷键（如 `A` 切换攻击模式、`S` 停止、`Space` 暂停）。

## 运行入口
- 单人模式入口：`python3 -m src.core.game_single_play --skip-menu`
- 渲染器文件：`src/render/single_process.py`
- 指令与玩家类：`src/ui/player.py`

### 示例运行
- `python3 run_game.py --skip-menu --terrain src/render/map/ground.png`
- 纯色背景：`python3 run_game.py --skip-menu --terrain "color:#1E90FF"`
- 若环境配置了 `default_map_json`，叠层默认开启；否则可用按钮或 `O` 键切换。