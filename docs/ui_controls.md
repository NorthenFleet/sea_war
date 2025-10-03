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

## 状态与渲染
- 单位血量条：基于 `HealthComponent(current_health, max_health)` 绘制；攻击后即时反映。
- 小地图：按比例绘制单位位置与地形；不支持攻击/移动交互，仅作态势参考。
- 右下状态面板：选中单位的基础信息展示。

## 环境指令处理（简化版）
- `MoveCommand`：更新单位 `MovementComponent.target_position`，触发 `PathfindingSystem` 规划路径并由 `MovementSystem` 执行移动。
- `AttackCommand`：直接对目标 `HealthComponent` 施加固定伤害（示例为 `10`）；后续将接入 `DeviceTable` 的射程、冷却与概率模型。
- `StopCommand`：清除 `MovementComponent.target_position`，并重置 `PathfindingComponent.current_goal`。

## 已知限制与后续计划
- 主动攻击逻辑尚未实现：`AttackSystem.update` 当前不自动寻找目标；攻击由玩家指令触发。
- 武器与射程模型：暂未使用 `init.DeviceTableDict` 提供的 `range_min/range_max/cooldown` 等参数，后续将接入。
- 友军/敌军识别：环境将基于 `GameData.unit_owner` 扩展敌我判定，以避免对友军造成伤害。
- 快捷键：当前仅提供按钮交互，后续增加键盘快捷键（如 `A` 切换攻击模式、`S` 停止、`Space` 暂停）。

## 运行入口
- 单人模式入口：`python3 -m src.core.game_single_play --skip-menu`
- 渲染器文件：`src/render/single_process.py`
- 指令与玩家类：`src/ui/player.py`