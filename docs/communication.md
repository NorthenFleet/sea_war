# 通信协议文档

## 概述

海战游戏使用TCP/IP协议进行客户端与服务器之间的通信，数据格式采用JSON。每条消息以换行符(`\n`)结尾，用于分隔多条消息。

## 消息类型

### 1. 命令消息 (客户端 -> 服务器)

客户端发送给服务器的命令消息，用于控制游戏实体。

```json
{
  "type": "command",
  "data": {
    "command_type": "move|attack|use_equipment",
    "actor": "entity_id",
    "target": [x, y, z] | "target_entity_id",
    "params": {
      // 命令特定参数
    }
  }
}